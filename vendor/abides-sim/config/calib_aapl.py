# calib_aapl — derivative of rmsc03 for Phase 5B AAPL calibration.
#
# 4D parameter exposure to fit synthetic LOB to real AAPL stylized facts:
#   --num-noise          (replaces hardcoded 5000)
#   --fund-vol           (already CLI in rmsc03 — keep default 1e-8 → grid 1e-5/1e-4/1e-3)
#   --mm-aggressiveness  ∈ {timid, balanced, aggressive}
#       maps to (spread_alpha, mm_pov, mm_skew_beta) presets
#   --num-momentum       (replaces hardcoded 25; Phase 5B+ knob to induce vol clustering)
#
# All other rmsc03 logic preserved.

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.OrderBookImbalanceAgent import OrderBookImbalanceAgent
from agent.examples.HerderAgent import HerderAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from model.LatencyModel import LatencyModel

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for RMSC03 config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='11:30:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
                    help='Volatility of fundamental time series.'
                    )

# === Phase 5B calibration knobs ===
parser.add_argument('--num-noise',
                    type=int,
                    default=5000,
                    help='Number of noise traders (Phase 5B calibration knob).'
                    )
parser.add_argument('--mm-aggressiveness',
                    type=str,
                    default='balanced',
                    choices=['timid', 'balanced', 'aggressive'],
                    help='Market-maker aggressiveness preset (timid | balanced | aggressive). '
                         'Overrides mm-spread-alpha / mm-pov / mm-skew-beta.'
                    )
parser.add_argument('--num-momentum',
                    type=int,
                    default=25,
                    help='Number of MomentumAgents (Phase 5B+ knob to induce vol clustering; rmsc03 default = 25).'
                    )

# === Phase 5B+ Stage 3 calibration knobs ===
parser.add_argument('--num-obi',
                    type=int,
                    default=0,
                    help='Number of OrderBookImbalanceAgents (mean-reverting; Brock-Hommes mix).'
                    )
parser.add_argument('--num-herder',
                    type=int,
                    default=0,
                    help='Number of HerderAgents (trend-following; Lux 1998 chartist).'
                    )
parser.add_argument('--herder-threshold-bps',
                    type=float,
                    default=None,
                    help='HerderAgent entry threshold in bps. From Step 0 data calibration. Required if --num-herder > 0.'
                    )
parser.add_argument('--herder-max-size',
                    type=int,
                    default=None,
                    help='HerderAgent max order size. From Step 0 data calibration. Required if --num-herder > 0.'
                    )
parser.add_argument('--herder-lookback-min-secs',
                    type=float,
                    default=3.0,
                    help='Min lookback for herder population (heterogeneous, CHAD lit).'
                    )
parser.add_argument('--herder-lookback-max-secs',
                    type=float,
                    default=30.0,
                    help='Max lookback for herder population.'
                    )
parser.add_argument('--herder-position-cap',
                    type=int,
                    default=50,
                    help='HerderAgent position cap (Stage 5+ tuning knob for long-episode cascade control).'
                    )
parser.add_argument('--herder-tolerance-ticks',
                    type=int,
                    default=5,
                    help='HerderAgent marketable limit tolerance in ticks.'
                    )

args, remaining_args = parser.parse_known_args()

# Apply mm-aggressiveness preset (overrides individual mm-* args)
_MM_PRESETS = {
    'timid':      {'spread_alpha': 1.5, 'pov': 0.01,  'skew_beta': 0.0},
    'balanced':   {'spread_alpha': 0.75, 'pov': 0.025, 'skew_beta': 0.0},
    'aggressive': {'spread_alpha': 0.3,  'pov': 0.05,  'skew_beta': 0.5},
}
_preset = _MM_PRESETS[args.mm_aggressiveness]
args.mm_spread_alpha = _preset['spread_alpha']
args.mm_pov = _preset['pov']
args.mm_skew_beta = _preset['skew_beta']
print(f"[calib_aapl] Using preset '{args.mm_aggressiveness}': {_preset}")
print(f"[calib_aapl] num_noise = {args.num_noise}, fund_vol = {args.fund_vol}, num_momentum = {args.num_momentum}")
print(f"[calib_aapl] num_obi = {args.num_obi}, num_herder = {args.num_herder}")

# Validate herder params (required if --num-herder > 0)
if args.num_herder > 0:
    if args.herder_threshold_bps is None or args.herder_max_size is None:
        parser.error("--num-herder > 0 requires --herder-threshold-bps and --herder-max-size "
                     "(from Step 0 data calibration)")
    print(f"[calib_aapl] herder_threshold_bps = {args.herder_threshold_bps}, "
          f"herder_max_size = {args.herder_max_size}, "
          f"lookback ∈ [{args.herder_lookback_min_secs}, {args.herder_lookback_max_secs}]s")

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = None
book_freq = 0

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 1e5
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Noise Agents
num_noise = args.num_noise   # Phase 5B calibration knob (was hardcoded 5000)
noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")  # These times needed for distribution of arrival times
                                                                # of Noise Agents
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

# 3) Value Agents
num_value = 100
agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=kappa,
                          lambda_a=lambda_a,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# 4) Market Maker Agents

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [(args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
             (args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size)
             ]

num_mm_agents = len(mm_params)
mm_cancel_limit_delay = 50  # 50 nanoseconds

agents.extend([AdaptiveMarketMakerAgent(id=j,
                                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                type='AdaptivePOVMarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                pov=mm_params[idx][1],
                                min_order_size=mm_params[idx][4],
                                window_size=mm_params[idx][0],
                                num_ticks=mm_params[idx][2],
                                wake_up_freq=mm_params[idx][3],
                                cancel_limit_delay=mm_cancel_limit_delay,
                                skew_beta=args.mm_skew_beta,
                                level_spacing=args.mm_level_spacing,
                                spread_alpha=args.mm_spread_alpha,
                                backstop_quantity=args.mm_backstop_quantity,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('POVMarketMakerAgent')


# 5) Momentum Agents
num_momentum_agents = args.num_momentum   # Phase 5B+ calibration knob (was hardcoded 25)

agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='1s',  # Phase 5B+: was '20s' (50 ticks × 20s = 16.7min warmup, longer than calib sim).
                                                  # 1s → ~50s warmup so momentum agents actually trade within 5-min calibration window.
                             log_orders=log_orders,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

# 5b) OrderBookImbalance Agents (Phase 5B+ Stage 3 — mean-reverting, Brock-Hommes mix)
num_obi_agents = args.num_obi
if num_obi_agents > 0:
    agents.extend([OrderBookImbalanceAgent(id=j,
                                            name="OBI_AGENT_{}".format(j),
                                            type="OrderBookImbalanceAgent",
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            log_orders=log_orders,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                       dtype='uint64')))
                   for j in range(agent_count, agent_count + num_obi_agents)])
    agent_count += num_obi_agents
    agent_types.extend("OrderBookImbalanceAgent")

# 5c) Herder Agents (Phase 5B+ Stage 3 — Lux 1998 chartist; heterogeneous lookback per CHAD lit)
num_herder_agents = args.num_herder
if num_herder_agents > 0:
    # Each herder samples its own lookback uniformly in [min, max] sec — CHAD lit recommendation
    # to avoid artificial resonance from synchronous agents.
    _herder_rng = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
    herder_lookbacks = _herder_rng.uniform(args.herder_lookback_min_secs,
                                            args.herder_lookback_max_secs,
                                            size=num_herder_agents)
    print(f"[calib_aapl] Herder lookbacks: median={np.median(herder_lookbacks):.1f}s, "
          f"min={herder_lookbacks.min():.1f}s, max={herder_lookbacks.max():.1f}s")
    agents.extend([HerderAgent(id=j,
                                name="HERDER_AGENT_{}".format(j),
                                type="HerderAgent",
                                symbol=symbol,
                                starting_cash=starting_cash,
                                lookback_window_secs=float(herder_lookbacks[idx]),
                                entry_threshold_bps=args.herder_threshold_bps,
                                max_size=args.herder_max_size,
                                position_cap=args.herder_position_cap,
                                tolerance_ticks=args.herder_tolerance_ticks,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                           dtype='uint64')))
                   for idx, j in enumerate(range(agent_count, agent_count + num_herder_agents))])
    agent_count += num_herder_agents
    agent_types.extend("HerderAgent")

# 6) Execution Agent

trade = True if args.execution_agents else False

#### Participation of Volume Agent parameters

pov_agent_start_time = mkt_open + pd.to_timedelta('00:30:00')
pov_agent_end_time = mkt_close - pd.to_timedelta('00:30:00')
pov_proportion_of_volume = args.execution_pov
pov_quantity = 12e5
pov_frequency = '1min'
pov_direction = "BUY"

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=pov_agent_start_time,
                              end_time=pov_agent_end_time,
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade,
                              log_orders=True,  # needed for plots so conflicts with others
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))

execution_agents = [pov_agent]
agents.extend(execution_agents)
agent_types.extend("ExecutionAgent")
agent_count += 1


########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("RMSC03 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
