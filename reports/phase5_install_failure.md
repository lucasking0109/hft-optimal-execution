# Phase 5 ABIDES 安裝失敗報告

**日期**：2026-05-03
**結論**：`abides-jpmc-public` 在現代 Python 環境（M1/M2 Mac、Python 3.9-3.11）**無法乾淨安裝**。
**處理**：依 NO Silent Fallback 原則 **STOP**，由 Lucas 決定下一步路徑。

---

## 失敗根因

ABIDES 在 2021 年發佈後 **依賴鎖死了那年的版本**，到 2026 年部分套件版本已從 PyPI 下架或 metadata 無法被現代 pip 解析。

### 具體錯誤紀錄

#### 嘗試 1：直接 PyPI 安裝
```bash
$ uv pip install abides-markets
× No solution found: abides-markets was not found in the package registry
```
**ABIDES 沒有發佈到 PyPI**。

#### 嘗試 2：clone JPM 官方 repo + `pip install -r requirements.txt`
```bash
$ git clone https://github.com/jpmorganchase/abides-jpmc-public
$ cd abides-jpmc-public && pip install -r requirements.txt
```

**錯誤 A** — `gym==0.18.0` 的 metadata 無效：
```
WARNING: Ignoring version 0.18.0 of gym since it has invalid metadata:
Requested gym==0.18.0 from ...gym-0.18.0.tar.gz has invalid metadata:
    opencv-python>=3.
                 ~~~^
Please use pip<24.1 if you need to use this version.
ERROR: No matching distribution found for gym==0.18.0
```

#### 嘗試 3：降 pip 到 24.0 後重試
```bash
$ pip install 'pip<24.1'
$ pip install -r requirements.txt
```

**錯誤 B** — `pomegranate==0.14.5` 在 PyPI 上根本不存在：
```
ERROR: Could not find a version that satisfies the requirement pomegranate==0.14.5
(versions available: ..., 0.14.4, 0.14.7, 0.14.8, ...)
```

PyPI 版本紀錄顯示 0.14.5 / 0.14.6 從未發佈或已被撤下。

### 即使這兩個錯誤解了，下一波會炸

`requirements.txt` 接下來還鎖：
- `numpy==1.22.0` — 與我們現有 polars / scipy / plotly 不相容
- `pandas==1.2.4` — API 跟我們現有程式不同
- `ray[rllib]==1.7.0` — Python 3.11 沒有 wheel，從原始碼編 ray 1.7 在 M1/M2 機器上會失敗
- `scipy==1.10.0` — 與我們 Phase 3 衝擊校準的 scipy>=1.13 衝突

**就算強行裝起來，主專案 venv 將被破壞**：Phase 0-4 的 polars / plotly / streamlit / 我們的 viz 模組都會壞掉。

---

## 為什麼不偷偷改用「自寫簡易 LOB simulator」

這是 plan 明確禁止的 silent fallback。Lucas 親口：「我寧願你直接出現錯誤，不要讓我們的整個研究跟開發的過程中出現任何 fallback... 我怕你 fallback 了其實我都不知道你 fallback 了，這樣子我們的最終效果其實沒那麼好」。

簡易 simulator 跟 ABIDES 不等效：
- ABIDES 模擬 5 種 agent type（noise / momentum / value / market maker / adversarial），交互產生有結構的 LOB
- 自寫的版本只能模擬 noise + momentum，**不會產生真實的 stylized facts**（fat tails、volatility clustering、leverage effect）
- 用差版 simulator 訓練 RL → agent 學到一堆假規則 → 實戰失敗

---

## 給你選擇的下一步（[1] / [2] / [3] / [4]）

### [1] **Docker 容器跑 ABIDES**
**做法**：建一個 ABIDES-only Docker container（dockerfile 用 Python 3.9 + 鎖定 2021 版本依賴），透過檔案共享（volumes）讓主專案讀取它產出的合成 LOB parquet。
- ✅ 主 venv 不污染
- ✅ ABIDES 真實版本可運作
- ❌ 你需要先安裝 Docker Desktop
- ❌ 整體流程：研究 → 寫 Docker driver → debug 容器 ↔ host 之間的 IO，多 1-2 週工程
- ⚠️ ABIDES 的 ray[rllib] 1.7.0 在 ARM Linux 容器仍可能編不過

### [2] **換成現代維護的 LOB simulator**
**候選**：
- **`mbt_gym`** (https://github.com/JJJerome/mbt_gym) — Cambridge 學術界，2022 後仍在維護，原生 Gymnasium 介面，Python 3.10+
- **`abides-research`** — 學界 fork，可能有人修復過依賴
- **MIT MIT-AI4HFT** (https://github.com/marcofavorito/abides) — 部分維護

⚠️ **不等同 ABIDES**，論文 paper 比較少，但相容現代 Python。我可以花半天調研後給你具體 recommendation。

### [3] **跳過 RL，深化現有方法**
做法：Phase 6 改用：
- **Dynamic Programming** 解 multi-period optimal execution（Bertsimas-Lo, Almgren-Chriss extension）
- **CVXPY 凸優化** 解約束下最佳排程
- **Statistical learning 替代 RL**：用 5 天 tick + 文獻 features 訓練 supervised model 預測 next-step optimal action

✅ 不需要模擬器，跟你現有資料相容
✅ 可解釋性高
❌ 失去 RL 的非線性 conditional 策略能力
❌ 偏離原計畫

### [4] **暫停 Phase 5，先把 Phase 7 dashboard 做完**
做法：把目前 Phase 0-4 成果整合到 strategy evaluation dashboard，做出可 demo 的 MVP，**之後再處理 ABIDES 問題**（給你時間想清楚要走哪條路）。

✅ 確保已有的成果不被卡住
✅ 多一份 demo 材料
❌ 延後 RL 主目標

---

## 我的建議排序

我認為最務實的是 **[2] mbt_gym + [4] 先做 dashboard**：
1. Phase 7 dashboard 不需要等 RL，可以先做（鎖定 Phase 0-4 成果）
2. 同時花時間調研 mbt_gym 是否真的能用
3. 如果 mbt_gym 行得通 → 走 Phase 6 RL；不行就走 [3] DP/CVXPY 路線

但我 **不會自己決定**。請你選。
