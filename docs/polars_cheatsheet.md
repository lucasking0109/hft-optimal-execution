# Polars 速查表（給 pandas 用戶）

Polars 用 Rust 寫，本質上是「**會自動並行化的 DataFrame**」。API 概念跟 pandas 像，但有兩個不同點：
- 用 **expressions**（`pl.col("X")` ）描述操作，比較像 SQL
- 多數 column 操作要包在 `select()` / `with_columns()` / `filter()` 裡

下面列我們專案會用到的常見操作對照。

---

## 載入

| pandas | polars |
|---|---|
| `pd.read_csv("x.csv")` | `pl.read_csv("x.csv")` |
| `pd.read_parquet("x.parquet")` | `pl.read_parquet("x.parquet")` |
| `df.head()` | `df.head()` ✅ 一樣 |
| `df.shape` | `df.shape` ✅ 一樣 |
| `df.columns` | `df.columns` ✅ 一樣 |

## 篩選

| pandas | polars |
|---|---|
| `df[df["price"] > 100]` | `df.filter(pl.col("price") > 100)` |
| `df[(df["a"] > 1) & (df["b"] < 5)]` | `df.filter((pl.col("a") > 1) & (pl.col("b") < 5))` |
| `df[df["x"].isin([1,2])]` | `df.filter(pl.col("x").is_in([1, 2]))` |
| `df[df["x"].isnull()]` | `df.filter(pl.col("x").is_null())` |

## 選欄位 / 改欄位

| pandas | polars |
|---|---|
| `df["price"]` | `df["price"]`（單欄）or `df.select("price")` |
| `df[["a","b"]]` | `df.select(["a", "b"])` |
| `df["c"] = df["a"] + df["b"]` | `df = df.with_columns((pl.col("a") + pl.col("b")).alias("c"))` |
| `df["c"] = df["a"].apply(fn)` | `df.with_columns(pl.col("a").map_elements(fn, return_dtype=pl.Int64).alias("c"))` |
| `df.rename(columns={"a":"x"})` | `df.rename({"a": "x"})` |
| `df.drop(columns=["a"])` | `df.drop("a")` |

## 聚合

| pandas | polars |
|---|---|
| `df["price"].mean()` | `df["price"].mean()`（純值） or `df.select(pl.col("price").mean())` |
| `df.groupby("x")["y"].mean()` | `df.group_by("x").agg(pl.col("y").mean())` |
| `df.groupby(["a","b"]).agg({"x":"mean","y":"sum"})` | `df.group_by(["a","b"]).agg(pl.col("x").mean(), pl.col("y").sum())` |
| `df["x"].cumsum()` | `df.with_columns(pl.col("x").cum_sum().alias("cumx"))` |
| `df["x"].rolling(10).mean()` | `df.with_columns(pl.col("x").rolling_mean(10).alias("rm"))` |

## 排序、Top N

| pandas | polars |
|---|---|
| `df.sort_values("x")` | `df.sort("x")` |
| `df.sort_values("x", ascending=False)` | `df.sort("x", descending=True)` |
| `df.nlargest(5, "x")` | `df.top_k(5, by="x")` 或 `df.sort("x", descending=True).head(5)` |

## Join

| pandas | polars |
|---|---|
| `pd.merge(a, b, on="k")` | `a.join(b, on="k")`（預設 inner） |
| `pd.merge(a, b, on="k", how="left")` | `a.join(b, on="k", how="left")` |
| `pd.merge_asof(a, b, on="t")` | `a.join_asof(b, on="t")` |

## Lazy（大資料時用）

```python
# 對 5GB CSV 不要 read_csv 一次吃全
lazy = pl.scan_csv("big.csv")
result = (
    lazy.filter(pl.col("ticker") == "AAPL")
    .group_by("Exchange")
    .agg(pl.col("Price").mean())
    .collect()  # 這時候才真的執行
)
```

`scan_csv` / `scan_parquet` 配 `.collect()` 通常比 eager 快 2-5 倍。

## 常見坑

1. **`df["x"]` 在 polars 是 Series，但操作回傳 expr 時要 wrap 在 `select` / `with_columns` 裡**：
   ```python
   # 錯
   df.col("price").cast(pl.Float64)
   # 對
   df.with_columns(pl.col("price").cast(pl.Float64))
   ```

2. **`apply` → `map_elements` 並要指定 return_dtype**（不指定會炸出 PolarsInefficientMapWarning）。

3. **string 比較沒有 `==` chain，用 `&` / `|` 加括號**：
   ```python
   df.filter((pl.col("a") == "x") & (pl.col("b") > 0))
   ```

4. **Null 不是 NaN**。Polars 把 missing 視為 Null（SQL 風格），不是 NaN。`is_null()` 不是 `isna()`。

5. **不要混用 pandas / polars 在同一管線**。轉換用 `df.to_pandas()` / `pl.from_pandas()`。

---

## 我們專案最常見的 pattern

```python
# 載入 AAPL 一天 tick
import polars as pl
from hft.data import load_eq_taq

df = load_eq_taq("AAPL", "20200113")

# 只看 trade
trades = df.filter(pl.col("EventType") == "TRADE")

# 算 VWAP
vwap = (trades["Price"] * trades["Quantity"]).sum() / trades["Quantity"].sum()

# 算每 5 分鐘量
trades_with_bucket = trades.with_columns(
    pl.col("Timestamp").str.slice(0, 5).alias("hhmm")  # HH:MM
)
five_min_volume = trades_with_bucket.group_by("hhmm").agg(pl.col("Quantity").sum())
```

如果你寫不出某個 query：**直接呈報、寫 issue**（per NO Silent Fallback 原則，不准偷偷 fallback pandas）。我會幫你補進這份 cheatsheet。
