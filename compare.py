
# v6.2 — Reliability patch: schema-lock from training, clean symbol labels, no leakage in reg validation,
# equities-only option, m1_reg training enforced, optional Gaussian fallback for 5D prob.
# Author: M365 Copilot for Singh, Karanveer
# Timezone normalization: Asia/Kolkata (IST)

import os, glob, json, time, sys, re, math, importlib, concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd

# Try pyarrow for parquet I/O
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PA_OK = True
except Exception:
    _PA_OK = False

# ===================== DEFAULTS / PATHS =====================
DATA_DIR_DEFAULT = r"C:\\Users\\karanvsi\\Desktop\\Pycharm\\Cache\\cache_daily_new"
PANEL_OUT = None
WATCHLIST_OUT = None
STATUS_PATH = None
META_PATH = None
LOG_DIR = None
LOAD_ERRORS_LOG = None
QUARANTINE_LIST = None
FEATURES_SCHEMA_PATH = None  # <out-dir>/features_train.json
OOS_REPORT_PATH = None       # <out-dir>/oos_report.json

# ===================== MODEL / CV =====================
GLOBAL_SEED = 42
FOLDS = 8
EMBARGO_DAYS = 3

# Estimators (can override via CLI)
N_EST_ALL = 3200
N_EST_1D = N_EST_ALL
N_EST_3D = N_EST_ALL
N_EST_5D = N_EST_ALL

# Early stopping
EARLY_STOPPING_ROUNDS = 200
LEARNING_RATE = 0.01
MAX_DEPTH_ALL = 6

# Filters for watchlist
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

# Chunking
CHUNK_SIZE = 1200

# Probability std options for 5D fallback
PROB_STD_METHOD = "residual"  # "residual" | "symbol_hist" | "cross" | "none"
PROB_STD_WINDOW = 252
PROB_STD_MIN_ROWS = 60

# 1D classification margin (fixed)
CLS_MARGIN = 0.08

# Monotone constraints (optional via CLI)
USE_MONOTONE = False
MONOTONE_MAP_DEFAULT = {
    "D_atr14_to_close_pct": +1,
    "D_cpr_width_pct": -1,
}

np.random.seed(GLOBAL_SEED)

# ===================== TZ helpers =====================
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def ensure_kolkata_tz(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return ts.dt.tz_convert("Asia/Kolkata")
    except Exception:
        return pd.to_datetime(series, errors="coerce").dt.tz_localize("Asia/Kolkata")

def _ensure_ts_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = ensure_kolkata_tz(df["timestamp"])
    return df

# ===================== Status / Progress =====================
def write_status(phase: str, note: str = ""):
    rec = {"ts": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="seconds"),
           "phase": phase, "note": note}
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass

class ProgressETA:
    def __init__(self, total:int, label:str=""):
        self.total = max(1, int(total)); self.label = label
        self.start = time.perf_counter(); self.done = 0; self._last = ""
    def tick(self, note:str=""):
        self.done += 1
        elapsed = max(1e-6, time.perf_counter() - self.start)
        rate = self.done / elapsed; remain = max(0, self.total - self.done)
        eta_s = int(remain / rate) if rate > 0 else 0
        m, s = divmod(eta_s, 60); h, m = divmod(m, 60)
        eta = f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"
        pct = 100 * self.done / self.total
        msg = f"[{self.label}] {self.done}/{self.total} ({pct:5.1f}%) ETA {eta}"
        if note: msg += f" {note}"
        if msg != self._last:
            self._last = msg
            print(msg)

# ===================== Paths setup =====================
def setup_paths(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    global PANEL_OUT, WATCHLIST_OUT, STATUS_PATH, META_PATH
    global LOG_DIR, LOAD_ERRORS_LOG, QUARANTINE_LIST, FEATURES_SCHEMA_PATH, OOS_REPORT_PATH
    PANEL_OUT = str(out / "panel_cache.parquet")
    WATCHLIST_OUT = str(out / "watchlist_model_next_1_3_5d.csv")
    STATUS_PATH = str(out / "status.json")
    META_PATH = str(out / "model_meta.json")
    FEATURES_SCHEMA_PATH = str(out / "features_train.json")
    OOS_REPORT_PATH = str(out / "oos_report.json")
    LOG_DIR = out / "logs"; LOG_DIR.mkdir(exist_ok=True)
    LOAD_ERRORS_LOG = LOG_DIR / "load_errors.csv"
    QUARANTINE_LIST = out / "quarantine_files.txt"

# ===================== IO helpers =====================
def _strict_file_list(data_dir: str,
                      symbols_like: Optional[str],
                      limit_files: Optional[int],
                      accept_any_daily: bool=False) -> List[Path]:
    paths: List[str] = []
    paths += glob.glob(os.path.join(data_dir, "*_daily.parquet"))
    paths += glob.glob(os.path.join(data_dir, "*_daily.csv"))
    if str(accept_any_daily).lower() in ("true","1","yes","y","t"):
        paths += glob.glob(os.path.join(data_dir, "*.parquet"))
        paths += glob.glob(os.path.join(data_dir, "*.csv"))
    paths = sorted(set(paths))
    if symbols_like:
        pat = re.compile(symbols_like)
        filtered = []
        for p in paths:
            sym = _derive_symbol_name(Path(p))
            if pat.search(sym): filtered.append(p)
        paths = filtered or paths
    if limit_files and limit_files > 0:
        paths = paths[:limit_files]
    return [Path(p) for p in paths]

def _log_load_error(sym: str, filename: str, error: str):
    rec = {"symbol": sym, "file": filename, "error": error,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        df = pd.DataFrame([rec])
        mode = "a" if Path(LOAD_ERRORS_LOG).exists() else "w"
        df.to_csv(LOAD_ERRORS_LOG, mode=mode,
                  header=not Path(LOAD_ERRORS_LOG).exists(), index=False)
    except Exception:
        pass
    with open(QUARANTINE_LIST, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")

def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    seen: Dict[str,int] = {}
    new_cols: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 1; new_cols.append(c)
        else:
            k = seen[c]; seen[c] = k + 1
            new_cols.append(f"{c}__dup{k}")
    df = df.copy(); df.columns = new_cols
    return df

# ===== CLEAN, DETERMINISTIC SYMBOL STRIPPING (fix) =====
def _derive_symbol_name(p: Path) -> str:
    base = p.name
    for suff in ("_daily.parquet", "_daily.csv", ".parquet", ".csv"):
        if base.endswith(suff):
            base = base[:-len(suff)]
            break
    return base.strip()

def _clean_symbol_label(label: str) -> str:
    s = str(label).strip()
    for suff in ("_daily.parquet", "_daily.csv"):
        if s.endswith(suff):
            s = s[:-len(suff)]
    return s

def load_one(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
            except ValueError:
                df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Load failed: {e}")
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        else:
            raise RuntimeError("'timestamp' column missing")
    if not (is_datetime64_any_dtype(df["timestamp"]) or is_datetime64tz_dtype(df["timestamp"])):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = _ensure_ts_ist(df)
    df = (df.dropna(subset=["timestamp"]).sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True))
    df = _ensure_unique_columns(df)
    return df

# ===================== Targets & features =====================
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in (1,3,5):
        df[f"ret_{h}d_close_pct"] = (df["close"].shift(-h) / df["close"] - 1) * 100
        hi = df["high"].shift(-1).rolling(h, min_periods=1).max()
        lo = df["low"].shift(-1).rolling(h, min_periods=1).min()
        df[f"mfe_{h}d_pct"] = (hi / df["close"] - 1) * 100
        df[f"mae_{h}d_pct"] = (lo / df["close"] - 1) * 100
    df["ret_1d_sign"] = np.sign(df["ret_1d_close_pct"])
    return df

def add_lags(df: pd.DataFrame, cols: List[str], lags: Tuple[int,int]=(1,2)):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            for L in lags:
                df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def _unify_categorical(df: pd.DataFrame, base_name: str) -> pd.Series:
    cols = [c for c in df.columns if c == base_name or c.startswith(base_name + "__dup")]
    if not cols:
        return pd.Series(index=df.index, dtype="object")
    s = pd.Series(index=df.index, dtype="object")
    for c in cols:
        sc = df[c].astype("string")
        s = sc.where(sc.notna(), s)
    return s

EXCLUDE_D_FEATURES = set()
LAG_FEATURES = [
    "D_rsi14_lag1","D_rsi14_lag2",
    "D_adx14_lag1","D_adx14_lag2",
    "D_ema20_angle_deg_lag1","D_ema20_angle_deg_lag2",
    "D_obv_slope_lag1","D_obv_slope_lag2",
]
CPR_YDAY = [f"CPR_Yday_{x}" for x in ("Above","Below","Inside","Overlap")]
CPR_TMR  = [f"CPR_Tmr_{x}"  for x in ("Above","Below","Inside","Overlap")]
STRUCT_ONEHOT = ["Struct_uptrend","Struct_downtrend","Struct_range"]
DAYTYPE_ONEHOT = ["DayType_bullish","DayType_bearish","DayType_inside"]

def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    n = len(y); idx = np.arange(n, dtype=float)
    def _one(i):
        lo = i - window + 1
        if lo < 0: lo = 0
        xs = idx[lo:i+1]; ys = y.iloc[lo:i+1]
        xs = xs - np.nanmean(xs); ys = ys - np.nanmean(ys)
        denom = np.dot(xs, xs)
        if denom <= 0 or np.isnan(denom): return np.nan
        return float(np.dot(xs, ys) / denom)
    return pd.Series([_one(i) for i in range(n)], index=y.index)

def discover_daily_features(df, exclude=None):
    exclude = set(exclude or [])
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("D_") and c not in exclude]
    return sorted(cols)

def featureize(df: pd.DataFrame):
    # Base auto
    base_auto = discover_daily_features(df, exclude=EXCLUDE_D_FEATURES)
    # Lags
    df = add_lags(df, ["D_rsi14","D_adx14","D_ema20_angle_deg","D_obv_slope"], lags=(1,2))
    # CPR categorical -> unified one-hots
    yday_unified = _unify_categorical(df, "D_cpr_vs_yday")
    tmr_unified  = _unify_categorical(df, "D_tmr_cpr_vs_today")
    if yday_unified.notna().any():
        df = df.drop(columns=[c for c in df.columns if c == "D_cpr_vs_yday" or c.startswith("D_cpr_vs_yday__dup")], errors="ignore")
        df["D_cpr_vs_yday_unified"] = yday_unified
        df = pd.get_dummies(df, columns=["D_cpr_vs_yday_unified"], prefix="CPR_Yday")
    if tmr_unified.notna().any():
        df = df.drop(columns=[c for c in df.columns if c == "D_tmr_cpr_vs_today" or c.startswith("D_tmr_cpr_vs_today__dup")], errors="ignore")
        df["D_tmr_cpr_vs_today_unified"] = tmr_unified
        df = pd.get_dummies(df, columns=["D_tmr_cpr_vs_today_unified"], prefix="CPR_Tmr")
    for col in CPR_YDAY:
        if col not in df.columns: df[col] = 0
    for col in CPR_TMR:
        if col not in df.columns: df[col] = 0
    # Structure & DayType one-hots (if present)
    if "D_structure_trend" in df.columns:
        df["D_structure_trend"] = df["D_structure_trend"].astype("string")
        df = pd.get_dummies(df, columns=["D_structure_trend"], prefix="Struct")
        for col in STRUCT_ONEHOT:
            if col not in df.columns: df[col] = 0
    if "D_day_type" in df.columns:
        df["D_day_type"] = df["D_day_type"].astype("string")
        df = pd.get_dummies(df, columns=["D_day_type"], prefix="DayType")
        for col in DAYTYPE_ONEHOT:
            if col not in df.columns: df[col] = 0

    # Interactions & rolling (safe)
    rsi14 = pd.to_numeric(df.get("D_rsi14", np.nan), errors="coerce")
    rsi7  = pd.to_numeric(df.get("D_rsi7",  np.nan), errors="coerce")
    obvs  = pd.to_numeric(df.get("D_obv_slope", np.nan), errors="coerce")
    adx14 = pd.to_numeric(df.get("D_adx14", np.nan), errors="coerce")
    atr14 = pd.to_numeric(df.get("D_atr14", np.nan), errors="coerce")
    close = pd.to_numeric(df.get("close",   np.nan), errors="coerce")
    macd_hist = pd.to_numeric(df.get("D_macd_hist", np.nan), errors="coerce")
    cprw = pd.to_numeric(df.get("D_cpr_width_pct", np.nan), errors="coerce").abs()
    df["D_rsi14_obv_x"] = rsi14 * obvs
    if "D_rsi7" in df.columns:
        df["D_rsi7_obv_x"] = rsi7 * obvs
    df["D_atr14_to_close_pct"] = (atr14 / close).replace([np.inf, -np.inf], np.nan) * 100.0

    # curated non-linear combos
    df["X_rsi14_adx14"] = rsi14 * adx14
    df["X_cprw_atr_pct"] = cprw * df["D_atr14_to_close_pct"]
    trend_code = pd.to_numeric(df.get("D_structure_trend_code", np.nan), errors="coerce")
    df["X_trend_atr_pct"] = trend_code * df["D_atr14_to_close_pct"]
    df["X_rsi_cross_strength"] = (rsi7 - rsi14) * adx14
    df["X_macd_nonlin"] = macd_hist * rsi14 / 50.0
    df["X_adx_sqr"] = (adx14 ** 2) / 100.0

    # ensure targets exist
    if "ret_1d_close_pct" not in df.columns or "ret_5d_close_pct" not in df.columns:
        df = add_targets(df)

    # safe rolling slope
    df["D_close_roll_slope_20"] = _rolling_slope(df["close"], window=20)
    df["D_close_roll_slope_50"] = _rolling_slope(df["close"], window=50)

    # past-returns dispersion (safe)
    daily_ret = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100.0
    df["D_ret_5d_pastret"]  = daily_ret.rolling(5, min_periods=5).sum()
    df["D_ret_5d_roll_std"] = df["D_ret_5d_pastret"].rolling(50, min_periods=10).std()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    feats = (base_auto + LAG_FEATURES + CPR_YDAY + CPR_TMR + STRUCT_ONEHOT + DAYTYPE_ONEHOT + [
        "D_rsi14_obv_x","D_rsi7_obv_x","D_atr14_to_close_pct",
        "D_ret_5d_roll_std","D_close_roll_slope_20","D_close_roll_slope_50",
        "X_rsi14_adx14","X_cprw_atr_pct","X_trend_atr_pct",
        "X_rsi_cross_strength","X_macd_nonlin","X_adx_sqr",
    ])
    # fill one-hots to int & numeric-like to float
    for c in feats:
        if c not in df.columns:
            df[c] = 0 if (c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
                          or c.startswith("Struct_") or c.startswith("DayType_")) else np.nan
    return df, feats

# ===================== Bias scores =====================
def score_bias(df: pd.DataFrame) -> pd.DataFrame:
    col = lambda c: df[c] if c in df.columns else pd.Series([np.nan]*len(df))
    golden    = col("D_golden_regime").fillna(False).astype(bool) if "D_golden_regime" in df.columns else pd.Series(False, index=df.index)
    ema_stack = col("D_ema_stack_20_50_100").fillna(False).astype(bool) if "D_ema_stack_20_50_100" in df.columns else pd.Series(False, index=df.index)
    rsi14     = pd.to_numeric(col("D_rsi14"), errors="coerce")
    rsi_cross = col("D_rsi7_gt_rsi14").fillna(False).astype(bool) if "D_rsi7_gt_rsi14" in df.columns else pd.Series(False, index=df.index)
    adx14     = pd.to_numeric(col("D_adx14"), errors="coerce")
    pdi14     = pd.to_numeric(col("D_pdi14"), errors="coerce")
    mdi14     = pd.to_numeric(col("D_mdi14"), errors="coerce")
    obv_rise  = col("D_price_and_obv_rising").fillna(False).astype(bool) if "D_price_and_obv_rising" in df.columns else pd.Series(False, index=df.index)
    atr14     = pd.to_numeric(col("D_atr14"), errors="coerce")
    cpr_w     = pd.to_numeric(col("D_cpr_width_pct"), errors="coerce").abs()
    vol       = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    avg20     = vol.rolling(20, min_periods=1).mean()
    cpr_above = (df.get("CPR_Yday_Above", 0) == 1)
    cpr_below = (df.get("CPR_Yday_Below", 0) == 1)
    long_score = (
        (golden*2).astype(float) + (ema_stack*2).astype(float) + (rsi_cross*1).astype(float)
        + (((rsi14.between(50,70, inclusive="both")).fillna(False)).astype(float)*1)
        + ((((adx14>20)&(pdi14>mdi14)).fillna(False)).astype(float)*1.5)
        + (obv_rise*1).astype(float)
        + cpr_above.astype(float)*0.5
        + (((pd.to_numeric(col("D_daily_trend"), errors="coerce")==1) &
            (pd.to_numeric(col("D_weekly_trend"), errors="coerce")==1)).fillna(False).astype(float)*0.5)
    )
    short_score = (
        ((~golden)*1).astype(float) + ((~ema_stack)*1).astype(float) + ((rsi14<45).fillna(False).astype(float)*1)
        + ((((adx14>20)&(mdi14>pdi14)).fillna(False)).astype(float)*1.5)
        + cpr_below.astype(float)*0.5
        + (((pd.to_numeric(col("D_daily_trend"), errors="coerce")==-1) &
            (pd.to_numeric(col("D_weekly_trend"), errors="coerce")==-1)).fillna(False).astype(float)*0.5)
    )
    atr_pct = (atr14/df["close"]).replace([np.inf,-np.inf],np.nan)*100 if "close" in df.columns else pd.Series(np.nan, index=df.index)
    risk_pen = (atr_pct>4).fillna(False).astype(float)*0.5 + (cpr_w>1.0).fillna(False).astype(float)*0.3
    liq_pen  = (avg20<MIN_AVG20_VOL).fillna(False).astype(float)*0.5
    df["long_score"]  = long_score - risk_pen - liq_pen
    df["short_score"] = short_score - risk_pen - liq_pen
    return df

# ===================== Panel schema =====================
def _unique_preserve(seq: List[str]) -> List[str]:
    out: List[str] = []; seen: set = set()
    for x in seq:
        if x not in seen: out.append(x); seen.add(x)
    return out

MASTER_KEEP_STATIC = _unique_preserve(
    ["timestamp","symbol","open","high","low","close","volume",
     "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
     "long_score","short_score","D_atr14","D_cpr_width_pct"]
)

class PanelParquetWriter:
    def __init__(self, out_path: str):
        if not _PA_OK:
            raise SystemExit("pyarrow is required to write panel_cache.parquet. Please run: pip install pyarrow")
        self.out_path = out_path; self._writer = None; self._schema = None
    def write_chunk(self, df: pd.DataFrame):
        if df is None or df.empty: return
        dynamic_keep = list(dict.fromkeys(
            MASTER_KEEP_STATIC + [c for c in df.columns if str(c).startswith(("D_","CPR_","Struct_","DayType_"))]
        ))
        # ensure presence
        for col in dynamic_keep:
            if col not in df.columns:
                if (str(col).startswith(("CPR_","Struct_","DayType_"))):
                    df[col] = 0
                else:
                    df[col] = np.nan
        df = df.copy()
        df["timestamp"] = ensure_kolkata_tz(pd.to_datetime(df["timestamp"], errors="coerce"))
        df["symbol"] = df["symbol"].astype(str).map(_clean_symbol_label)
        onehot_prefixes = ("CPR_Yday_","CPR_Tmr_","Struct_","DayType_")
        for c in df.columns:
            if str(c).startswith(onehot_prefixes):
                if df[c].dtype == bool: df[c] = df[c].astype(np.int32)
                else: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(np.int32)
        numeric_like = ["open","high","low","close","volume",
                        "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                        "long_score","short_score","D_atr14","D_cpr_width_pct"]
        for c in df.columns:
            if (c in numeric_like or str(c).startswith("D_") or str(c).startswith("ret_") or str(c).endswith("_pct")):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int32)
        df = df.reindex(columns=dynamic_keep)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.out_path, self._schema, compression="snappy")
        self._writer.write_table(table)
    def close(self):
        if self._writer is not None:
            self._writer.close(); self._writer = None

def append_panel_rows_parquet(writer: PanelParquetWriter, chunks: List[pd.DataFrame]):
    if not chunks: return
    aligned: List[pd.DataFrame] = []
    for df in chunks:
        df = _ensure_unique_columns(df)
        aligned.append(df)
    df_all = pd.concat(aligned, ignore_index=True, sort=False)
    writer.write_chunk(df_all)

def last_ts_by_symbol_from_panel(panel_path: str) -> dict:
    p = Path(panel_path)
    if not p.exists(): return {}
    try:
        df = pd.read_parquet(p)
        df["symbol"] = df["symbol"].astype(str).map(_clean_symbol_label)
        df["timestamp"] = ensure_kolkata_tz(pd.to_datetime(df["timestamp"], errors="coerce"))
        df = df.dropna(subset=["timestamp"])
        last = df.sort_values(["symbol","timestamp"]).groupby("symbol")["timestamp"].tail(1)
        return (df.loc[last.index, ["symbol","timestamp"]]
                .set_index("symbol")["timestamp"].to_dict())
    except Exception:
        return {}

# ===================== Collect panel =====================
def _prepare_panel_rows(path_obj: Path, min_ts_map: dict):
    sym = _derive_symbol_name(path_obj)
    try:
        df = load_one(path_obj)
        min_ts = min_ts_map.get(sym, None)
        if min_ts is not None:
            df = df[df["timestamp"] > pd.to_datetime(min_ts)]
        if df.empty:
            return sym, None, None, f"NO NEW ROWS {sym}"
        # sanity check example
        def _range_check(col, lo, hi):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                bad = (~s.between(lo,hi)) & s.notna()
                if bad.any():
                    _log_load_error(sym, str(path_obj), f"Range anomaly {col} out of [{lo},{hi}] on {bad.sum()} rows")
        _range_check("D_rsi14", 0, 100)
        df = add_targets(df)
        df, feats = featureize(df)
        df = score_bias(df)
        df["symbol"] = sym
        df_train = df.dropna(subset=["ret_1d_close_pct"])
        if df_train.empty:
            nrows = len(df)
            latest_ts = ensure_kolkata_tz(df["timestamp"]).max()
            label_counts = {h: int(df[f"ret_{h}d_close_pct"].notna().sum()) for h in (1,3,5)}
            msg = (f"NO TRAIN {sym} rows={nrows} labels={{1d:{label_counts[1]},3d:{label_counts[3]},5d:{label_counts[5]}}} "
                   f"latest_ts={latest_ts}")
            return sym, None, feats, msg
        rows = df_train[["timestamp","symbol","open","high","low","close","volume"] + feats +
                        ["ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                         "long_score","short_score","D_atr14","D_cpr_width_pct"]].copy()
        return sym, rows, feats, None
    except Exception as e:
        return sym, None, None, e

def collect_panel_from_paths(paths: List[Path], load_workers: int = 10):
    expanded: List[Path] = []
    for p in paths:
        if Path(p).is_dir():
            expanded += _strict_file_list(str(p), None, None, accept_any_daily=False)
        else:
            expanded.append(Path(p))
    paths = sorted(expanded)
    total = len(paths)
    if total == 0:
        empty = pd.DataFrame(columns=MASTER_KEEP_STATIC)
        if not _PA_OK:
            raise SystemExit("pyarrow is required to write panel_cache.parquet. Please run: pip install pyarrow")
        table = pa.Table.from_pandas(empty, preserve_index=False)
        pq.write_table(table, PANEL_OUT, compression="snappy")
        raise SystemExit("No matching files found.\nSelect files via --files or --gui, or provide --data-dir with *_daily.* files.")
    min_ts_map = last_ts_by_symbol_from_panel(PANEL_OUT)
    eta = ProgressETA(total=total, label="Load+Engineer")
    chunk: List[pd.DataFrame] = []
    total_rows_written = 0
    feats: Optional[List[str]] = None
    existing_panel = None
    if Path(PANEL_OUT).exists():
        try:
            existing_panel = pd.read_parquet(PANEL_OUT)
        except Exception:
            existing_panel = None
    writer = PanelParquetWriter(PANEL_OUT)
    if existing_panel is not None and not existing_panel.empty:
        existing_panel["timestamp"] = ensure_kolkata_tz(pd.to_datetime(existing_panel["timestamp"], errors="coerce"))
        existing_panel["symbol"] = existing_panel["symbol"].astype(str).map(_clean_symbol_label)
        # cast one-hots
        for c in existing_panel.columns:
            if (str(c).startswith(("CPR_Yday_","CPR_Tmr_","Struct_","DayType_"))):
                existing_panel[c] = pd.to_numeric(existing_panel[c], errors="coerce").fillna(0).clip(0,1).astype(np.int32)
        for c in existing_panel.columns:
            if (c in ("open","high","low","close","volume",
                      "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                      "long_score","short_score","D_atr14","D_cpr_width_pct") or str(c).startswith("D_")):
                existing_panel[c] = pd.to_numeric(existing_panel[c], errors="coerce").astype(np.float64)
        writer.write_chunk(existing_panel)

    def _prepare_with_path(path_obj: Path):
        return path_obj, _prepare_panel_rows(path_obj, min_ts_map)

    try:
        if load_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(load_workers)) as ex:
                for path_obj, result in ex.map(_prepare_with_path, paths):
                    sym, rows, feats_out, msg_or_err = result
                    if isinstance(msg_or_err, Exception):
                        _log_load_error(sym, str(path_obj), str(msg_or_err))
                        eta.tick(f"ERR {sym}: {msg_or_err}"); continue
                    if msg_or_err:
                        eta.tick(msg_or_err); continue
                    chunk.append(rows)
                    if feats_out is not None: feats = feats_out
                    total_rows_written += len(rows)
                    if len(chunk) >= CHUNK_SIZE:
                        append_panel_rows_parquet(writer, chunk); chunk.clear()
                    eta.tick(f"OK {sym} (+{len(rows)} rows)")
        else:
            for path_obj in paths:
                sym, rows, feats_out, msg_or_err = _prepare_panel_rows(path_obj, min_ts_map)
                if isinstance(msg_or_err, Exception):
                    _log_load_error(sym, str(path_obj), str(msg_or_err))
                    eta.tick(f"ERR {sym}: {msg_or_err}"); continue
                if msg_or_err:
                    eta.tick(msg_or_err); continue
                chunk.append(rows)
                if feats_out is not None: feats = feats_out
                total_rows_written += len(rows)
                if len(chunk) >= CHUNK_SIZE:
                    append_panel_rows_parquet(writer, chunk); chunk.clear()
                eta.tick(f"OK {sym} (+{len(rows)} rows)")
    except KeyboardInterrupt:
        print("\nInterrupted! Autosaving current chunk...")
        if chunk: append_panel_rows_parquet(writer, chunk); chunk.clear()
        writer.close(); raise

    if chunk: append_panel_rows_parquet(writer, chunk); chunk.clear()
    writer.close()
    print(f"[Panel] Appended new rows: {total_rows_written}")
    panel = pd.read_parquet(PANEL_OUT)
    panel["symbol"] = panel["symbol"].astype(str).map(_clean_symbol_label)
    panel["timestamp"] = ensure_kolkata_tz(pd.to_datetime(panel["timestamp"], errors="coerce"))
    panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    feats = [c for c in panel.columns if (
        str(c).startswith("D_") or str(c).startswith("CPR_Yday_") or str(c).startswith("CPR_Tmr_")
        or str(c).startswith("Struct_") or str(c).startswith("DayType_")
    )]
    return panel, feats

# ===================== Feature matrix sanitize =====================
def sanitize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    for c in X.columns:
        ser = X[c]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        X[c] = ser
        if ser.dtype == bool:
            X[c] = ser.astype(int)
        elif ser.dtype == object:
            s = ser.astype(str).str.lower()
            uniq = set(pd.Series(s).unique())
            if uniq <= {"true","false","nan"}:
                X[c] = pd.Series(s).map({"true":1, "false":0}).astype("Int64").fillna(0).astype(int)
            else:
                X[c] = pd.to_numeric(s, errors="coerce")
        if str(c).startswith(("CPR_Yday_","CPR_Tmr_","Struct_","DayType_")):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)
    return X

# ===================== Schema lock (from training matrix) =====================
def compute_impute_stats(X: pd.DataFrame) -> Dict[str, float]:
    return X.median(numeric_only=True).to_dict()

def save_schema(schema_path: str, feats: List[str], impute: Dict[str, float]):
    data = {"features": list(feats), "impute": {k: float(v) for k, v in impute.items()}}
    Path(schema_path).write_text(json.dumps(data, indent=2))

def load_schema(schema_path: str) -> Tuple[List[str], Dict[str, float]]:
    data = json.loads(Path(schema_path).read_text())
    return list(data["features"]), {k: float(v) for k, v in data["impute"].items()}

def reindex_and_impute(X_last: pd.DataFrame, feats: List[str], impute: Dict[str, float]) -> pd.DataFrame:
    X = X_last.reindex(columns=feats).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(impute.get(c, 0.0))
    const_cols = (X.nunique(dropna=False) <= 1)
    zero_cols  = (X == 0).all()
    print(f"[Schema] infer matrix shape={X.shape} const_cols={const_cols.mean()*100:.1f}% zero_cols={zero_cols.mean()*100:.1f}%")
    return X

# ===================== Gaussian CDF helpers (fallback prob) =====================
def _phi_approx(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos_inf = np.isposinf(z); neg_inf = np.isneginf(z)
    out[pos_inf] = 1.0; out[neg_inf] = 0.0
    finite = np.isfinite(z)
    if np.any(finite):
        x = z[finite]
        t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
        a1,a2,a3,a4,a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        poly = (((a5*t + a4)*t + a3)*t + a2)*t + a1
        nd = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * x*x)
        approx = 1.0 - nd * poly * t
        out[finite] = np.where(x >= 0, approx, 1.0 - approx)
    return out

def prob_up_from_gaussian(mean, std):
    mean = np.asarray(mean, dtype=float); std = np.asarray(std, dtype=float)
    z = np.divide(mean, std, out=np.full_like(mean, np.nan, dtype=float), where=(std > 0))
    p = _phi_approx(z)
    fallback = np.where(mean > 0, 0.75, np.where(mean < 0, 0.25, 0.50))
    p = np.where(np.isfinite(p), p, fallback)
    return np.clip(p, 0.0, 1.0)

# ===================== Walk-forward splits =====================
def time_cv_by_timestamp(panel: pd.DataFrame,
                         n_splits: int = 10,
                         embargo_days: int = 0,
                         target_mask: Optional[pd.Series] = None):
    idx = panel.index if target_mask is None else panel.index[target_mask]
    ts_all = pd.to_datetime(panel.loc[idx, "timestamp"]).dt.normalize()
    uniq_dates = pd.Series(ts_all).sort_values().unique()
    if len(uniq_dates) < n_splits + 1:
        n_splits = max(1, min(len(uniq_dates) - 1, n_splits))
    cut = np.linspace(0, len(uniq_dates), n_splits + 1, dtype=int)
    for i in range(n_splits):
        start_date = uniq_dates[cut[i]]
        end_date = uniq_dates[cut[i+1]-1] if i < n_splits - 1 else uniq_dates[-1]
        te_mask = (panel["timestamp"].dt.normalize() >= start_date) & (panel["timestamp"].dt.normalize() <= end_date)
        tr_mask = (panel["timestamp"].dt.normalize() < start_date)
        if embargo_days and embargo_days > 0:
            embargo_edge = start_date - pd.Timedelta(days=int(embargo_days))
            tr_mask = (panel["timestamp"].dt.normalize() <= embargo_edge)
        tr_idx = panel.index[tr_mask & panel.index.isin(idx)]
        te_idx = panel.index[te_mask & panel.index.isin(idx)]
        if len(te_idx) > 0 and len(tr_idx) > 0:
            yield tr_idx, te_idx

# ===================== LightGBM / XGBoost / CatBoost checks =====================
def _check_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor, LGBMClassifier
        return lgb, LGBMRegressor, LGBMClassifier
    except Exception as e:
        raise SystemExit("LightGBM is not installed. Please run: pip install lightgbm") from e

def _check_xgboost():
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier, XGBRegressor
        return xgb, XGBClassifier, XGBRegressor
    except Exception as e:
        print(f"XGBoost not available ({e}); stacking will skip XGB.")
        return None, None, None

def _check_catboost():
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostClassifier, CatBoostRegressor
    except Exception as e:
        print(f"CatBoost not available ({e}); stacking will skip CatBoost.")
        return None, None

# ===================== Split helpers =====================
def split_train_val_by_time(panel: pd.DataFrame, candidate_idx: np.ndarray,
                            val_frac: float = 0.15, min_val: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_idx is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    idx = np.asarray(candidate_idx)
    if len(idx) < 3:
        return idx, np.array([], dtype=idx.dtype)
    ts = pd.to_datetime(panel.loc[idx, "timestamp"]).values
    order = np.argsort(ts)
    val_n = max(1, int(round(len(order) * val_frac)))
    val_n = max(val_n, min_val)
    val_n = min(len(order) // 2, val_n)
    if val_n == 0:
        return idx, np.array([], dtype=idx.dtype)
    val_order   = order[-val_n:]
    train_order = order[:-val_n]
    return idx[train_order], idx[val_order]

# ===================== Calibration helper =====================
def _calibrate_best_brier(est, X_val, y_val):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    iso = CalibratedClassifierCV(estimator=est, method="isotonic", cv="prefit")
    iso.fit(X_val, y_val)
    p_iso = iso.predict_proba(X_val)[:, 1]
    br_iso = brier_score_loss(y_val, p_iso)
    sig = CalibratedClassifierCV(estimator=est, method="sigmoid", cv="prefit")
    sig.fit(X_val, y_val)
    p_sig = sig.predict_proba(X_val)[:, 1]
    br_sig = brier_score_loss(y_val, p_sig)
    chosen = iso if br_iso <= br_sig else sig
    info = {"brier_isotonic": float(br_iso), "brier_sigmoid": float(br_sig),
            "chosen": "isotonic" if br_iso <= br_sig else "sigmoid"}
    return chosen, info

# ===================== 1D / 5D classification =====================
def _build_monotone_vector(X_cols: List[str], mono_map: Dict[str,int]):
    return [int(mono_map.get(c, 0)) for c in X_cols]

def _lgbm_cls_params(X_cols: List[str], use_monotone: bool, mono_map: Dict[str,int], rnd: int):
    depth = MAX_DEPTH_ALL if isinstance(MAX_DEPTH_ALL, int) and MAX_DEPTH_ALL > 0 else -1
    params = dict(
        n_estimators=int(N_EST_1D), learning_rate=LEARNING_RATE,
        num_leaves=56, max_depth=depth, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=200, min_gain_to_split=0.03, max_bin=127,
        reg_alpha=0.6, reg_lambda=9.0, class_weight='balanced',
        n_jobs=-1, random_state=int(rnd), verbosity=-1,
    )
    if use_monotone:
        params["monotone_constraints"] = _build_monotone_vector(list(X_cols), mono_map)
    return params

def train_1d_cls_calibrated(panel: pd.DataFrame, feats: List[str], margin_pct: float,
                            n_splits: int, embargo_days: int,
                            early_stopping_rounds: int, use_monotone: bool, mono_map: Dict[str,int]):
    lgb, _, LGBMClassifier = _check_lightgbm()
    r = pd.to_numeric(panel["ret_1d_close_pct"], errors="coerce")
    y = pd.Series(np.nan, index=panel.index)
    y[(r > margin_pct)] = 1
    y[(r < -margin_pct)] = 0
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labeled rows for 1D classification after applying margin.")
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)
    eta = ProgressETA(total=len(folds), label="Train 1D-CLS")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0: continue
        params = _lgbm_cls_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + fold_no)
        clf = LGBMClassifier(**params)
        tr_core_idx, val_idx = split_train_val_by_time(panel, valid_idx[tr_pos], val_frac=0.2, min_val=25)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos     = np.where(np.isin(valid_idx, val_idx))[0]
        X_te  = X_full.iloc[te_pos];   y_te  = y_full[te_pos]
        X_tr  = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr  = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        # ===== SAFE VALIDATION FALLBACK (train slice), not test =====
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos]      if len(val_pos)>0 else y_tr
        import lightgbm as _lgb_mod
        callbacks = [_lgb_mod.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0 and len(val_pos) > 0:
            callbacks.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)
        calib, diag = _calibrate_best_brier(clf, X_val, y_val)
        prob = calib.predict_proba(X_te)[:, 1]
        oos_prob[te_pos] = prob
        eta.tick(f"fold {fold_no} chosen={diag['chosen']} br_iso={diag['brier_isotonic']:.5f} br_sig={diag['brier_sigmoid']:.5f}")
    # Final calibrated model on last 20% by time
    final_base = LGBMClassifier(**_lgbm_cls_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + 1000))
    df_valid = panel.loc[mask, ["timestamp"]].copy()
    order = np.argsort(df_valid["timestamp"].values)
    split = int(round(len(order) * 0.8)); tr_order = order[:split]; te_order = order[split:]
    X_tr_all = X_full.iloc[tr_order]; y_tr_all = y_full[tr_order]
    X_te_all = X_full.iloc[te_order]; y_te_all = y_full[te_order]
    import lightgbm as _lgb_mod
    callbacks = [_lgb_mod.callback.log_evaluation(period=0)]
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
    final_base.fit(X_tr_all, y_tr_all, eval_set=[(X_te_all, y_te_all)], eval_metric="binary_logloss", callbacks=callbacks)
    final_calib, final_diag = _calibrate_best_brier(final_base, X_te_all, y_te_all)
    print(f"[Calib-final 1D] chosen={final_diag['chosen']} brier_iso={final_diag['brier_isotonic']:.5f} brier_sig={final_diag['brier_sigmoid']:.5f}")
    return final_calib, oos_prob, valid_idx

def train_5d_cls_calibrated(panel: pd.DataFrame, feats: List[str],
                            n_splits: int, embargo_days: int,
                            early_stopping_rounds: int, use_monotone: bool, mono_map: Dict[str,int]):
    # Label: up/down by sign of 5D return
    lgb, _, LGBMClassifier = _check_lightgbm()
    r = pd.to_numeric(panel["ret_5d_close_pct"], errors="coerce")
    y = (r > 0).astype(int)
    mask = r.notna()
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)
    eta = ProgressETA(total=len(folds), label="Train 5D-CLS")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0: continue
        params = _lgbm_cls_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + 500 + fold_no)
        clf = LGBMClassifier(**params)
        tr_core_idx, val_idx = split_train_val_by_time(panel, valid_idx[tr_pos], val_frac=0.2, min_val=25)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos     = np.where(np.isin(valid_idx, val_idx))[0]
        X_te  = X_full.iloc[te_pos];   y_te  = y_full[te_pos]
        X_tr  = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr  = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        # ===== SAFE VALIDATION FALLBACK (train slice), not test =====
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos]      if len(val_pos)>0 else y_tr
        import lightgbm as _lgb_mod
        callbacks = [_lgb_mod.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0 and len(val_pos) > 0:
            callbacks.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)
        calib, diag = _calibrate_best_brier(clf, X_val, y_val)
        prob = calib.predict_proba(X_te)[:, 1]
        oos_prob[te_pos] = prob
        eta.tick(f"fold {fold_no} chosen={diag['chosen']} br_iso={diag['brier_isotonic']:.5f} br_sig={diag['brier_sigmoid']:.5f}")
    # Final calibrated model on last 20% by time
    final_base = LGBMClassifier(**_lgbm_cls_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + 1500))
    df_valid = panel.loc[mask, ["timestamp"]].copy()
    order = np.argsort(df_valid["timestamp"].values)
    split = int(round(len(order) * 0.8)); tr_order = order[:split]; te_order = order[split:]
    X_tr_all = X_full.iloc[tr_order]; y_tr_all = y_full[tr_order]
    X_te_all = X_full.iloc[te_order]; y_te_all = y_full[te_order]
    import lightgbm as _lgb_mod
    callbacks = [_lgb_mod.callback.log_evaluation(period=0)]
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
    final_base.fit(X_tr_all, y_tr_all, eval_set=[(X_te_all, y_te_all)], eval_metric="binary_logloss", callbacks=callbacks)
    final_calib, final_diag = _calibrate_best_brier(final_base, X_te_all, y_te_all)
    print(f"[Calib-final 5D] chosen={final_diag['chosen']} brier_iso={final_diag['brier_isotonic']:.5f} brier_sig={final_diag['brier_sigmoid']:.5f}")
    return final_calib, oos_prob, valid_idx

# ===================== Regression (LightGBM) =====================
def _lgbm_reg_params(X_cols: List[str], use_monotone: bool, mono_map: Dict[str,int], rnd: int, n_estimators: int):
    depth = MAX_DEPTH_ALL if isinstance(MAX_DEPTH_ALL, int) and MAX_DEPTH_ALL > 0 else -1
    params = dict(
        n_estimators=int(n_estimators), learning_rate=LEARNING_RATE,
        num_leaves=63, max_depth=depth, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=255,
        reg_alpha=0.3, reg_lambda=5.0, n_jobs=-1, random_state=int(rnd), verbosity=-1,
    )
    if use_monotone and mono_map:
        params["monotone_constraints"] = _build_monotone_vector(list(X_cols), mono_map)
    return params

def train_rf(panel: pd.DataFrame, feats: List[str], target_col: str, label: str,
             n_estimators: int = N_EST_ALL, refit_final: bool = False, n_splits: int = FOLDS,
             embargo_days: int = EMBARGO_DAYS, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
             use_monotone: bool = False, mono_map: Dict[str,int] = None):
    lgb, LGBMRegressor, _ = _check_lightgbm()
    mask = panel[target_col].notna()
    if not mask.any(): raise ValueError(f"No non-NaN rows for target {target_col}")
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = panel.loc[mask, target_col].values
    oos = np.full(len(X_full), np.nan)
    eta = ProgressETA(total=len(folds), label=f"Train {label}")
    models = []
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0: continue
        gbm = LGBMRegressor(**_lgbm_reg_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + fold_no, n_estimators=n_estimators))
        tr_core_idx, val_idx = split_train_val_by_time(panel, valid_idx[tr_pos], val_frac=0.2, min_val=25)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos     = np.where(np.isin(valid_idx, val_idx))[0]
        X_tr  = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr  = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        # ===== SAFE VALIDATION FALLBACK (train slice), not test =====
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos]      if len(val_pos)>0 else y_tr
        X_te  = X_full.iloc[te_pos]
        import lightgbm as _lgb_mod
        callbacks = [_lgb_mod.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        gbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="l2", callbacks=callbacks)
        pred = gbm.predict(X_te, num_iteration=getattr(gbm, "best_iteration_", None))
        oos[te_pos] = pred
        mae = float(np.mean(np.abs(y_full[te_pos] - pred)))
        eta.tick(f"fold {fold_no} MAE={mae:.3f}% n={len(te_pos)}")
        models.append(gbm)
    final = models[-1] if models else LGBMRegressor(**_lgbm_reg_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + 2000, n_estimators=n_estimators))
    if not models:
        import lightgbm as _lgb_mod
        final.fit(X_full, y_full, callbacks=[_lgb_mod.callback.log_evaluation(period=0)])
    if refit_final:
        print(f"[Train {label}] refit on ALL valid rows ...")
        import lightgbm as _lgb_mod
        final = LGBMRegressor(**_lgbm_reg_params(X_full.columns, use_monotone, mono_map, GLOBAL_SEED + 2001, n_estimators=n_estimators))
        callbacks = [_lgb_mod.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        final.fit(X_full, y_full, eval_set=[(X_full, y_full)], eval_metric="l2", callbacks=callbacks)
    return final, oos, valid_idx

# ===================== OOS save/load =====================
def _load_joblib():
    spec = importlib.util.find_spec("joblib")
    if spec is None:
        raise SystemExit("joblib is required for saving/loading checkpoints. Please run `pip install joblib`.")
    return importlib.import_module("joblib")

def maybe_load_model(path: Path):
    joblib = _load_joblib()
    try:
        if path.exists():
            print(f"Loading checkpoint: {path}")
            return joblib.load(path)
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
    return None

def save_model(path: Path, model):
    joblib = _load_joblib()
    try:
        joblib.dump(model, path)
        print(f"Saved model checkpoint: {path}")
    except Exception as e:
        print(f"Failed to save checkpoint {path}: {e}")

def save_oos(path: Path, oos: np.ndarray, valid_idx: np.ndarray):
    try:
        df = pd.DataFrame({"panel_idx": valid_idx, "oos_pred": oos})
        df.to_csv(path, index=False)
        print(f"Saved OOS predictions: {path} (rows={df.shape[0]})")
    except Exception as e:
        print(f"Failed to save OOS {path}: {e}")

def load_oos(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load OOS {path}: {e}")
    return None

# ===================== Std helpers for 5D fallback =====================
def _cross_sectional_std(values: np.ndarray, floor: float = 1e-6) -> float:
    s = float(np.nanstd(values))
    return s if np.isfinite(s) and s > floor else 1.0

def _symbol_hist_roll_std(panel: pd.DataFrame, window: int, min_rows: int) -> pd.Series:
    df = panel.sort_values(["symbol","timestamp"])
    grp = df.groupby("symbol", sort=False)
    roll = grp["ret_5d_close_pct"].apply(lambda x: pd.to_numeric(x, errors="coerce").rolling(window, min_periods=min_rows).std())
    df2 = df.copy()
    df2["roll_std_5d"] = roll.values
    last = df2.groupby("symbol", as_index=True)["roll_std_5d"].tail(1)
    return last

def _residual_std_5d(panel: pd.DataFrame, oos5_df: pd.DataFrame) -> Dict[str, float]:
    if oos5_df is None or oos5_df.empty: return {}
    idx  = oos5_df["panel_idx"].values
    pred = pd.to_numeric(oos5_df["oos_pred"], errors="coerce").values
    in_panel = np.isin(idx, panel.index.values)
    idx  = idx[in_panel]; pred = pred[in_panel]
    reals = pd.to_numeric(panel.loc[idx, "ret_5d_close_pct"], errors="coerce").values
    syms  = panel.loc[idx, "symbol"].values
    resid = reals - pred
    df    = pd.DataFrame({"symbol": syms, "resid": resid})
    stds  = df.groupby("symbol")["resid"].std().to_dict()
    return {k: float(v) for k, v in stds.items() if np.isfinite(v) and v > 1e-6}

# ===================== Watchlist helpers =====================
def nightly_watchlist(panel: pd.DataFrame, feats: List[str],
                      m1_cls, m1_reg, m3_reg, m5_reg,
                      m5_cls_calib,
                      oos5_path: Optional[Path],
                      prob_std_method: str = PROB_STD_METHOD,
                      prob_std_window: int = PROB_STD_WINDOW,
                      prob_std_min_rows: int = PROB_STD_MIN_ROWS,
                      exclude_pattern: Optional[str] = r"(LIQUID|GOLD|BEES|IETF|ETF|CASE|ADD)$"):
    panel = panel.copy().sort_values(["symbol", "timestamp"])
    panel["avg20_vol"] = panel.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    last = panel.groupby("symbol", as_index=False).tail(1).copy()

    # ===== Equities-only filtering (optional) =====
    if exclude_pattern:
        mask = ~last["symbol"].astype(str).str.contains(exclude_pattern, regex=True, na=False)
        last = last.loc[mask].copy()

    # Schema lock: load & apply impute
    feats_schema, impute_stats = load_schema(FEATURES_SCHEMA_PATH)
    X_raw = sanitize_feature_matrix(last[feats_schema].copy())
    X     = reindex_and_impute(X_raw, feats_schema, impute_stats)

    # Predictions
    last["prob_up_1d"]       = m1_cls.predict_proba(X)[:, 1] if m1_cls is not None else np.nan
    last["pred_ret_1d_pct"]  = m1_reg.predict(X)              if m1_reg is not None else np.nan
    best_it_3 = getattr(m3_reg, "best_iteration_", None) if m3_reg is not None else None
    last["pred_ret_3d_pct"]  = (m3_reg.predict(X, num_iteration=best_it_3) if m3_reg is not None else np.nan)
    best_it_5 = getattr(m5_reg, "best_iteration_", None) if m5_reg is not None else None
    last["pred_ret_5d_pct"]  = (m5_reg.predict(X, num_iteration=best_it_5) if m5_reg is not None else np.nan)

    # 5D probability — prefer calibrated classifier if present
    if m5_cls_calib is not None:
        last["prob_up_5d"] = m5_cls_calib.predict_proba(X)[:, 1]
        last["pred_std_5d"] = np.nan  # classifier path: std not needed
    else:
        std5 = np.full(len(last), np.nan, dtype=float)
        if prob_std_method == "residual" and oos5_path is not None and Path(oos5_path).exists():
            oos5_df      = pd.read_csv(oos5_path)
            residual_map = _residual_std_5d(panel, oos5_df)
            std5         = last["symbol"].map(residual_map).astype(float).values
            if np.isnan(std5).mean() > 0.5:
                sym_hist = _symbol_hist_roll_std(panel, window=int(prob_std_window), min_rows=int(prob_std_min_rows))
                std_map2 = sym_hist.to_dict()
                sh       = last["symbol"].map(std_map2).astype(float).values
                std5     = np.where(np.isfinite(std5), std5, sh)
            if np.isnan(std5).mean() > 0.5:
                cs  = _cross_sectional_std(last["pred_ret_5d_pct"].values)
                std5 = np.where(np.isfinite(std5), std5, cs)
        elif prob_std_method == "symbol_hist":
            sym_hist = _symbol_hist_roll_std(panel, window=int(prob_std_window), min_rows=int(prob_std_min_rows))
            std_map  = sym_hist.to_dict()
            std5     = last["symbol"].map(std_map).astype(float).values
            if np.isnan(std5).mean() > 0.5:
                cs  = _cross_sectional_std(last["pred_ret_5d_pct"].values)
                std5 = np.where(np.isfinite(std5), std5, cs)
        elif prob_std_method == "cross":
            cs  = _cross_sectional_std(last["pred_ret_5d_pct"].values)
            std5[:] = cs
        last["pred_std_5d"]  = std5
        last["prob_up_5d"]   = prob_up_from_gaussian(last["pred_ret_5d_pct"].values, std5)

    wl = last[(last["close"] >= MIN_CLOSE) & (last["avg20_vol"] >= MIN_AVG20_VOL)].copy()
    wl["bias"] = np.where(wl["long_score"] >= wl["short_score"], "LONG", "SHORT")
    wl = wl[["symbol","timestamp","close","avg20_vol",
             "prob_up_1d","pred_ret_1d_pct","pred_ret_3d_pct","pred_ret_5d_pct",
             "pred_std_5d","prob_up_5d",
             "D_atr14","D_cpr_width_pct","long_score","short_score","bias"]].sort_values(
        ["prob_up_1d","pred_ret_5d_pct"], ascending=[False, False]
    )
    wl.to_csv(WATCHLIST_OUT, index=False)
    print(f"Saved: {WATCHLIST_OUT} rows={len(wl)}")
    return wl

# ===================== OOS quick report =====================
def oos_report(panel: pd.DataFrame,
               oos_1d_prob: Optional[np.ndarray], oos_1d_idx: Optional[np.ndarray],
               oos_3d_reg: Optional[np.ndarray],  oos_3d_idx: Optional[np.ndarray],
               oos_5d_reg: Optional[np.ndarray],  oos_5d_idx: Optional[np.ndarray],
               oos_5d_prob: Optional[np.ndarray], oos_5d_prob_idx: Optional[np.ndarray]) -> Dict[str, Dict]:
    from sklearn.metrics import brier_score_loss, accuracy_score, mean_absolute_error
    rep = {}

    # 1D classification
    if oos_1d_prob is not None and oos_1d_idx is not None:
        y_real = (pd.to_numeric(panel.loc[oos_1d_idx, "ret_1d_close_pct"], errors="coerce") > 0).astype(int).values
        y_pred_prob = np.asarray(oos_1d_prob)
        y_pred_cls  = (y_pred_prob >= 0.5).astype(int)
        rep["1d_cls"] = {
            "n": int(len(y_real)),
            "acc": float(accuracy_score(y_real, y_pred_cls)),
            "brier": float(brier_score_loss(y_real, y_pred_prob))
        }

    # 3D regression
    if oos_3d_reg is not None and oos_3d_idx is not None:
        y_real = pd.to_numeric(panel.loc[oos_3d_idx, "ret_3d_close_pct"], errors="coerce").values
        y_pred = np.asarray(oos_3d_reg)
        rep["3d_reg"] = {
            "n": int(len(y_real)),
            "mae_pct": float(mean_absolute_error(y_real, y_pred))
        }

    # 5D regression
    if oos_5d_reg is not None and oos_5d_idx is not None:
        y_real = pd.to_numeric(panel.loc[oos_5d_idx, "ret_5d_close_pct"], errors="coerce").values
        y_pred = np.asarray(oos_5d_reg)
        rep["5d_reg"] = {
            "n": int(len(y_real)),
            "mae_pct": float(mean_absolute_error(y_real, y_pred))
        }

    # 5D classification (if trained)
    if oos_5d_prob is not None and oos_5d_prob_idx is not None and np.isfinite(oos_5d_prob).any():
        y_real = (pd.to_numeric(panel.loc[oos_5d_prob_idx, "ret_5d_close_pct"], errors="coerce") > 0).astype(int).values
        y_pred_prob = np.asarray(oos_5d_prob)
        y_pred_cls  = (y_pred_prob >= 0.5).astype(int)
        rep["5d_cls"] = {
            "n": int(len(y_real)),
            "acc": float(accuracy_score(y_real, y_pred_cls)),
            "brier": float(brier_score_loss(y_real, y_pred_prob)),
            "pct_extreme_ge_0_99": float((y_pred_prob >= 0.99).mean()*100.0)
        }

    Path(OOS_REPORT_PATH).write_text(json.dumps(rep, indent=2))
    print(f"[OOS] Saved report: {OOS_REPORT_PATH}")
    return rep

# ===================== Minimal CLI =====================
def parse_cli(argv=None):
    import argparse
    p = argparse.ArgumentParser("predictor_v6_2")
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symbols-like", default=None)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--accept-any-daily", default="false")
    p.add_argument("--load-workers", type=int, default=8)

    p.add_argument("--cv-splits", type=int, default=FOLDS)
    p.add_argument("--embargo-days", type=int, default=EMBARGO_DAYS)
    p.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)

    p.add_argument("--prob-std-method", default=PROB_STD_METHOD, choices=["residual","symbol_hist","cross","none"])
    p.add_argument("--exclude-pattern", default=r"(LIQUID|GOLD|BEES|IETF|ETF|CASE|ADD)$")

    p.add_argument("--train-1d-mode", default="both", choices=["cls","reg","both"])  # ensure m1_reg trained or accept NaN
    return p.parse_args(argv)

def main(argv=None):
    args = parse_cli(argv)
    setup_paths(args.out_dir)

    # 1) Build panel
    paths = _strict_file_list(args.data_dir, args.symbols_like, args.limit_files, accept_any_daily=args.accept_any_daily)
    panel, feats = collect_panel_from_paths(paths, load_workers=args.load_workers)
    print(f"[Panel] final rows={len(panel)} cols={len(panel.columns)} feats={len(feats)}")

    # 2) Train models
    # 1D CLS (always train so prob_up_1d is present)
    m1_cls, oos1_prob, oos1_idx = train_1d_cls_calibrated(panel, feats, CLS_MARGIN,
                                                          n_splits=args.cv_splits, embargo_days=args.embargo_days,
                                                          early_stopping_rounds=args.early_stopping_rounds,
                                                          use_monotone=USE_MONOTONE, mono_map=MONOTONE_MAP_DEFAULT)
    # 1D REG (train unless user sets 'cls')
    if args.train_1d_mode in ("reg","both"):
        m1_reg, oos1_reg, oos1reg_idx = train_rf(panel, feats, "ret_1d_close_pct", "1D",
                                                 n_estimators=N_EST_1D, refit_final=False, n_splits=args.cv_splits,
                                                 embargo_days=args.embargo_days, early_stopping_rounds=args.early_stopping_rounds,
                                                 use_monotone=USE_MONOTONE, mono_map=MONOTONE_MAP_DEFAULT)
    else:
        m1_reg, oos1_reg, oos1reg_idx = None, None, None

    # 3D REG
    m3_reg, oos3_reg, oos3_idx = train_rf(panel, feats, "ret_3d_close_pct", "3D",
                                          n_estimators=N_EST_3D, refit_final=False, n_splits=args.cv_splits,
                                          embargo_days=args.embargo_days, early_stopping_rounds=args.early_stopping_rounds,
                                          use_monotone=USE_MONOTONE, mono_map=MONOTONE_MAP_DEFAULT)
    # 5D REG
    m5_reg, oos5_reg, oos5_idx = train_rf(panel, feats, "ret_5d_close_pct", "5D",
                                          n_estimators=N_EST_5D, refit_final=False, n_splits=args.cv_splits,
                                          embargo_days=args.embargo_days, early_stopping_rounds=args.early_stopping_rounds,
                                          use_monotone=USE_MONOTONE, mono_map=MONOTONE_MAP_DEFAULT)
    # 5D CLS (optional; enabled by default here)
    m5_cls_calib, oos5_prob, oos5prob_idx = train_5d_cls_calibrated(panel, feats,
                                                                    n_splits=args.cv_splits, embargo_days=args.embargo_days,
                                                                    early_stopping_rounds=args.early_stopping_rounds,
                                                                    use_monotone=USE_MONOTONE, mono_map=MONOTONE_MAP_DEFAULT)

    # 3) Schema lock — from the ACTUAL training matrix (use 5D REG mask to maximize coverage)
    mask_5d = panel["ret_5d_close_pct"].notna()
    X_train = sanitize_feature_matrix(panel.loc[mask_5d, feats].copy())
    impute_stats = compute_impute_stats(X_train)
    save_schema(FEATURES_SCHEMA_PATH, feats, impute_stats)

    # 4) OOS report
    rep = oos_report(panel,
                     oos1_prob, oos1_idx,
                     oos3_reg,  oos3_idx,
                     oos5_reg,  oos5_idx,
                     oos5_prob, oos5prob_idx)
    print(json.dumps(rep, indent=2))

    # 5) Nightly watchlist
    wl = nightly_watchlist(panel, feats, m1_cls, m1_reg, m3_reg, m5_reg, m5_cls_calib,
                           oos5_path=Path(args.out_dir) / "oos_5d_reg.csv",
                           prob_std_method=args.prob_std_method,
                           exclude_pattern=args.exclude_pattern)
    return 0

if __name__ == "__main__":
    sys.exit(main())
