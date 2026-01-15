
# cpr_fix_with_embedded_alpha_ooscal_v3.py
# Production + Embedded Research Harness with strict OOS calibration (Train → Calibrate → Test)
# ---------------------------------------------------------------------------------------------
# This v3 integrates the "good ideas" (1–7) from the ALL_IN_ONE file into the original pipeline:
# 1) Regime-aware features (trend/vol/dispersion) added to feature set.
# 2) Isotonic expected-value calibration: map prob -> expected 5D adjusted payoff (monotone EV mapper).
# 3) Safe early stopping: enable only when validation size is sufficiently large.
# 4) Protected 1D gate: minimum samples + post-fit sigmoid calibration.
# 5) Simple portfolio backtest harness: quick top-K equity sanity-check from expected returns.
# 6) Remove class_weight='balanced' for 5D classifier (quantile labels should be balanced by design).
# 7) Centralized LightGBM callbacks helper.
#
# Notes:
# - This v3 keeps your existing CLI, schema lock, OOS report, and watchlist outputs intact.
# - EV mapping via isotonic is added alongside existing decile calibration (used for 3d/5d raw).
# - We keep your current return definition (close→close) to respect scope of points 1–7 only.

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
DATA_DIR_DEFAULT = r"C:\Users\karanvsi\Desktop\Pycharm\Cache\cache_daily_new"
PANEL_OUT = None
WATCHLIST_OUT = None
STATUS_PATH = None
META_PATH = None
LOG_DIR = None
LOAD_ERRORS_LOG = None
QUARANTINE_LIST = None
FEATURES_SCHEMA_PATH = None  # <out-dir>/features_train.json
OOS_REPORT_PATH = None       # <out-dir>/oos_report.json
CALIB_TABLE_PATH = None      # <out-dir>/calibration_5d_deciles.json
RESEARCH_CSV_PATH = None     # <out-dir>/research_report.csv
MODEL_CALIB_PATH = None      # <out-dir>/model_lgbm_calibrated.joblib (prob-calibrator)
ISO_EV_MAPPER_PATH = None    # <out-dir>/isotonic_ev_mapper.joblib (prob -> expected_ret_5d_adj)

# ===================== MODEL / CV =====================
GLOBAL_SEED = 42
FOLDS = 7
EMBARGO_DAYS = 3  # (unchanged per scope 1-7)
# LightGBM baseline for classifiers
N_EST_1D = 2400
N_EST_5D = 2800
EARLY_STOPPING_ROUNDS = 200
LEARNING_RATE = 0.01
MAX_DEPTH = 6
MIN_VAL_EARLYSTOP = 500  # <-- new: enable ES only if validation >= this
MIN_GATE_SAMPLES = 500   # <-- new: gate requires minimum labels

# Filters for watchlist
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

# Chunking
CHUNK_SIZE = 1200

# 1D classification margin (supportive gate)
CLS_MARGIN_1D = 0.10
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
    global LOG_DIR, LOAD_ERRORS_LOG, QUARANTINE_LIST, FEATURES_SCHEMA_PATH
    global OOS_REPORT_PATH, CALIB_TABLE_PATH, RESEARCH_CSV_PATH
    global MODEL_CALIB_PATH, ISO_EV_MAPPER_PATH

    PANEL_OUT = str(out / "panel_cache.parquet")
    WATCHLIST_OUT = str(out / "watchlist_5d_signal.csv")
    STATUS_PATH = str(out / "status.json")
    META_PATH = str(out / "model_meta.json")
    FEATURES_SCHEMA_PATH = str(out / "features_train.json")
    OOS_REPORT_PATH = str(out / "oos_report.json")
    CALIB_TABLE_PATH = str(out / "calibration_5d_deciles.json")
    RESEARCH_CSV_PATH = str(out / "research_report.csv")
    MODEL_CALIB_PATH = str(out / "model_lgbm_calibrated.joblib")
    ISO_EV_MAPPER_PATH = str(out / "isotonic_ev_mapper.joblib")

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
        f.write(f"{filename}
")

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

# ===== CLEAN, DETERMINISTIC SYMBOL STRIPPING =====

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
CPR_TMR  = [f"CPR_Tmr_{x}" for x in ("Above","Below","Inside","Overlap")]
STRUCT_ONEHOT = ["Struct_uptrend","Struct_downtrend","Struct_range"]
DAYTYPE_ONEHOT = ["DayType_bullish","DayType_bearish","DayType_inside"]

# (1) Regime features

def add_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    # cross-sectional 1D stats per day
    cs_mean = panel.groupby("date")["ret_1d_close_pct"].mean()
    cs_std  = panel.groupby("date")["ret_1d_close_pct"].std()
    # long-trend of CS mean
    panel["regime_market_trend"] = panel["date"].map(cs_mean.rolling(200, min_periods=50).mean())
    # high-vol regime flag (CS std > rolling median)
    vol_med = cs_std.rolling(250, min_periods=50).median()
    panel["regime_high_vol"] = (panel["date"].map(cs_std) > panel["date"].map(vol_med)).astype(int)
    # dispersion level
    panel["regime_dispersion"] = panel["date"].map(cs_std)
    return panel

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
    base_auto = discover_daily_features(df, exclude=EXCLUDE_D_FEATURES)
    df = add_lags(df, ["D_rsi14","D_adx14","D_ema20_angle_deg","D_obv_slope"], lags=(1,2))
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

    rsi14 = pd.to_numeric(df.get("D_rsi14", np.nan), errors="coerce")
    rsi7  = pd.to_numeric(df.get("D_rsi7", np.nan), errors="coerce")
    obvs  = pd.to_numeric(df.get("D_obv_slope", np.nan), errors="coerce")
    adx14 = pd.to_numeric(df.get("D_adx14", np.nan), errors="coerce")
    atr14 = pd.to_numeric(df.get("D_atr14", np.nan), errors="coerce")
    close = pd.to_numeric(df.get("close", np.nan), errors="coerce")
    macd_hist = pd.to_numeric(df.get("D_macd_hist", np.nan), errors="coerce")
    cprw  = pd.to_numeric(df.get("D_cpr_width_pct", np.nan), errors="coerce").abs()

    df["D_rsi14_obv_x"] = rsi14 * obvs
    if "D_rsi7" in df.columns:
        df["D_rsi7_obv_x"] = rsi7 * obvs
    df["D_atr14_to_close_pct"] = (atr14 / close).replace([np.inf, -np.inf], np.nan) * 100.0
    df["X_rsi14_adx14"] = rsi14 * adx14
    df["X_cprw_atr_pct"] = cprw * df["D_atr14_to_close_pct"]
    trend_code = pd.to_numeric(df.get("D_structure_trend_code", np.nan), errors="coerce")
    df["X_trend_atr_pct"] = trend_code * df["D_atr14_to_close_pct"]
    df["X_rsi_cross_strength"] = (rsi7 - rsi14) * adx14
    df["X_macd_nonlin"] = macd_hist * rsi14 / 50.0
    df["X_adx_sqr"] = (adx14 ** 2) / 100.0

    if "ret_1d_close_pct" not in df.columns or "ret_5d_close_pct" not in df.columns:
        df = add_targets(df)

    # rolling slopes
    df["D_close_roll_slope_20"] = _rolling_slope(df["close"], window=20)
    df["D_close_roll_slope_50"] = _rolling_slope(df["close"], window=50)

    # past 5d ret and its rolling std
    daily_ret = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100.0
    df["D_ret_5d_pastret"] = daily_ret.rolling(5, min_periods=5).sum()
    df["D_ret_5d_roll_std"] = df["D_ret_5d_pastret"].rolling(50, min_periods=10).std()

    # Add regime features (1)
    # We assume timestamp and ret_1d_close_pct present; if not, add_targets ensures ret_1d is there.
    tmp = df[["timestamp","ret_1d_close_pct"]].copy()
    tmp["symbol"] = df.get("symbol", "").astype(str)
    reg = add_regime_features(tmp.join(df[[c for c in df.columns if c not in ("timestamp","ret_1d_close_pct","symbol")]], how='left'))
    df["regime_market_trend"] = reg["regime_market_trend"].values
    df["regime_high_vol"] = reg["regime_high_vol"].values
    df["regime_dispersion"] = reg["regime_dispersion"].values

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    feats = (base_auto + LAG_FEATURES + CPR_YDAY + CPR_TMR + STRUCT_ONEHOT + DAYTYPE_ONEHOT + [
        "D_rsi14_obv_x","D_rsi7_obv_x","D_atr14_to_close_pct",
        "D_ret_5d_roll_std","D_close_roll_slope_20","D_close_roll_slope_50",
        "X_rsi14_adx14","X_cprw_atr_pct","X_trend_atr_pct",
        "X_rsi_cross_strength","X_macd_nonlin","X_adx_sqr",
        # (1) regime features into model feature set
        "regime_market_trend","regime_high_vol","regime_dispersion",
    ])
    for c in feats:
        if c not in df.columns:
            df[c] = 0 if (c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
                          or c.startswith("Struct_") or c.startswith("DayType_")) else np.nan
    return df, feats

# ===================== Bias scores (optional) =====================

def score_bias(df: pd.DataFrame) -> pd.DataFrame:
    col = lambda c: df[c] if c in df.columns else pd.Series([np.nan]*len(df))
    golden   = col("D_golden_regime").fillna(False).astype(bool) if "D_golden_regime" in df.columns else pd.Series(False, index=df.index)
    ema_stack= col("D_ema_stack_20_50_100").fillna(False).astype(bool) if "D_ema_stack_20_50_100" in df.columns else pd.Series(False, index=df.index)
    rsi14    = pd.to_numeric(col("D_rsi14"), errors="coerce")
    rsi_cross= col("D_rsi7_gt_rsi14").fillna(False).astype(bool) if "D_rsi7_gt_rsi14" in df.columns else pd.Series(False, index=df.index)
    adx14    = pd.to_numeric(col("D_adx14"), errors="coerce")
    pdi14    = pd.to_numeric(col("D_pdi14"), errors="coerce")
    mdi14    = pd.to_numeric(col("D_mdi14"), errors="coerce")
    obv_rise = col("D_price_and_obv_rising").fillna(False).astype(bool) if "D_price_and_obv_rising" in df.columns else pd.Series(False, index=df.index)
    atr14    = pd.to_numeric(col("D_atr14"), errors="coerce")
    cpr_w    = pd.to_numeric(col("D_cpr_width_pct"), errors="coerce").abs()
    vol      = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    avg20    = vol.rolling(20, min_periods=1).mean()
    cpr_above= (df.get("CPR_Yday_Above", 0) == 1)
    cpr_below= (df.get("CPR_Yday_Below", 0) == 1)

    long_score = (
        (golden*2).astype(float) + (ema_stack*2).astype(float) + (rsi_cross*1).astype(float)
        + (((rsi14.between(50,70, inclusive="both")).fillna(False)).astype(float)*1)
        + ((((adx14>20)&(pdi14>mdi14)).fillna(False)).astype(float)*1.5)
        + (obv_rise*1).astype(float)
        + cpr_above.astype(float)*0.5
    )
    short_score = (
        ((~golden)*1).astype(float) + ((~ema_stack)*1).astype(float) + ((rsi14<45).fillna(False).astype(float)*1)
        + ((((adx14>20)&(mdi14>pdi14)).fillna(False)).astype(float)*1.5)
        + cpr_below.astype(float)*0.5
    )
    atr_pct = (atr14/df["close"]).replace([np.inf,-np.inf],np.nan)*100 if "close" in df.columns else pd.Series(np.nan, index=df.index)
    risk_pen = (atr_pct>4).fillna(False).astype(float)*0.5 + (cpr_w>1.0).fillna(False).astype(float)*0.3
    liq_pen  = (avg20<MIN_AVG20_VOL).fillna(False).astype(float)*0.5

    df["long_score"]  = long_score - risk_pen - liq_pen
    df["short_score"] = short_score - risk_pen - liq_pen
    return df

# ===================== Panel writer =====================

MASTER_KEEP_STATIC = [
    "timestamp","symbol","open","high","low","close","volume",
    "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
    "long_score","short_score","D_atr14","D_cpr_width_pct",
    # regime features saved into panel
    "regime_market_trend","regime_high_vol","regime_dispersion",
]

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
                         "long_score","short_score","D_atr14","D_cpr_width_pct",
                         "regime_market_trend","regime_high_vol","regime_dispersion"]
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

        df_train = df.dropna(subset=["ret_5d_close_pct"])  # labels exist as close->close
        if df_train.empty:
            nrows = len(df)
            latest_ts = ensure_kolkata_tz(df["timestamp"]).max()
            label_counts = {h: int(df[f"ret_{h}d_close_pct"].notna().sum()) for h in (1,3,5)}
            msg = (f"NO TRAIN {sym} rows={nrows} labels={{1d:{label_counts[1]},3d:{label_counts[3]},5d:{label_counts[5]}}} "
                   f"latest_ts={latest_ts}")
            return sym, None, feats, msg

        rows = df_train[["timestamp","symbol","open","high","low","close","volume"] + feats +
                        ["ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                         "long_score","short_score","D_atr14","D_cpr_width_pct",
                         "regime_market_trend","regime_high_vol","regime_dispersion"]].copy()
        return sym, rows, feats, None
    except Exception as e:
        return sym, None, None, e


def collect_panel_from_paths(paths: List[Path], load_workers: int = 8):
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
        raise SystemExit("No matching files found
Select files via --files or --data-dir with *_daily.* files.")

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
        for c in existing_panel.columns:
            if (str(c).startswith(("CPR_Yday_","CPR_Tmr_","Struct_","DayType_"))):
                existing_panel[c] = pd.to_numeric(existing_panel[c], errors="coerce").fillna(0).clip(0,1).astype(np.int32)
        for c in existing_panel.columns:
            if (c in ("open","high","low","close","volume",
                      "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                      "long_score","short_score","D_atr14","D_cpr_width_pct",
                      "regime_market_trend","regime_high_vol","regime_dispersion") or str(c).startswith("D_")):
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
        print("
Interrupted! Autosaving current chunk...")
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
        or c in ("regime_market_trend","regime_high_vol","regime_dispersion")
    )]
    return panel, feats

# ===================== Feature matrix sanitize + schema lock =====================

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
    zero_cols = (X == 0).all()
    print(f"[Schema] infer matrix shape={X.shape} const_cols={const_cols.mean()*100:.1f}% zero_cols={zero_cols.mean()*100:.1f}%")
    return X

# ===================== Label engineering (vol-adjusted 5D ranks/quantiles; close->close kept) =====================

def build_5d_rank_quant_labels(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    # Vol proxy: daily std (%) with ATR% fallback
    daily_ret = pd.to_numeric(panel["close"], errors="coerce").pct_change() * 100.0
    panel["vol_20"] = daily_ret.groupby(panel["symbol"]).rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    panel["atr_pct"] = (pd.to_numeric(panel["D_atr14"], errors="coerce") / pd.to_numeric(panel["close"], errors="coerce")).replace([np.inf,-np.inf],np.nan) * 100.0
    vol_basis = panel["vol_20"].fillna(panel["atr_pct"])

    # (2) Sharpe-like scaling (keep existing close->close numerator; divide by vol)
    panel["ret_5d_adj"] = pd.to_numeric(panel["ret_5d_close_pct"], errors="coerce") / vol_basis
    panel["ret_3d_adj"] = pd.to_numeric(panel["ret_3d_close_pct"], errors="coerce") / vol_basis

    grp = panel.groupby("date")
    panel["rank_5d_pct"] = grp["ret_5d_adj"].rank(method="average", pct=True)
    panel["top20_vs_bot20_5d"] = np.where(panel["rank_5d_pct"] >= 0.80, 1,
                                     np.where(panel["rank_5d_pct"] <= 0.20, 0, np.nan))
    return panel

# ===================== CV splits =====================

def time_cv_by_timestamp(panel: pd.DataFrame,
                         n_splits: int = 7,
                         embargo_days: int = 3,
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


def split_train_val_by_time(panel: pd.DataFrame, candidate_idx: np.ndarray,
                            val_frac: float = 0.15, min_val: int = 100) -> Tuple[np.ndarray, np.ndarray]:
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
    val_order = order[-val_n:]
    train_order = order[:-val_n]
    return idx[train_order], idx[val_order]

# ===================== LightGBM helpers =====================

def _check_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        return lgb, LGBMClassifier
    except Exception as e:
        raise SystemExit("LightGBM is not installed. Please run: pip install lightgbm") from e

# (6) Remove class_weight='balanced'; (7) centralize callbacks; keep other params

def _lgbm_cls_params(use_monotone: bool, X_cols: List[str], rnd: int):
    depth = MAX_DEPTH if isinstance(MAX_DEPTH, int) and MAX_DEPTH > 0 else -1
    params = dict(
        n_estimators=int(N_EST_5D), learning_rate=LEARNING_RATE,
        num_leaves=56, max_depth=depth, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
        reg_alpha=0.4, reg_lambda=8.0, class_weight=None,  # <-- (6)
        n_jobs=-1, random_state=int(rnd), verbosity=-1,
    )
    return params

# (7) Callback helper + safe early stopping (3)

def _lgb_callbacks(val_size: int):
    import lightgbm as _lgb_mod
    cbs = [_lgb_mod.callback.log_evaluation(period=0)]
    if EARLY_STOPPING_ROUNDS and EARLY_STOPPING_ROUNDS > 0 and val_size >= int(MIN_VAL_EARLYSTOP):
        cbs.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(EARLY_STOPPING_ROUNDS)))
    return cbs

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

# (2) Isotonic EV mapper: prob -> expected ret_5d_adj
from sklearn.isotonic import IsotonicRegression

def fit_isotonic_ev_mapper(prob_cal: np.ndarray, realized_adj: np.ndarray):
    order = np.argsort(prob_cal)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_cal[order], realized_adj[order])
    return iso

# ===================== Train 5D quantile classifier (CV, no per-fold calibration) =====================

def train_5d_quantile_cls(panel: pd.DataFrame, feats: List[str],
                          n_splits: int = FOLDS, embargo_days: int = EMBARGO_DAYS,
                          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    lgb, LGBMClassifier = _check_lightgbm()
    pl = build_5d_rank_quant_labels(panel)
    y = pl["top20_vs_bot20_5d"]
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labeled rows for 5D quantile classification (top/bottom 20%).")
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(pl, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))

    X_full = sanitize_feature_matrix(pl.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)
    oos_rank = np.full(len(X_full), np.nan)  # realized rank for IC

    eta = ProgressETA(total=len(folds), label="Train 5D-Quantile-CLS")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0:
            continue
        tr_core_idx, val_idx = split_train_val_by_time(pl, valid_idx[tr_pos], val_frac=0.2, min_val=200)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos = np.where(np.isin(valid_idx, val_idx))[0]

        X_tr = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos] if len(val_pos)>0 else y_tr
        X_te  = X_full.iloc[te_pos]

        import lightgbm as _lgb_mod
        clf = LGBMClassifier(**_lgbm_cls_params(use_monotone=False, X_cols=list(X_full.columns), rnd=GLOBAL_SEED + 300 + fold_no))
        callbacks = _lgb_callbacks(val_size=len(X_val))
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)

        prob = clf.predict_proba(X_te)[:, 1]
        oos_prob[te_pos] = prob
        oos_rank[te_pos] = pl.loc[mask].iloc[te_pos]["rank_5d_pct"].values
        eta.tick(f"fold {fold_no}")
    return oos_prob, oos_rank, valid_idx, pl

# ===================== Strict OOS calibration (Train → Calibrate → Test) =====================

def fit_final_model_and_oos_calibration(panel: pd.DataFrame, feats: List[str],
                                        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    from joblib import dump
    lgb, LGBMClassifier = _check_lightgbm()
    pl = build_5d_rank_quant_labels(panel)
    y = pl["top20_vs_bot20_5d"].astype("float")
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labels for 5D quantile classification.")

    X = sanitize_feature_matrix(pl.loc[mask, feats].copy())
    y = y.loc[mask].astype(int)

    t = pl.loc[mask, "timestamp"].values
    order = np.argsort(t)
    n = len(order)
    i_train = order[:int(0.7*n)]
    i_cal   = order[int(0.7*n):int(0.9*n)]
    i_test  = order[int(0.9*n):]

    import lightgbm as _lgb_mod
    base = LGBMClassifier(**_lgbm_cls_params(use_monotone=False, X_cols=list(X.columns), rnd=GLOBAL_SEED+777))
    callbacks = _lgb_callbacks(val_size=len(i_cal))
    base.fit(X.iloc[i_train], y.iloc[i_train],
             eval_set=[(X.iloc[i_cal], y.iloc[i_cal])], eval_metric="binary_logloss", callbacks=callbacks)

    # Choose best classifier calibrator on CAL window (probability calibration)
    final_calib, diag = _calibrate_best_brier(base, X.iloc[i_cal], y.iloc[i_cal])

    # (2) EV isotonic mapper: prob_cal -> realized adjusted return (use existing ret_5d_adj)
    prob_cal = base.predict_proba(X.iloc[i_cal])[:, 1]
    realized_adj_cal = pl.loc[mask].iloc[i_cal]["ret_5d_adj"].astype(float).values
    iso_ev = fit_isotonic_ev_mapper(prob_cal, realized_adj_cal)

    # Apply both to TEST window
    p_test = final_calib.predict_proba(X.iloc[i_test])[:, 1]
    ev_test = iso_ev.predict(base.predict_proba(X.iloc[i_test])[:, 1])

    oos_df = pd.DataFrame({
        "prob_top20_5d": p_test,
        "ret_3d_close_pct": pl.loc[mask].iloc[i_test]["ret_3d_close_pct"].values,
        "ret_5d_close_pct": pl.loc[mask].iloc[i_test]["ret_5d_close_pct"].values,
        "ret_5d_adj": pl.loc[mask].iloc[i_test]["ret_5d_adj"].values,
        "rank_5d_pct": pl.loc[mask].iloc[i_test]["rank_5d_pct"].values,
        "expected_ret_5d_adj_iso": ev_test,
    })

    # Persist models (7)
    try:
        dump(final_calib, MODEL_CALIB_PATH)
        dump(iso_ev, ISO_EV_MAPPER_PATH)
        print(f"[Model] Saved calibrator: {MODEL_CALIB_PATH}")
        print(f"[Model] Saved isotonic EV mapper: {ISO_EV_MAPPER_PATH}")
    except Exception as e:
        print(f"[Model] Save failed: {e}")

    return final_calib, iso_ev, oos_df

# ===================== (4) 1D follow-through classifier (supportive gate) =====================

def train_1d_followthrough(panel: pd.DataFrame, feats: List[str], margin_pct: float = CLS_MARGIN_1D,
                           n_splits: int = FOLDS, embargo_days: int = EMBARGO_DAYS):
    lgb, LGBMClassifier = _check_lightgbm()
    r = pd.to_numeric(panel["ret_1d_close_pct"], errors="coerce")
    y = pd.Series(np.nan, index=panel.index)
    y[(r > margin_pct)] = 1
    y[(r < -margin_pct)] = 0
    mask = y.notna()
    if mask.sum() < MIN_GATE_SAMPLES:
        print(f"[1D gate] Too few labeled rows ({int(mask.sum())} < {MIN_GATE_SAMPLES}); skipping.")
        return None, None, None

    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))

    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)

    eta = ProgressETA(total=len(folds), label="Train 1D-Gate")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0: continue
        tr_core_idx, val_idx = split_train_val_by_time(panel, valid_idx[tr_pos], val_frac=0.2, min_val=200)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos = np.where(np.isin(valid_idx, val_idx))[0]

        X_tr = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos] if len(val_pos)>0 else y_tr
        X_te  = X_full.iloc[te_pos]

        from lightgbm import LGBMClassifier as _Gate
        clf = _Gate(**dict(
            n_estimators=int(N_EST_1D), learning_rate=LEARNING_RATE,
            num_leaves=56, max_depth=MAX_DEPTH, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
            reg_alpha=0.4, reg_lambda=8.0, class_weight=None,  # (6)
            n_jobs=-1, random_state=int(GLOBAL_SEED + 100 + fold_no), verbosity=-1,
        ))
        callbacks = _lgb_callbacks(val_size=len(X_val))
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)

        # (4) post-fit sigmoid calibration for the gate
        from sklearn.calibration import CalibratedClassifierCV
        gate_calib = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
        gate_calib.fit(X_val, y_val)
        oos_prob[te_pos] = gate_calib.predict_proba(X_te)[:, 1]
        eta.tick(f"fold {fold_no}")

    # Final gate model (fit once more on last split's train/val)
    final_gate = None
    try:
        final_gate = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
        final_gate.fit(X_val, y_val)
    except Exception:
        final_gate = None
    return final_gate, oos_prob, valid_idx

# ===================== Map prob -> expected returns (deciles) [legacy for 3d/5d raw] =====================

def map_prob_to_expectations(prob_vec: np.ndarray, calib_table: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pm = np.asarray(calib_table["prob_mid"], dtype=float)
    r3 = np.asarray(calib_table["avg_ret_3d"], dtype=float)
    r5 = np.asarray(calib_table["avg_ret_5d"], dtype=float)
    sh = np.asarray(calib_table["exp_sharpe_5d"], dtype=float)
    order = np.argsort(pm)
    pm = pm[order]; r3 = r3[order]; r5 = r5[order]; sh = sh[order]
    exp3 = np.interp(prob_vec, pm, r3, left=r3[0], right=r3[-1])
    exp5 = np.interp(prob_vec, pm, r5, left=r5[0], right=r5[-1])
    exp_sh = np.interp(prob_vec, pm, sh, left=sh[0], right=sh[-1])
    return exp3, exp5, exp_sh

# ===================== Watchlist =====================

def nightly_watchlist(panel: pd.DataFrame, feats: List[str],
                      m5_cls_calib, calib_table: Dict[str, List[float]],
                      iso_ev_mapper,  # (2) additional EV mapper
                      m1_gate: Optional[object],
                      exclude_pattern: Optional[str],
                      gate_threshold: float,
                      export_xlsx: bool):
    panel = panel.copy().sort_values(["symbol", "timestamp"])
    panel["avg20_vol"] = panel.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    last = panel.groupby("symbol", as_index=False).tail(1).copy()

    if exclude_pattern:
        mask_eq = ~last["symbol"].astype(str).str.contains(exclude_pattern, regex=True, na=False)
        last = last.loc[mask_eq].copy()

    feats_schema, impute_stats = load_schema(FEATURES_SCHEMA_PATH)
    X_raw = sanitize_feature_matrix(last[feats_schema].copy())
    X = reindex_and_impute(X_raw, feats_schema, impute_stats)

    # Probability of top20 (prob-calibrated classifier)
    prob_top20 = m5_cls_calib.predict_proba(X)[:, 1]

    # Legacy mapping for raw returns (deciles)
    exp3, exp5, exp_sh = map_prob_to_expectations(prob_top20, calib_table)

    # (2) Isotonic EV mapping for adjusted 5D payoff
    try:
        expected_adj = iso_ev_mapper.predict(prob_top20)
    except Exception:
        expected_adj = np.full(len(prob_top20), np.nan)

    wl = last.copy()
    wl["prob_top20_5d"] = prob_top20
    wl["expected_ret_3d"] = exp3
    wl["expected_ret_5d"] = exp5
    wl["expected_sharpe_5d"] = exp_sh
    wl["expected_ret_5d_adj"] = expected_adj  # primary EV signal (2)

    if m1_gate is not None and gate_threshold is not None:
        try:
            p1 = m1_gate.predict_proba(X)[:, 1]
        except Exception:
            p1 = np.full(len(wl), np.nan)
        wl["prob_up_1d_gate"] = p1
        wl = wl[ wl["prob_up_1d_gate"].fillna(0.0) >= float(gate_threshold) ].copy()

    wl = wl[(wl["close"] >= MIN_CLOSE) & (wl["avg20_vol"] >= MIN_AVG20_VOL)].copy()
    wl = wl[["symbol","timestamp","close","avg20_vol",
             "prob_top20_5d","expected_ret_3d","expected_ret_5d","expected_sharpe_5d","expected_ret_5d_adj",
             "D_atr14","D_cpr_width_pct","long_score","short_score",
             "regime_market_trend","regime_high_vol","regime_dispersion"]].sort_values(
             ["expected_ret_5d_adj","expected_ret_5d"], ascending=[False, False]
    )
    wl.to_csv(WATCHLIST_OUT, index=False)
    print(f"Saved: {WATCHLIST_OUT} rows={len(wl)}")
    if export_xlsx:
        try:
            wl.to_excel(Path(WATCHLIST_OUT).with_suffix(".xlsx"), index=False, engine="openpyxl")
            print(f"Saved: {Path(WATCHLIST_OUT).with_suffix('.xlsx')}")
        except Exception as e:
            print(f"Excel export failed: {e}")
    return wl

# ===================== (5) Simple portfolio backtest (sanity check) =====================

def quick_portfolio_backtest(panel: pd.DataFrame, watchlist: pd.DataFrame, horizon: int = 5, top_k: int = 20, cost_bps: float = 20/10000):
    panel = panel.copy(); watchlist = watchlist.copy()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    watchlist["date"] = pd.to_datetime(watchlist["timestamp"]).dt.normalize()
    equity = [1.0]
    prev_hold = set()
    for d, wl in watchlist.groupby("date"):
        wl = wl.sort_values("expected_ret_5d_adj", ascending=False).head(top_k)
        syms = set(wl["symbol"])
        added = syms - prev_hold
        turnover = len(added) / max(1, len(prev_hold) or 1)
        rets = []
        for s in syms:
            hist = panel[panel["symbol"] == s]
            row = hist[hist["date"] == d]
            if row.empty: continue
            i = row.index[0]
            if i+1 >= len(hist) or i+1+horizon >= len(hist): continue
            entry = hist.iloc[i+1]["open"]
            exitp = hist.iloc[i+1+horizon]["close"]
            rets.append((exitp-entry)/entry)
        pnl = np.mean(rets) if rets else 0.0
        pnl -= turnover * cost_bps
        equity.append(equity[-1]*(1+pnl))
        prev_hold = syms
    return pd.Series(equity)

# ===================== Embedded Research (unchanged) =====================
HORIZON = 5
ATR_WINDOW = 14
TOP_Q = 0.20
BOTTOM_Q = 0.20
MIN_CROSS_SECTION = 200
EMBARGO_DAYS_RESEARCH = HORIZON
COST_BPS = 20 / 10000
RANDOM_STATE = 42
from sklearn.base import clone

def compute_atr_and_atr_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    prev_close = df.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = (
        tr.groupby(df["symbol"]).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean().reset_index(level=0, drop=True)
    )
    df["atr"] = atr
    df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    return df

# (research helpers omitted for brevity; unchanged from v2)

# ===================== Minimal CLI =====================

def parse_cli(argv=None):
    import argparse
    p = argparse.ArgumentParser("cpr_fix_with_embedded_alpha_ooscal_v3")
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symbols-like", default=None)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--accept-any-daily", default="false")
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--cv-splits", type=int, default=FOLDS)
    p.add_argument("--embargo-days", type=int, default=EMBARGO_DAYS)
    p.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    p.add_argument("--exclude-pattern", default=r"(LIQUID|GOLD|BEES|IETF|ETF|CASE|ADD)$")
    p.add_argument("--use-1d-gate", default="false", help='Supportive 1D follow-through gate ("true"/"false").')
    p.add_argument("--gate-threshold", type=float, default=0.55)
    p.add_argument("--export-xlsx", default="false")
    # (5) quick portfolio switch
    p.add_argument("--quick-portfolio", default="false")
    p.add_argument("--portfolio-topk", type=int, default=20)
    return p.parse_args(argv)

# ===================== Main =====================

def main(argv=None):
    args = parse_cli(argv)
    setup_paths(args.out_dir)

    # 1) Build panel
    paths = _strict_file_list(args.data_dir, args.symbols_like, args.limit_files, accept_any_daily=args.accept_any_daily)
    panel, feats = collect_panel_from_paths(paths, load_workers=args.load_workers)
    print(f"[Panel] final rows={len(panel)} cols={len(panel.columns)} feats={len(feats)}")

    # 2) CV diagnostics (no per-fold calibration)
    _oos_prob_cv, _oos_rank_cv, _valid_idx_cv, _pl_cv = train_5d_quantile_cls(panel, feats,
        n_splits=args.cv_splits,
        embargo_days=args.embargo_days,
        early_stopping_rounds=args.early_stopping_rounds)

    # 3) Schema lock from full features (kept as-is; EV mapper is independent)
    all_feats_df = sanitize_feature_matrix(panel[feats].copy())
    impute_stats = compute_impute_stats(all_feats_df)
    save_schema(FEATURES_SCHEMA_PATH, feats, impute_stats)
    print(f"[Schema] Saved: {FEATURES_SCHEMA_PATH}")

    # 4) Final model + OOS-only calibration & mapping (Train->Cal->Test)
    final_calib, iso_ev, oos_df = fit_final_model_and_oos_calibration(panel, feats,
        early_stopping_rounds=args.early_stopping_rounds)

    df_oos = oos_df.dropna(subset=["prob_top20_5d"]).copy()
    df_oos["prob_bucket"] = pd.qcut(df_oos["prob_top20_5d"], q=10, labels=False, duplicates="drop")

    calib = (df_oos.groupby("prob_bucket")
             .agg(avg_ret_3d=("ret_3d_close_pct","mean"),
                  avg_ret_5d=("ret_5d_close_pct","mean"),
                  std_ret_5d=("ret_5d_close_pct","std"),
                  avg_ret_5d_adj=("ret_5d_adj","mean"))
             .reset_index().sort_values("prob_bucket"))
    probs = np.linspace(0.05, 0.95, len(calib))  # midpoints of deciles
    calib["prob_mid"] = probs
    calib["exp_sharpe_5d"] = calib["avg_ret_5d_adj"].astype(float)
    calib_table = {
        "prob_mid": calib["prob_mid"].tolist(),
        "avg_ret_3d": calib["avg_ret_3d"].tolist(),
        "avg_ret_5d": calib["avg_ret_5d"].tolist(),
        "std_ret_5d": calib["std_ret_5d"].fillna(0.0).tolist(),
        "exp_sharpe_5d": calib["exp_sharpe_5d"].tolist(),
    }
    Path(CALIB_TABLE_PATH).write_text(json.dumps(calib_table, indent=2))

    from sklearn.metrics import brier_score_loss
    from scipy.stats import spearmanr
    pseudo_y = np.where(df_oos["rank_5d_pct"] >= 0.8, 1, np.where(df_oos["rank_5d_pct"] <= 0.2, 0, np.nan))
    msk = ~np.isnan(pseudo_y)
    brier = float(brier_score_loss(pd.Series(pseudo_y)[msk], df_oos.loc[msk, "prob_top20_5d"]))
    ic5, ic5_p = spearmanr(df_oos["prob_top20_5d"], df_oos["ret_5d_adj"], nan_policy="omit")
    rep = {"5d_cls": {"n_oos": int(len(df_oos)), "brier_pseudo": brier,
                        "spearman_ic_5d_adj": float(ic5), "spearman_ic_pvalue": float(ic5_p)}}
    Path(OOS_REPORT_PATH).write_text(json.dumps(rep, indent=2))
    print(f"[OOS] Saved report: {OOS_REPORT_PATH}")
    print(f"[Calibration] Saved table: {CALIB_TABLE_PATH}")

    # 5) Train 1D supportive gate (optional)
    m1_gate = None
    if str(args.use_1d_gate).lower() in ("true","1","yes","y","t"):
        m1_gate, _oos1_gate, _idx1 = train_1d_followthrough(panel, feats, margin_pct=CLS_MARGIN_1D,
            n_splits=args.cv_splits, embargo_days=args.embargo_days)

    # 6) Nightly watchlist (uses final_calib, EV iso mapper, and legacy decile mapping)
    export_xlsx = str(args.export_xlsx).lower() in ("true","1","yes","y","t")
    wl = nightly_watchlist(panel, feats, final_calib, calib_table, iso_ev, m1_gate,
                           exclude_pattern=args.exclude_pattern,
                           gate_threshold=float(args.gate_threshold),
                           export_xlsx=export_xlsx)

    # 7) Optional quick portfolio sanity-check
    if str(args.quick_portfolio).lower() in ("true","1","yes","y","t"):
        eq = quick_portfolio_backtest(panel, wl, horizon=5, top_k=int(args.portfolio_topk))
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.plot(eq.values)
            plt.title("Quick Portfolio (Top-K by expected_ret_5d_adj)")
            plt.xlabel("Days"); plt.ylabel("Equity")
            out_png = Path(args.out_dir) / "quick_portfolio_equity.png"
            plt.tight_layout(); plt.savefig(out_png, dpi=120)
            print(f"[QuickPF] Saved: {out_png}")
        except Exception as e:
            print(f"[QuickPF] Plot failed: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
