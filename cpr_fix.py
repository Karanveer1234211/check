
# cpr_fix_v3_2_lib.py
# ----------------------------------------------------------------------------------------------------
# v3.2 (library mode, NO CLI)
# - Public API: run_pipeline(...)
# - TODAY-BASED INFERENCE: panel now retains ALL rows (even without labels),
#   so nightly watchlist uses the latest cached date.
# - Regime features: cross-sectional by date on FULL panel, LAGGED by 1 day.
# - EV mapper trained & applied on CALIBRATED probabilities (consistency fix).
# - 1-day gate: mandatory; final refit on FULL labeled history with 80/20
#   time-ordered Train/Cal split.
# - Embargo default increased to 5.
# - Train-only imputation medians saved to schema and reused for CAL/TEST + live.
# - Early symbol exclude (ETF/illiquid regex) is applied BEFORE modeling.
# - Optional EV target: 'cc' (close→close, default) or 'oc' (next open→future close).
# ----------------------------------------------------------------------------------------------------
import os, glob, json, time, sys, re, math, concurrent.futures
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
CALIB_TABLE_PATH = None      # <out-dir>/calibration_5d_deciles.json
RESEARCH_CSV_PATH = None     # <out-dir>/research_report.csv
MODEL_CALIB_PATH = None      # <out-dir>/model_lgbm_calibrated.joblib (optional)
ISO_EV_MAPPER_PATH = None    # <out-dir>/isotonic_ev_mapper.joblib (optional)
# ===================== MODEL / CV =====================
GLOBAL_SEED = 42
FOLDS = 7
EMBARGO_DAYS = 5  # increased to match 5d horizon
# LightGBM baseline
N_EST_1D = 2400
N_EST_5D = 2800
EARLY_STOPPING_ROUNDS = 200
LEARNING_RATE = 0.009
MAX_DEPTH = 6
# Safe early stopping threshold
MIN_VAL_EARLYSTOP = 500
# Gate controls
MIN_GATE_SAMPLES = 500  # mandatory gate requires at least this many labeled rows
CLS_MARGIN_1D = 0.10
# Filters for watchlist
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000
# Chunking
CHUNK_SIZE = 1200
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
        # OC variant (next open to future close)
        df[f"ret_{h}d_oc_pct"] = (df["close"].shift(-h) / df["open"].shift(-1) - 1) * 100
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
CPR_TMR = [f"CPR_Tmr_{x}" for x in ("Above","Below","Inside","Overlap")]
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
    base_auto = discover_daily_features(df, exclude=EXCLUDE_D_FEATURES)
    df = add_lags(df, ["D_rsi14","D_adx14","D_ema20_angle_deg","D_obv_slope"], lags=(1,2))
    # Uniform CPR and day-type categoricals
    yday_unified = _unify_categorical(df, "D_cpr_vs_yday")
    tmr_unified = _unify_categorical(df, "D_tmr_cpr_vs_today")
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
    # Interactions / engineered
    rsi14 = pd.to_numeric(df.get("D_rsi14", np.nan), errors="coerce")
    rsi7 = pd.to_numeric(df.get("D_rsi7", np.nan), errors="coerce")
    obvs = pd.to_numeric(df.get("D_obv_slope", np.nan), errors="coerce")
    adx14 = pd.to_numeric(df.get("D_adx14", np.nan), errors="coerce")
    atr14 = pd.to_numeric(df.get("D_atr14", np.nan), errors="coerce")
    close = pd.to_numeric(df.get("close", np.nan), errors="coerce")
    macd_hist = pd.to_numeric(df.get("D_macd_hist", np.nan), errors="coerce")
    cprw = pd.to_numeric(df.get("D_cpr_width_pct", np.nan), errors="coerce").abs()
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
    df["D_close_roll_slope_20"] = _rolling_slope(df["close"], window=20)
    df["D_close_roll_slope_50"] = _rolling_slope(df["close"], window=50)
    daily_ret = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100.0
    df["D_ret_5d_pastret"] = daily_ret.rolling(5, min_periods=5).sum()
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
    for c in feats:
        if c not in df.columns:
            df[c] = 0 if (c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
                          or c.startswith("Struct_") or c.startswith("DayType_")) else np.nan
    return df, feats
# ===================== Panel writer =====================
MASTER_KEEP_STATIC = [
    "timestamp","symbol","open","high","low","close","volume",
    "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
    "ret_1d_oc_pct","ret_3d_oc_pct","ret_5d_oc_pct",
    "long_score","short_score","D_atr14","D_cpr_width_pct",
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
                        "ret_1d_oc_pct","ret_3d_oc_pct","ret_5d_oc_pct",
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
        return (df.loc[last.index, ["symbol","timestamp"]
                ].set_index("symbol")["timestamp"].to_dict())
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
        # simple sanity check
        if "D_rsi14" in df.columns:
            s = pd.to_numeric(df["D_rsi14"], errors="coerce")
            bad = (~s.between(0,100)) & s.notna()
            if bad.any():
                _log_load_error(sym, str(path_obj), f"Range anomaly D_rsi14 on {int(bad.sum())} rows")
        # targets + features
        df = add_targets(df)
        df, feats = featureize(df)
        # lightweight bias block
        def _bias_block(d):
            col = lambda c: d[c] if c in d.columns else pd.Series([np.nan]*len(d))
            rsi14 = pd.to_numeric(col("D_rsi14"), errors="coerce")
            atr14 = pd.to_numeric(col("D_atr14"), errors="coerce")
            close = pd.to_numeric(col("close"), errors="coerce")
            atr_pct = (atr14/close).replace([np.inf,-np.inf],np.nan)*100
            cpr_w = pd.to_numeric(col("D_cpr_width_pct"), errors="coerce").abs()
            long_score = ((rsi14.between(50,70, inclusive="both")).fillna(False)).astype(float)
            short_score= ((rsi14<45).fillna(False)).astype(float)
            risk_pen = (atr_pct>4).fillna(False).astype(float)*0.5 + (cpr_w>1.0).fillna(False).astype(float)*0.3
            d["long_score"] = long_score - risk_pen
            d["short_score"] = short_score - risk_pen
            return d
        df = _bias_block(df)
        df["symbol"] = sym
        # v3.2: WRITE ALL ROWS (keep unlabeled for today-based inference)
        rows = df[["timestamp","symbol","open","high","low","close","volume"] + feats +
                  ["ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                   "ret_1d_oc_pct","ret_3d_oc_pct","ret_5d_oc_pct",
                   "long_score","short_score","D_atr14","D_cpr_width_pct"]].copy()
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
        raise SystemExit("No matching files found Select files via *_daily.* files.")
    min_ts_map = last_ts_by_symbol_from_panel(PANEL_OUT)
    eta = ProgressETA(total=total, label="Load+Engineer")
    chunk: List[pd.DataFrame] = []
    total_rows_written = 0
    feats: Optional[List[str]] = None
    writer = PanelParquetWriter(PANEL_OUT)
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
        print("Interrupted! Autosaving current chunk...")
        if chunk: append_panel_rows_parquet(writer, chunk); chunk.clear()
        writer.close(); raise
    if chunk: append_panel_rows_parquet(writer, chunk); chunk.clear()
    writer.close()
    print(f"[Panel] Appended new rows: {total_rows_written}")
    # ---- Load the full (combined) panel and compute cross-sectional regime features ----
    panel = pd.read_parquet(PANEL_OUT)
    panel["symbol"] = panel["symbol"].astype(str).map(_clean_symbol_label)
    panel["timestamp"] = ensure_kolkata_tz(pd.to_datetime(panel["timestamp"], errors="coerce"))
    panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    # Build date and ensure 1D returns exist per symbol
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    if "ret_1d_close_pct" not in panel.columns or panel["ret_1d_close_pct"].isna().all():
        panel["ret_1d_close_pct"] = panel.groupby("symbol")["close"].pct_change() * 100.0
    # Cross-sectional (by date) stats, then lag by 1 day to avoid any same-day info
    cs_mean = panel.groupby("date")["ret_1d_close_pct"].mean()
    cs_std = panel.groupby("date")["ret_1d_close_pct"].std()
    trend = cs_mean.rolling(200, min_periods=50).mean().shift(1)
    std_lag = cs_std.shift(1)
    vol_med = cs_std.rolling(250, min_periods=50).median().shift(1)
    panel["regime_market_trend"] = panel["date"].map(trend)
    panel["regime_high_vol"] = (panel["date"].map(std_lag) > panel["date"].map(vol_med)).astype(int)
    panel["regime_dispersion"] = panel["date"].map(std_lag)
    panel = panel.drop(columns=["date"])
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
    return X
# ===================== EV label engineering =====================
def build_5d_rank_quant_labels(panel: pd.DataFrame, ev_target: str = "cc") -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    daily_ret = pd.to_numeric(panel["close"], errors="coerce").pct_change() * 100.0
    panel["vol_20"] = daily_ret.groupby(panel["symbol"]).rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    panel["atr_pct"] = (pd.to_numeric(panel["D_atr14"], errors="coerce") / pd.to_numeric(panel["close"], errors="coerce")).replace([np.inf,-np.inf],np.nan) * 100.0
    vol_basis = panel["vol_20"].fillna(panel["atr_pct"]).replace(0.0, np.nan)
    if ev_target == "oc":
        r5 = pd.to_numeric(panel["ret_5d_oc_pct"], errors="coerce")
        r3 = pd.to_numeric(panel["ret_3d_oc_pct"], errors="coerce")
    else:
        r5 = pd.to_numeric(panel["ret_5d_close_pct"], errors="coerce")
        r3 = pd.to_numeric(panel["ret_3d_close_pct"], errors="coerce")
    panel["ret_5d_adj"] = r5 / vol_basis
    panel["ret_3d_adj"] = r3 / vol_basis
    grp = panel.groupby("date")
    panel["rank_5d_pct"] = grp["ret_5d_adj"].rank(method="average", pct=True)
    panel["top20_vs_bot20_5d"] = np.where(panel["rank_5d_pct"] >= 0.80, 1,
                                  np.where(panel["rank_5d_pct"] <= 0.20, 0, np.nan))
    return panel
# ===================== CV splits =====================
def time_cv_by_timestamp(panel: pd.DataFrame,
                         n_splits: int = 7,
                         embargo_days: int = 5,
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
        return np.array([] ,dtype=int), np.array([], dtype=int)
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
def _lgbm_cls_params(rnd: int):
    depth = MAX_DEPTH if isinstance(MAX_DEPTH, int) and MAX_DEPTH > 0 else -1
    params = dict(
        n_estimators=int(N_EST_5D), learning_rate=LEARNING_RATE,
        num_leaves=56, max_depth=depth, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
        reg_alpha=0.4, reg_lambda=8.0, class_weight=None,
        n_jobs=-1, random_state=int(rnd), verbosity=-1,
    )
    return params
def _lgb_callbacks(val_size: int):
    import lightgbm as _lgb_mod
    cbs = [_lgb_mod.callback.log_evaluation(period=0)]
    if EARLY_STOPPING_ROUNDS and EARLY_STOPPING_ROUNDS > 0 and val_size >= int(MIN_VAL_EARLYSTOP):
        cbs.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(EARLY_STOPPING_ROUNDS)))
    return cbs
# ===================== Calibration helpers =====================
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
from sklearn.isotonic import IsotonicRegression
def fit_isotonic_ev_mapper(prob_calibrated: np.ndarray, realized_adj: np.ndarray):
    order = np.argsort(prob_calibrated)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_calibrated[order], realized_adj[order])
    return iso
# ===================== Train 5D quantile classifier (CV diagnostics) =====================
def train_5d_quantile_cls(panel: pd.DataFrame, feats: List[str], ev_target: str,
                          n_splits: int = FOLDS, embargo_days: int = EMBARGO_DAYS,
                          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    lgb, LGBMClassifier = _check_lightgbm()
    pl = build_5d_rank_quant_labels(panel, ev_target=ev_target)
    y = pl["top20_vs_bot20_5d"]
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labeled rows for 5D quantile classification (top/bottom 20%).")
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(pl, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(pl.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)
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
        X_te = X_full.iloc[te_pos]
        clf = LGBMClassifier(**_lgbm_cls_params(GLOBAL_SEED + 300 + fold_no))
        callbacks = _lgb_callbacks(len(X_val))
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)
        prob = clf.predict_proba(X_te)[:, 1]
        oos_prob[te_pos] = prob
        eta.tick(f"fold {fold_no}")
    return oos_prob, valid_idx, pl
# ===================== Final Train → Calibrate → Test (EV mapper fixed) =====================
def fit_final_model_and_oos_calibration(panel: pd.DataFrame, feats: List[str], ev_target: str,
                                        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    lgb, LGBMClassifier = _check_lightgbm()
    pl = build_5d_rank_quant_labels(panel, ev_target=ev_target)
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
    from lightgbm import LGBMClassifier as _LGB
    base = _LGB(**_lgbm_cls_params(GLOBAL_SEED+777))
    callbacks = _lgb_callbacks(len(i_cal))
    base.fit(X.iloc[i_train], y.iloc[i_train],
             eval_set=[(X.iloc[i_cal], y.iloc[i_cal])], eval_metric="binary_logloss", callbacks=callbacks)
    # Probability calibrator on CAL
    final_calib, diag = _calibrate_best_brier(base, X.iloc[i_cal], y.iloc[i_cal])
    # Train-only imputation medians (on TRAIN slice only)
    impute_stats = compute_impute_stats(X.iloc[i_train])
    save_schema(FEATURES_SCHEMA_PATH, feats, impute_stats)
    # EV mapper trained on CALIBRATED probabilities (consistency)
    prob_cal_calibrated = final_calib.predict_proba(X.iloc[i_cal])[:, 1]
    realized_adj_cal = pl.loc[mask].iloc[i_cal]["ret_5d_adj"].astype(float).values
    iso_ev = fit_isotonic_ev_mapper(prob_cal_calibrated, realized_adj_cal)
    # TEST inference
    p_test = final_calib.predict_proba(X.iloc[i_test])[:, 1]
    ev_test = iso_ev.predict(p_test)
    oos_df = pd.DataFrame({
        "prob_top20_5d": p_test,
        "ret_3d_close_pct": pl.loc[mask].iloc[i_test]["ret_3d_close_pct"].values,
        "ret_5d_close_pct": pl.loc[mask].iloc[i_test]["ret_5d_close_pct"].values,
        "ret_5d_adj": pl.loc[mask].iloc[i_test]["ret_5d_adj"].values,
        "rank_5d_pct": pl.loc[mask].iloc[i_test]["rank_5d_pct"].values,
        "expected_ret_5d_adj_iso": ev_test,
    })
    return final_calib, iso_ev, oos_df
# ===================== 1D follow-through (mandatory; final refit on full history) =====================
def train_1d_followthrough(panel: pd.DataFrame, feats: List[str], margin_pct: float = CLS_MARGIN_1D,
                           n_splits: int = FOLDS, embargo_days: int = EMBARGO_DAYS):
    lgb, LGBMClassifier = _check_lightgbm()
    r = pd.to_numeric(panel["ret_1d_close_pct"], errors="coerce")
    y = pd.Series(np.nan, index=panel.index)
    y[(r > margin_pct)] = 1
    y[(r < -margin_pct)] = 0
    mask = y.notna()
    n_labels = int(mask.sum())
    if n_labels < MIN_GATE_SAMPLES:
        raise SystemExit(
            f"[FATAL] 1D gate is mandatory, but found only {n_labels} labeled rows (< {MIN_GATE_SAMPLES}). "
            f"Increase history, relax CLS_MARGIN_1D={margin_pct}, or lower MIN_GATE_SAMPLES."
        )
    # CV diagnostics (optional)
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    eta = ProgressETA(total=len(folds), label="Train 1D-Gate (CV diag)")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0:
            eta.tick(f"fold {fold_no} (skip)"); continue
        tr_core_idx, val_idx = split_train_val_by_time(panel, valid_idx[tr_pos], val_frac=0.2, min_val=200)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos = np.where(np.isin(valid_idx, val_idx))[0]
        X_tr = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos] if len(val_pos)>0 else y_tr
        from lightgbm import LGBMClassifier as _Gate
        clf = _Gate(**dict(
            n_estimators=int(N_EST_1D), learning_rate=LEARNING_RATE,
            num_leaves=56, max_depth=MAX_DEPTH, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
            reg_alpha=0.4, reg_lambda=8.0, class_weight=None,
            n_jobs=-1, random_state=int(GLOBAL_SEED + 500 + fold_no), verbosity=-1,
        ))
        callbacks = _lgb_callbacks(len(X_val))
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)
        eta.tick(f"fold {fold_no}")
    # FINAL refit on full labeled history with time-ordered Train/Cal split
    ts_all = pd.to_datetime(panel.loc[mask, "timestamp"]).values
    order = np.argsort(ts_all)
    cut = max(int(0.8 * len(order)), 1)
    tr_pos_full = order[:cut]
    cal_pos_full = order[cut:]
    from lightgbm import LGBMClassifier as _Gate
    final_clf = _Gate(**dict(
        n_estimators=int(N_EST_1D), learning_rate=LEARNING_RATE,
        num_leaves=56, max_depth=MAX_DEPTH, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
        reg_alpha=0.4, reg_lambda=8.0, class_weight=None,
        n_jobs=-1, random_state=int(GLOBAL_SEED + 999), verbosity=-1,
    ))
    final_clf.fit(X_full.iloc[tr_pos_full], y_full[tr_pos_full])
    from sklearn.calibration import CalibratedClassifierCV
    final_gate = CalibratedClassifierCV(estimator=final_clf, method="sigmoid", cv="prefit")
    final_gate.fit(X_full.iloc[cal_pos_full], y_full[cal_pos_full])
    return final_gate
# ===================== Map prob -> expectations (legacy decile mapping) =====================
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
# ===================== Watchlist (gate mandatory; TODAY-BASED inference) =====================
def nightly_watchlist(panel: pd.DataFrame, feats: List[str],
                      m5_cls_calib, calib_table: Dict[str, List[float]],
                      iso_ev_mapper,
                      m1_gate,
                      exclude_pattern: Optional[str],
                      gate_threshold: float,
                      export_xlsx: bool):
    if m1_gate is None:
        raise SystemExit("[FATAL] 1D gate expected but not provided.")
    panel = panel.copy().sort_values(["symbol", "timestamp"])
    panel["avg20_vol"] = panel.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    # TODAY-BASED inference: pick the latest row per symbol regardless of label availability
    last = panel.groupby("symbol", as_index=False).tail(1).copy()
    if exclude_pattern:
        last = last[
            ~last["symbol"].astype(str).str.contains(exclude_pattern, regex=True, na=False)
        ].copy()
    feats_schema, impute_stats = load_schema(FEATURES_SCHEMA_PATH)
    X_raw = sanitize_feature_matrix(last[feats_schema].copy())
    X = reindex_and_impute(X_raw, feats_schema, impute_stats)
    # Calibrated probability of top20
    prob_top20 = m5_cls_calib.predict_proba(X)[:, 1]
    # EV mapping on calibrated prob (consistent)
    expected_adj = iso_ev_mapper.predict(prob_top20)
    # Legacy raw 3d/5d expectations via deciles (diagnostic)
    exp3, exp5, exp_sh = map_prob_to_expectations(prob_top20, calib_table)
    # Mandatory 1D gate
    p1 = m1_gate.predict_proba(X)[:, 1]
    wl = last.copy()
    wl["prob_top20_5d"] = prob_top20
    wl["expected_ret_5d_adj"] = expected_adj
    wl["expected_ret_3d"] = exp3
    wl["expected_ret_5d"] = exp5
    wl["expected_sharpe_5d"] = exp_sh
    wl["prob_up_1d_gate"] = p1
    wl = wl[ wl["prob_up_1d_gate"] >= float(gate_threshold) ].copy()
    wl = wl[(wl["close"] >= MIN_CLOSE) & (wl["avg20_vol"] >= MIN_AVG20_VOL)].copy()
    wl = wl[["symbol","timestamp","close","avg20_vol",
             "prob_top20_5d","prob_up_1d_gate",
             "expected_ret_5d_adj","expected_ret_5d","expected_ret_3d","expected_sharpe_5d",
             "regime_market_trend","regime_high_vol","regime_dispersion",
             "D_atr14","D_cpr_width_pct","long_score","short_score"
            ]].sort_values(["expected_ret_5d_adj","expected_ret_5d"], ascending=[False, False])

    # Always save with a timestamp suffix to avoid locking conflicts and retain history.
    ts = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y%m%d_%H%M%S")
    base = Path(WATCHLIST_OUT)
    ts_csv = base.with_name(base.stem + f"_{ts}").with_suffix(".csv")
    wl.to_csv(ts_csv, index=False)
    print(f"Saved: {ts_csv} rows={len(wl)}")

    if export_xlsx:
        try:
            ts_xlsx = base.with_name(base.stem + f"_{ts}").with_suffix(".xlsx")
            wl.to_excel(ts_xlsx, index=False, engine="openpyxl")
            print(f"Saved: {ts_xlsx}")
        except Exception as e:
            print(f"Excel export failed: {e}")

    return wl
# ===================== Quick Portfolio (optional smoke) =====================
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
# ===================== PUBLIC ENTRY POINT =====================
def run_pipeline(*,
    data_dir: str,
    out_dir: str,
    symbols_like: Optional[str]=None,
    limit_files: Optional[int]=None,
    accept_any_daily: bool=False,
    load_workers: int=8,
    cv_splits: int=FOLDS,
    embargo_days: int=EMBARGO_DAYS,
    early_stopping_rounds: int=EARLY_STOPPING_ROUNDS,
    # Clean, single-line, case-insensitive alternation. Matches these at the END of the symbol.
    exclude_pattern: Optional[str] = (
        r"(?i)(?:LIQUIDPLUS|GROWWLIQID|GLOBUSSPR|MAHKTECH|LIQUID1|"
        r"GROWWNIFTY|LIQUID|GOLD|BEES|IETF|ETF|CASE|ADD)$"
    ),
    gate_threshold: float=0.55,
    export_xlsx: bool=False,
    quick_portfolio: bool=False,
    portfolio_topk: int=20,
    ev_target: str = "cc" # 'cc' or 'oc'
) -> Dict[str, object]:
    """
    End-to-end pipeline with NO CLI. Import and call from PyCharm / your script.
    Returns a dict with references to panel, features, models, watchlist, and report paths.
    """
    setup_paths(out_dir)
    # 0) Collect panel (keeps ALL rows, labels may be NaN near the end)
    paths = _strict_file_list(data_dir, symbols_like, limit_files, accept_any_daily=accept_any_daily)
    panel, feats = collect_panel_from_paths(paths, load_workers=load_workers)
    print(f"[Panel] final rows={len(panel)} cols={len(panel.columns)} feats={len(feats)}")
    # 1) Early excludes for MODELING (training/calibration); inference will re-apply
    if exclude_pattern:
        panel_train = panel[
            ~panel["symbol"].astype(str).str.contains(exclude_pattern, regex=True, na=False)
        ].copy()
    else:
        panel_train = panel.copy()
    # 2) CV diagnostics for the 5d classifier
    _oos_prob_cv, _valid_idx_cv, _pl_cv = train_5d_quantile_cls(panel_train, feats, ev_target,
        n_splits=cv_splits, embargo_days=embargo_days, early_stopping_rounds=early_stopping_rounds)
    # 3) Final model + OOS-only calibrations (probability & EV) + TRAIN-only imputation schema
    final_calib, iso_ev, oos_df = fit_final_model_and_oos_calibration(panel_train, feats, ev_target,
        early_stopping_rounds=early_stopping_rounds)
    # 4) Build a simple decile table from TEST
    df_oos = oos_df.dropna(subset=["prob_top20_5d"]).copy()
    df_oos["prob_bucket"] = pd.qcut(df_oos["prob_top20_5d"], q=min(10, max(3, df_oos.shape[0]//50)), labels=False, duplicates="drop")
    calib = (df_oos.groupby("prob_bucket")
        .agg(avg_ret_3d=("ret_3d_close_pct","mean"),
             avg_ret_5d=("ret_5d_close_pct","mean"),
             std_ret_5d=("ret_5d_close_pct","std"),
             avg_ret_5d_adj=("ret_5d_adj","mean"))
        .reset_index().sort_values("prob_bucket"))
    # spread representative prob grid
    probs = np.linspace(0.05, 0.95, len(calib)) if len(calib) >= 3 else np.array([0.05, 0.5, 0.95])
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
    # 5) OOS report
    from sklearn.metrics import brier_score_loss
    from scipy.stats import spearmanr
    pseudo_y = np.where(df_oos["prob_top20_5d"].notna(), df_oos["prob_top20_5d"], np.nan)
    # Brier vs pseudo top20 label from rank was already used upstream; keep same as v3.1
    # Recompute quickly here to fill the report consistently
    # Using rank bin proxy to derive a {0,1} target for brier estimation
    # (only for TEST diagnostics)
    # Build proxy target
    target_proxy = np.where(df_oos["rank_5d_pct"] >= 0.8, 1, np.where(df_oos["rank_5d_pct"] <= 0.2, 0, np.nan))
    msk = ~np.isnan(target_proxy)
    brier = float(brier_score_loss(pd.Series(target_proxy)[msk], df_oos.loc[msk, "prob_top20_5d"]))
    ic5, ic5_p = spearmanr(df_oos["prob_top20_5d"], df_oos["ret_5d_adj"], nan_policy="omit")
    rep = {"5d_cls": {"n_oos": int(len(df_oos)), "brier_pseudo": brier,
                      "spearman_ic_5d_adj": float(ic5), "spearman_ic_pvalue": float(ic5_p),
                      "ev_target": ev_target, "embargo_days": int(embargo_days)}}
    Path(OOS_REPORT_PATH).write_text(json.dumps(rep, indent=2))
    print(f"[OOS] Saved report: {OOS_REPORT_PATH}")
    print(f"[Calibration] Saved table: {CALIB_TABLE_PATH}")
    # 6) Train mandatory 1D gate on training panel
    m1_gate = train_1d_followthrough(panel_train, feats, margin_pct=CLS_MARGIN_1D,
        n_splits=cv_splits, embargo_days=embargo_days)
    # 7) Nightly watchlist (TODAY-BASED inference)
    wl = nightly_watchlist(panel, feats, final_calib, calib_table, iso_ev, m1_gate,
        exclude_pattern=exclude_pattern,
        gate_threshold=float(gate_threshold),
        export_xlsx=export_xlsx)
    # 8) Optional quick portfolio sanity-check
    eq_path = None
    if quick_portfolio:
        eq = quick_portfolio_backtest(panel, wl, horizon=5, top_k=int(portfolio_topk))
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.plot(eq.values)
            plt.title("Quick Portfolio (Top-K by expected_ret_5d_adj)")
            plt.xlabel("Days"); plt.ylabel("Equity")
            out_png = Path(Path(WATCHLIST_OUT).parent) / "quick_portfolio_equity.png"
            plt.tight_layout(); plt.savefig(out_png, dpi=120)
            print(f"[QuickPF] Saved: {out_png}")
            eq_path = str(out_png)
        except Exception as e:
            print(f"[QuickPF] Plot failed: {e}")
    return {
        "panel": panel,
        "features": feats,
        "final_calibrator": final_calib,
        "iso_ev_mapper": iso_ev,
        "gate_model": m1_gate,
        "watchlist": wl,
        "oos_report_path": OOS_REPORT_PATH,
        "calibration_table_path": CALIB_TABLE_PATH,
        "panel_path": PANEL_OUT,
        "watchlist_path": WATCHLIST_OUT,
        "equity_plot_path": eq_path,
    }
if __name__ == "__main__":
    print(
        """This module exposes run_pipeline(...). Import and call it from your script.
Example:
from cpr_fix import run_pipeline
res = run_pipeline(
    data_dir=r"C:\\path\\to\\cache",
    out_dir=r"C:\\path\\to\\out",
    ev_target="cc"
)
"""
    )
