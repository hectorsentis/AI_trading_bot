import os
from pathlib import Path


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv_file() -> None:
    """Load simple KEY=VALUE pairs from repo .env without overriding the real environment."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}


_load_dotenv_file()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_DIR = DATA_DIR / "db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

DB_FILE = DB_DIR / "market_data.sqlite"
QUALITY_LOGS_DIR = LOGS_DIR / "data_quality"


# =========================================================
# TABLAS DE BASE DE DATOS
# =========================================================
PRICES_TABLE = "prices"
FEATURES_TABLE = "features"
SIGNALS_TABLE = "signals"
ORDERS_TABLE = "orders"
POSITIONS_TABLE = "positions"
PORTFOLIO_SNAPSHOTS_TABLE = "portfolio_snapshots"
VALIDATION_PREDICTIONS_TABLE = "validation_predictions"
INGESTION_LOG_TABLE = "ingestion_log"
DATA_GAPS_TABLE = "data_gaps"
DATA_COVERAGE_TABLE = "data_coverage"
MODEL_REGISTRY_TABLE = "model_registry"

PRICE_PRIMARY_KEY = ["symbol", "timeframe", "datetime_utc"]


# =========================================================
# PROVEEDOR DE DATOS
# =========================================================
DATA_PROVIDER = "binance"

BINANCE_REST_BASE_URL = "https://api.binance.com"
BINANCE_WS_BASE_URL = "wss://data-stream.binance.vision/ws"

BINANCE_TESTNET_REST_BASE_URL = "https://testnet.binance.vision"
BINANCE_TESTNET_WS_BASE_URL = "wss://stream.testnet.binance.vision/ws"
BINANCE_USE_TESTNET = _env_bool("BINANCE_USE_TESTNET", _env_bool("BINANCE_DEMO_TRADING", False))
BINANCE_ACCOUNT_REST_BASE_URL = BINANCE_TESTNET_REST_BASE_URL if BINANCE_USE_TESTNET else BINANCE_REST_BASE_URL
BINANCE_EXECUTION_REST_BASE_URL = BINANCE_TESTNET_REST_BASE_URL if BINANCE_USE_TESTNET else BINANCE_REST_BASE_URL


# =========================================================
# MODO DE TRABAJO
# =========================================================
MARKET_TYPE = "spot"
ENVIRONMENT = "prod"


# =========================================================
# SÍMBOLOS E INTERVALO
# =========================================================
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    # "BNBUSDT",
    # "XRPUSDT",
]

TIMEFRAME = "1h"
SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h"]


# =========================================================
# DESCARGA HISTÓRICA / API
# =========================================================
KLINES_LIMIT = 1000
INITIAL_BACKFILL_DAYS = 365
FULL_BACKFILL_START_DATE = "2017-01-01"
OVERLAP_BARS = 5
API_SLEEP_SECONDS = 0.25
HTTP_TIMEOUT_SECONDS = 30


# =========================================================
# RAW SNAPSHOTS
# =========================================================
RAW_FILE_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
SAVE_RAW_AS_GZIP = False


# =========================================================
# COLUMNAS ESPERADAS EN prices
# =========================================================
PRICE_COLUMNS = [
    "symbol",
    "timeframe",
    "datetime_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_utc",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "provider",
    "ingestion_ts_utc",
]


# =========================================================
# DATA QUALITY
# =========================================================
ENABLE_DATA_GAP_CHECK = True


# =========================================================
# OPCIONES DE DESCARGA DESDE TERMINAL
# =========================================================
DEFAULT_DOWNLOAD_MODE = "incremental"
DEFAULT_RECENT_BARS = 500
DEFAULT_START_DATE = None
DEFAULT_END_DATE = None


# =========================================================
# LIMPIEZA / VALIDACIÓN
# =========================================================
DROP_ZERO_VOLUME_BARS = False
DROP_DUPLICATE_BARS = True
SORT_BEFORE_INSERT = True


# =========================================================
# FUTURO: TIEMPO REAL
# =========================================================
ENABLE_REALTIME_INGESTION = False
REALTIME_STREAM_TYPE = "kline"


# =========================================================
# FUTURO: TRADING
# =========================================================
ENABLE_TRADING = False
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or None
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or None
BINANCE_RECV_WINDOW_MS = int(os.getenv("BINANCE_RECV_WINDOW_MS", "5000"))

TRADE_MODE = "spot"
DEFAULT_ORDER_TYPE = "MARKET"
DEFAULT_QUOTE_SIZE_USDT = 50.0
DRY_RUN = True


# =========================================================
# FEATURES / MODELO
# =========================================================
FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "hl_range",
    "oc_range",
    "atr_14",
    "volatility_10",
    "volatility_20",
    "vol_ratio_10",
    "vol_zscore_20",
    "dist_ma_5",
    "dist_ma_10",
    "dist_ma_20",
    "slope_ma_5",
    "slope_ma_10",
    "rolling_max_dist_20",
    "rolling_min_dist_20",
    "rsi_14",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "is_doji",
    "is_hammer",
    "is_shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
    "inside_bar",
    "outside_bar",
    "breakout_20",
    "breakdown_20",
    "ma_cross_5_20",
    "double_top_proxy",
    "double_bottom_proxy",
    "atr_pct",
    "ema_12_dist",
    "ema_26_dist",
    "ema_12_26_spread",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width_20",
    "bb_percent_b_20",
    "stoch_k_14",
    "stoch_d_3",
    "ret_zscore_20",
    "volatility_ratio_10_20",
    "volume_change_1",
    "volume_trend_5",
    "range_zscore_20",
    "trend_strength_20",
    "consecutive_up_3",
    "consecutive_down_3",
    "hour_sin",
    "hour_cos",
    # --- Phase C (v4): volatility / regime / momentum ---
    "ret_24",
    "roc_10",
    "rsi_7",
    "volatility_50",
    "volatility_ratio_20_50",
    "downside_volatility_20",
    "volatility_regime_score",
    "dist_ma_50",
    "dist_ma_200",
    "price_above_sma_50",
    "trend_strength_50",
    "rolling_drawdown_50",
    "dist_from_high_50",
    # --- Phase C (v4): microstructure from taker data ---
    "taker_buy_ratio",
    "taker_imbalance",
    "taker_imbalance_zscore_20",
    "avg_trade_size_zscore_20",
    # --- Phase C (v4): cross-asset BTC context (neutral when context absent) ---
    "btc_ret_24",
    "rel_strength_vs_btc_24",
    "corr_btc_50",
    "beta_btc_50",
    # --- Phase C (v5): multi-timeframe context (two higher TFs, closed candles only) ---
    "htf1_rsi_14",
    "htf1_trend_strength",
    "htf1_volatility",
    "htf2_rsi_14",
    "htf2_trend_strength",
    "htf2_volatility",
]

FEATURE_VERSION = "v5_multitimeframe"
LABEL_VERSION = "triple_barrier_tp_sl_v2"

# Phase C: cross-asset context + optional external data (funding/OI/fear-greed).
CROSS_ASSET_REFERENCE_SYMBOL = os.getenv("CROSS_ASSET_REFERENCE_SYMBOL", "BTCUSDT")
ENABLE_CROSS_ASSET_FEATURES = _env_bool("ENABLE_CROSS_ASSET_FEATURES", True)

# Phase C: multi-timeframe features. For each base timeframe, attach context from up to two
# strictly-higher timeframes (htf1, htf2) using only CLOSED higher-TF candles (leakage-safe).
ENABLE_MULTI_TIMEFRAME_FEATURES = _env_bool("ENABLE_MULTI_TIMEFRAME_FEATURES", True)
HIGHER_TIMEFRAME_MAP = {
    "15m": ["1h", "4h"],
    "1h": ["4h", "1d"],
    "4h": ["1d", "1w"],
    "1d": ["1w", "1M"],
}
EXTERNAL_DATA_TABLE = "external_data"
ENABLE_EXTERNAL_DATA = _env_bool("ENABLE_EXTERNAL_DATA", False)  # opt-in; requires network
FEAR_GREED_API_URL = os.getenv("FEAR_GREED_API_URL", "https://api.alternative.me/fng/")
BINANCE_FUTURES_BASE_URL = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")

LOOKAHEAD_BARS = 6
TP_MULTIPLIER = 1.5
SL_MULTIPLIER = 1.0
REQUIRE_TP_SL_ON_ENTRY = True
MIN_TP_SL_RISK_REWARD = 1.0

# Feature store incremental:
# recalculate only the latest overlap window instead of the full history each run.
FEATURE_STORE_RECALC_OVERLAP_BARS = 120
FEATURE_STORE_WARMUP_BARS = 120

MODEL_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.25,
    "class_weight": "balanced",
    "random_state": 42,
    "verbosity": -1,
}

# Native return-distribution models (Phase B). These are trained alongside the direction
# classifier and produce real expected-return / quantile / MFE / MAE estimates, replacing the
# synthetic fields derived from classifier probabilities. Disable to fall back to the derived path.
ENABLE_NATIVE_PREDICTION_MODELS = _env_bool("ENABLE_NATIVE_PREDICTION_MODELS", True)
NATIVE_PREDICTION_HORIZON_BARS = int(os.getenv("NATIVE_PREDICTION_HORIZON_BARS", str(LOOKAHEAD_BARS)))
NATIVE_PREDICTION_MIN_ROWS = int(os.getenv("NATIVE_PREDICTION_MIN_ROWS", "500"))
NATIVE_QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Lighter than the classifier params: 7 regressors are trained per candidate, so keep them cheap.
NATIVE_MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.25,
    "random_state": 42,
    "verbosity": -1,
}

# Probability calibration (Phase B). Calibrate the multiclass classifier so reported
# confidence/probabilities are trustworthy (they gate signals, proposals and allocation).
# Stored in the artifact as `calibrator` and used by the live prediction path; disable for
# faster pool maintenance. Falls back to raw model probabilities when absent.
ENABLE_PROBABILITY_CALIBRATION = _env_bool("ENABLE_PROBABILITY_CALIBRATION", True)
PROBABILITY_CALIBRATION_METHOD = os.getenv("PROBABILITY_CALIBRATION_METHOD", "isotonic")
PROBABILITY_CALIBRATION_CV = int(os.getenv("PROBABILITY_CALIBRATION_CV", "3"))
PROBABILITY_CALIBRATION_MIN_ROWS = int(os.getenv("PROBABILITY_CALIBRATION_MIN_ROWS", "500"))

LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.55

TRAIN_SIZE = 250
TEST_SIZE = 50
RETRAIN_STEP = 50
COST_PER_TRADE = 0.0005

MIN_TRAIN_ROWS = 1000


# =========================================================
# GATING / SELECCION DE MODELOS
# =========================================================
MIN_ACCEPTABLE_SHARPE = 0.20
MIN_ACCEPTABLE_PROFIT_FACTOR = 1.05
MAX_ACCEPTABLE_DRAWDOWN = 0.20
MIN_ACCEPTABLE_TRADES = 10
MIN_ACCEPTABLE_F1_MACRO = 0.34
MIN_ACCEPTABLE_ACCURACY = 0.34
MIN_ACCEPTABLE_STRATEGY_RETURN = 0.00
REQUIRE_OUTPERFORM_BASELINE = True
MAX_TRAIN_VALIDATION_DRIFT = 0.20
REQUIRE_OOS_FOR_ACCEPTANCE = True

MODEL_SELECTION_ACCEPTANCE_ORDER = ["accepted", "candidate"]
PREFER_ACTIVE_MODEL = True


# =========================================================
# MODEL POOL MAINTENANCE
# =========================================================
TARGET_ACCEPTED_MODELS = int(os.getenv("TARGET_ACCEPTED_MODELS", "3"))
ENABLE_MODEL_POOL_MAINTENANCE = _env_bool("ENABLE_MODEL_POOL_MAINTENANCE", True)
MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE = int(os.getenv("MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE", "4"))
MODEL_POOL_VALIDATION_MAX_FOLDS = int(os.getenv("MODEL_POOL_VALIDATION_MAX_FOLDS", "5"))
MODEL_POOL_TRAINING_ENABLED_IN_BOT = _env_bool("MODEL_POOL_TRAINING_ENABLED_IN_BOT", True)
MODEL_POOL_MAINTENANCE_INTERVAL_SECONDS = int(os.getenv("MODEL_POOL_MAINTENANCE_INTERVAL_SECONDS", "3600"))


# =========================================================
# SIGNAL ENGINE
# =========================================================
SIGNAL_MIN_CONFIDENCE = 0.55
SIGNAL_MIN_MARGIN = 0.08


# =========================================================
# PAPER TRADING (DRY RUN)
# =========================================================
PAPER_INITIAL_CASH_USDT = 10_000.0
PAPER_FEE_RATE = 0.0005
PAPER_SLIPPAGE_BPS = 2.0
PAPER_POSITION_STEP_SIZE = 0.0001
PAPER_MIN_NOTIONAL_USDT = 10.0
PAPER_MAX_EXPOSURE_PER_ASSET = 0.35
PAPER_MAX_POSITION_NOTIONAL_USDT = 3_000.0
PAPER_MAX_NEW_TRADES_PER_DAY = 20
PAPER_MAX_DAILY_LOSS_USDT = 500.0

# =========================================================
# AUTONOMOUS PLATFORM CONFIG (safe-by-default)
# =========================================================

def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else float(default)
    except ValueError:
        return float(default)


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


DB_FILE = Path(_env_str("SQLITE_DB_PATH", str(DB_FILE))).expanduser()
if not DB_FILE.is_absolute():
    DB_FILE = (BASE_DIR / DB_FILE).resolve()
DB_DIR = DB_FILE.parent

FILLS_TABLE = "fills"
PAPER_MODEL_METRICS_TABLE = "paper_model_metrics"
MODEL_LIFECYCLE_EVENTS_TABLE = "model_lifecycle_events"
BOT_EVENTS_TABLE = "bot_events"
RISK_EVENTS_TABLE = "risk_events"
BOT_STATUS_TABLE = "bot_status"
LABELS_TABLE = "labels"
MODEL_PREDICTIONS_TABLE = "model_predictions"
MODEL_PERFORMANCE_TABLE = "model_performance"
TRADE_PROPOSALS_TABLE = "trade_proposals"
ALLOCATIONS_TABLE = "allocations"
TRADES_TABLE = "trades"
ACCOUNT_SNAPSHOTS_TABLE = "account_snapshots"
BALANCE_SNAPSHOTS_TABLE = "balance_snapshots"
SHADOW_TRADES_TABLE = "shadow_trades"
SHADOW_TRADE_EVENTS_TABLE = "shadow_trade_events"
RECONCILIATION_EVENTS_TABLE = "reconciliation_events"
SYSTEM_STATUS_TABLE = "system_status"

# Phase D: paper degradation / quarantine thresholds (softer than the hard reject gates).
# A model that drifts below these is paused (paper_degraded) and can recover; severe breaches
# quarantine it. These are intentionally tighter than PAPER_MAX_DRAWDOWN / PAPER_MIN_PROFIT_FACTOR.
PAPER_DEGRADE_MIN_TRADES = int(os.getenv("PAPER_DEGRADE_MIN_TRADES", "5"))
PAPER_DEGRADE_DRAWDOWN = float(os.getenv("PAPER_DEGRADE_DRAWDOWN", "0.06"))
PAPER_DEGRADE_PROFIT_FACTOR = float(os.getenv("PAPER_DEGRADE_PROFIT_FACTOR", "1.0"))
PAPER_DEGRADE_MIN_RETURN = float(os.getenv("PAPER_DEGRADE_MIN_RETURN", "-0.02"))
PAPER_QUARANTINE_DRAWDOWN = float(os.getenv("PAPER_QUARANTINE_DRAWDOWN", "0.10"))
PAPER_QUARANTINE_MIN_RETURN = float(os.getenv("PAPER_QUARANTINE_MIN_RETURN", "-0.05"))

MODEL_LIFECYCLE_STATUSES = [
    "candidate", "validation_rejected", "validation_accepted", "backtest_rejected",
    "backtest_accepted", "paper_active", "paper_rejected", "paper_validated",
    "real_ready", "real_active", "real_paused", "real_rejected", "paper_degraded",
    "quarantined", "archived",
]

ACCOUNT_MODE_LOCAL_PAPER = "local_paper"
ACCOUNT_MODE_TESTNET_PAPER = "testnet_paper"
ACCOUNT_MODE_SHADOW_REAL = "shadow_real"
ACCOUNT_MODE_REAL = "real"
ACCOUNT_MODES = [ACCOUNT_MODE_LOCAL_PAPER, ACCOUNT_MODE_TESTNET_PAPER, ACCOUNT_MODE_SHADOW_REAL, ACCOUNT_MODE_REAL]

DRY_RUN = _env_bool("DRY_RUN", True)
ENABLE_TESTNET_PAPER_TRADING = _env_bool("ENABLE_TESTNET_PAPER_TRADING", True)
ENABLE_BINANCE_TESTNET_PAPER_TRADING = _env_bool(
    "ENABLE_BINANCE_TESTNET_PAPER_TRADING",
    ENABLE_TESTNET_PAPER_TRADING,
)
ENABLE_TESTNET_PAPER_TRADING = ENABLE_BINANCE_TESTNET_PAPER_TRADING
ENABLE_LOCAL_SIMULATED_PAPER = _env_bool("ENABLE_LOCAL_SIMULATED_PAPER", True)
ENABLE_LIVE_TRADING = _env_bool("ENABLE_LIVE_TRADING", False)
ENABLE_REAL_ORDER_EXECUTION = _env_bool("ENABLE_REAL_ORDER_EXECUTION", False)
ENABLE_REAL_BINANCE_ACCOUNT = _env_bool("ENABLE_REAL_BINANCE_ACCOUNT", False)
ALLOW_AUTO_PROMOTE_TO_REAL = _env_bool("ALLOW_AUTO_PROMOTE_TO_REAL", False)
ENABLE_TRADING = ENABLE_LIVE_TRADING and ENABLE_REAL_ORDER_EXECUTION and ENABLE_REAL_BINANCE_ACCOUNT and not DRY_RUN

BINANCE_ENV = (_env_str("BINANCE_ENV", "prod") or "prod").lower()
BINANCE_PUBLIC_BASE_URL = _env_str("BINANCE_PUBLIC_BASE_URL", "https://api.binance.com")
BINANCE_REST_BASE_URL = _env_str("BINANCE_REST_BASE_URL", BINANCE_PUBLIC_BASE_URL)
BINANCE_WS_API_URL = _env_str("BINANCE_WS_API_URL", "")
BINANCE_WS_STREAM_URL = _env_str("BINANCE_WS_STREAM_URL", "")
BINANCE_WS_COMBINED_STREAM_URL = _env_str("BINANCE_WS_COMBINED_STREAM_URL", "")

# Binance Spot Demo Mode aliases are accepted because Binance labels this
# environment differently from the older "testnet" wording.
BINANCE_DEMO_API_KEY = _env_str("BINANCE_DEMO_API_KEY", "") or None
BINANCE_DEMO_API_SECRET = _env_str("BINANCE_DEMO_API_SECRET", "") or None
BINANCE_TESTNET_API_KEY = _env_str("BINANCE_TESTNET_API_KEY", BINANCE_DEMO_API_KEY or _env_str("BINANCE_API_KEY", "")) or None
BINANCE_TESTNET_API_SECRET = _env_str("BINANCE_TESTNET_API_SECRET", BINANCE_DEMO_API_SECRET or _env_str("BINANCE_API_SECRET", "")) or None
BINANCE_TESTNET_BASE_URL = _env_str(
    "BINANCE_TESTNET_BASE_URL",
    _env_str("BINANCE_REST_BASE_URL", "https://testnet.binance.vision"),
)
if BINANCE_ENV in {"demo", "demo_mode", "spot_demo", "spot_demo_mode"}:
    BINANCE_USE_TESTNET = True
    BINANCE_TESTNET_BASE_URL = BINANCE_REST_BASE_URL
BINANCE_REAL_API_KEY = _env_str("BINANCE_REAL_API_KEY", "") or None
BINANCE_REAL_API_SECRET = _env_str("BINANCE_REAL_API_SECRET", "") or None
BINANCE_REAL_BASE_URL = _env_str("BINANCE_REAL_BASE_URL", "https://api.binance.com")
BINANCE_ACCOUNT_REST_BASE_URL = BINANCE_TESTNET_BASE_URL if BINANCE_USE_TESTNET else BINANCE_REAL_BASE_URL
BINANCE_EXECUTION_REST_BASE_URL = BINANCE_ACCOUNT_REST_BASE_URL

SYMBOLS = _env_list("SYMBOLS", SYMBOLS)
TIMEFRAME = _env_str("TIMEFRAME", TIMEFRAME) or TIMEFRAME
TIMEFRAMES = _env_list("TIMEFRAMES", [TIMEFRAME])
if not TIMEFRAMES:
    TIMEFRAMES = [TIMEFRAME]
if TIMEFRAME not in TIMEFRAMES:
    TIMEFRAMES = [TIMEFRAME] + [tf for tf in TIMEFRAMES if tf != TIMEFRAME]
TRAINING_SCOPE = (_env_str("TRAINING_SCOPE", "per_symbol") or "per_symbol").replace("-", "_")
if TRAINING_SCOPE not in {"multi_symbol", "per_symbol", "both"}:
    TRAINING_SCOPE = "both"
TARGET_ACCEPTED_MODELS = _env_int("TARGET_ACCEPTED_MODELS", 5)
MAX_TRAINING_ATTEMPTS_PER_CYCLE = _env_int("MAX_TRAINING_ATTEMPTS_PER_CYCLE", _env_int("MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE", 50))
MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE = MAX_TRAINING_ATTEMPTS_PER_CYCLE
AUTO_REPLACE_REJECTED_MODELS = _env_bool("AUTO_REPLACE_REJECTED_MODELS", True)
TRAINING_CUTOFF_HOURS_BEFORE_NOW = _env_int("TRAINING_CUTOFF_HOURS_BEFORE_NOW", 168)
VALIDATION_WINDOW_HOURS = _env_int("VALIDATION_WINDOW_HOURS", 168)
WALK_FORWARD_ENABLED = _env_bool("WALK_FORWARD_ENABLED", True)
LOOKAHEAD_BARS = _env_int("LOOKAHEAD_BARS", LOOKAHEAD_BARS)
TP_MULTIPLIER = _env_float("TP_MULTIPLIER", TP_MULTIPLIER)
SL_MULTIPLIER = _env_float("SL_MULTIPLIER", SL_MULTIPLIER)
REQUIRE_TP_SL_ON_ENTRY = _env_bool("REQUIRE_TP_SL_ON_ENTRY", True)
MIN_TP_SL_RISK_REWARD = _env_float("MIN_TP_SL_RISK_REWARD", 1.0)

MIN_PAPER_VALIDATION_DAYS = _env_int("MIN_PAPER_VALIDATION_DAYS", 7)
MIN_PAPER_VALIDATION_TRADES = _env_int("MIN_PAPER_VALIDATION_TRADES", 20)
PAPER_MIN_PROFIT_FACTOR = _env_float("PAPER_MIN_PROFIT_FACTOR", 1.05)
PAPER_MAX_DRAWDOWN = _env_float("PAPER_MAX_DRAWDOWN", 0.08)
PAPER_MIN_TOTAL_RETURN = _env_float("PAPER_MIN_TOTAL_RETURN", 0.0)
PAPER_MIN_WIN_RATE = _env_float("PAPER_MIN_WIN_RATE", 0.45)

MAX_EXPOSURE_PER_MODEL_USDT = _env_float("MAX_EXPOSURE_PER_MODEL_USDT", 100.0)
MAX_EXPOSURE_TOTAL_USDT = _env_float("MAX_EXPOSURE_TOTAL_USDT", 500.0)
MAX_TOTAL_EXPOSURE_USDT = _env_float("MAX_TOTAL_EXPOSURE_USDT", MAX_EXPOSURE_TOTAL_USDT)
MAX_SYMBOL_EXPOSURE_USDT = _env_float("MAX_SYMBOL_EXPOSURE_USDT", 250.0)
MAX_MODEL_OPEN_EXPOSURE_USDT = _env_float("MAX_MODEL_OPEN_EXPOSURE_USDT", MAX_EXPOSURE_PER_MODEL_USDT)
MAX_OPEN_TRADES_TOTAL = _env_int("MAX_OPEN_TRADES_TOTAL", 10)
MAX_OPEN_TRADES_PER_MODEL = _env_int("MAX_OPEN_TRADES_PER_MODEL", 3)
MAX_OPEN_TRADES_PER_SYMBOL = _env_int("MAX_OPEN_TRADES_PER_SYMBOL", 5)
MAX_POSITION_PCT_PER_SYMBOL = _env_float("MAX_POSITION_PCT_PER_SYMBOL", 0.20)
MAX_DAILY_LOSS_USDT = _env_float("MAX_DAILY_LOSS_USDT", 50.0)
MAX_TOTAL_DRAWDOWN_PCT = _env_float("MAX_TOTAL_DRAWDOWN_PCT", 0.08)
MAX_TRADE_LOSS_USDT = _env_float("MAX_TRADE_LOSS_USDT", 25.0)
MAX_TRADES_PER_DAY_PER_MODEL = _env_int("MAX_TRADES_PER_DAY_PER_MODEL", 10)
MAX_TRADES_PER_DAY_TOTAL = _env_int("MAX_TRADES_PER_DAY_TOTAL", 50)
MAX_ORDER_NOTIONAL_USDT = _env_float("MAX_ORDER_NOTIONAL_USDT", 50.0)
MIN_ORDER_NOTIONAL_USDT = _env_float("MIN_ORDER_NOTIONAL_USDT", 10.0)
MIN_CASH_RESERVE_USDT = _env_float("MIN_CASH_RESERVE_USDT", 50.0)
STALE_DATA_MAX_SECONDS = _env_int("STALE_DATA_MAX_SECONDS", 120)
RECONCILIATION_REQUIRED = _env_bool("RECONCILIATION_REQUIRED", True)
ENABLE_SHADOW_TRADES_FOR_REJECTED_PROPOSALS = _env_bool("ENABLE_SHADOW_TRADES_FOR_REJECTED_PROPOSALS", True)
TRADE_PROPOSAL_MIN_CONFIDENCE = _env_float("TRADE_PROPOSAL_MIN_CONFIDENCE", 0.52)
TRADE_PROPOSAL_MIN_EXPECTED_RETURN_PCT = _env_float("TRADE_PROPOSAL_MIN_EXPECTED_RETURN_PCT", 0.0005)
ALLOCATOR_MIN_SCORE = _env_float("ALLOCATOR_MIN_SCORE", 0.0)
DEFAULT_SIGNAL_HORIZON_BARS = _env_int("DEFAULT_SIGNAL_HORIZON_BARS", LOOKAHEAD_BARS)
EMERGENCY_STOP_EXTRA_ADVERSE_MULTIPLIER = _env_float("EMERGENCY_STOP_EXTRA_ADVERSE_MULTIPLIER", 1.25)
KILL_SWITCH_ENABLED = _env_bool("KILL_SWITCH_ENABLED", True)

DEFAULT_QUOTE_SIZE_USDT = min(_env_float("DEFAULT_QUOTE_SIZE_USDT", DEFAULT_QUOTE_SIZE_USDT), MAX_ORDER_NOTIONAL_USDT)
PAPER_MIN_NOTIONAL_USDT = MIN_ORDER_NOTIONAL_USDT
PAPER_MAX_POSITION_NOTIONAL_USDT = MAX_EXPOSURE_PER_MODEL_USDT
PAPER_MAX_NEW_TRADES_PER_DAY = MAX_TRADES_PER_DAY_PER_MODEL
PAPER_MAX_DAILY_LOSS_USDT = MAX_DAILY_LOSS_USDT
PAPER_MAX_EXPOSURE_PER_ASSET = MAX_POSITION_PCT_PER_SYMBOL

BOT_POLL_SECONDS = _env_int("BOT_POLL_SECONDS", 60)
MODEL_EVALUATION_INTERVAL_SECONDS = _env_int("MODEL_EVALUATION_INTERVAL_SECONDS", 3600)
MODEL_MAINTENANCE_INTERVAL_SECONDS = _env_int("MODEL_MAINTENANCE_INTERVAL_SECONDS", 3600)
MODEL_POOL_MAINTENANCE_INTERVAL_SECONDS = MODEL_MAINTENANCE_INTERVAL_SECONDS
DASHBOARD_REFRESH_SECONDS = _env_int("DASHBOARD_REFRESH_SECONDS", 30)
