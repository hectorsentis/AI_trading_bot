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
    "hour_sin",
    "hour_cos",
]

FEATURE_VERSION = "v2_ta_patterns"
LABEL_VERSION = "triple_barrier_v1"

LOOKAHEAD_BARS = 6
TP_MULTIPLIER = 1.5
SL_MULTIPLIER = 1.0

# Feature store incremental:
# recalculate only the latest overlap window instead of the full history each run.
FEATURE_STORE_RECALC_OVERLAP_BARS = 120
FEATURE_STORE_WARMUP_BARS = 120

MODEL_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "random_state": 42,
}

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

MODEL_LIFECYCLE_STATUSES = [
    "candidate", "validation_rejected", "validation_accepted", "backtest_rejected",
    "backtest_accepted", "paper_active", "paper_rejected", "paper_validated",
    "real_ready", "real_active", "real_paused", "real_rejected", "archived",
]

ACCOUNT_MODE_LOCAL_PAPER = "local_paper"
ACCOUNT_MODE_TESTNET_PAPER = "testnet_paper"
ACCOUNT_MODE_SHADOW_REAL = "shadow_real"
ACCOUNT_MODE_REAL = "real"
ACCOUNT_MODES = [ACCOUNT_MODE_LOCAL_PAPER, ACCOUNT_MODE_TESTNET_PAPER, ACCOUNT_MODE_SHADOW_REAL, ACCOUNT_MODE_REAL]

DRY_RUN = _env_bool("DRY_RUN", True)
ENABLE_TESTNET_PAPER_TRADING = _env_bool("ENABLE_TESTNET_PAPER_TRADING", True)
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
    BINANCE_TESTNET_BASE_URL = BINANCE_REST_BASE_URL
BINANCE_REAL_API_KEY = _env_str("BINANCE_REAL_API_KEY", "") or None
BINANCE_REAL_API_SECRET = _env_str("BINANCE_REAL_API_SECRET", "") or None
BINANCE_REAL_BASE_URL = _env_str("BINANCE_REAL_BASE_URL", "https://api.binance.com")
BINANCE_ACCOUNT_REST_BASE_URL = BINANCE_TESTNET_BASE_URL if BINANCE_USE_TESTNET else BINANCE_REAL_BASE_URL
BINANCE_EXECUTION_REST_BASE_URL = BINANCE_ACCOUNT_REST_BASE_URL

SYMBOLS = _env_list("SYMBOLS", SYMBOLS)
TIMEFRAME = _env_str("TIMEFRAME", TIMEFRAME) or TIMEFRAME
TARGET_ACCEPTED_MODELS = _env_int("TARGET_ACCEPTED_MODELS", 5)
MAX_TRAINING_ATTEMPTS_PER_CYCLE = _env_int("MAX_TRAINING_ATTEMPTS_PER_CYCLE", _env_int("MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE", 50))
MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE = MAX_TRAINING_ATTEMPTS_PER_CYCLE
AUTO_REPLACE_REJECTED_MODELS = _env_bool("AUTO_REPLACE_REJECTED_MODELS", True)
TRAINING_CUTOFF_HOURS_BEFORE_NOW = _env_int("TRAINING_CUTOFF_HOURS_BEFORE_NOW", 168)
VALIDATION_WINDOW_HOURS = _env_int("VALIDATION_WINDOW_HOURS", 168)
WALK_FORWARD_ENABLED = _env_bool("WALK_FORWARD_ENABLED", True)

MIN_PAPER_VALIDATION_DAYS = _env_int("MIN_PAPER_VALIDATION_DAYS", 7)
MIN_PAPER_VALIDATION_TRADES = _env_int("MIN_PAPER_VALIDATION_TRADES", 20)
PAPER_MIN_PROFIT_FACTOR = _env_float("PAPER_MIN_PROFIT_FACTOR", 1.05)
PAPER_MAX_DRAWDOWN = _env_float("PAPER_MAX_DRAWDOWN", 0.08)
PAPER_MIN_TOTAL_RETURN = _env_float("PAPER_MIN_TOTAL_RETURN", 0.0)
PAPER_MIN_WIN_RATE = _env_float("PAPER_MIN_WIN_RATE", 0.45)

MAX_EXPOSURE_PER_MODEL_USDT = _env_float("MAX_EXPOSURE_PER_MODEL_USDT", 100.0)
MAX_EXPOSURE_TOTAL_USDT = _env_float("MAX_EXPOSURE_TOTAL_USDT", 500.0)
MAX_POSITION_PCT_PER_SYMBOL = _env_float("MAX_POSITION_PCT_PER_SYMBOL", 0.20)
MAX_DAILY_LOSS_USDT = _env_float("MAX_DAILY_LOSS_USDT", 50.0)
MAX_TRADES_PER_DAY_PER_MODEL = _env_int("MAX_TRADES_PER_DAY_PER_MODEL", 10)
MAX_ORDER_NOTIONAL_USDT = _env_float("MAX_ORDER_NOTIONAL_USDT", 50.0)
MIN_ORDER_NOTIONAL_USDT = _env_float("MIN_ORDER_NOTIONAL_USDT", 10.0)
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
