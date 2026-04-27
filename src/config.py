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
