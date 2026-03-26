from pathlib import Path


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

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
INGESTION_LOG_TABLE = "ingestion_log"
DATA_GAPS_TABLE = "data_gaps"

PRICE_PRIMARY_KEY = ["symbol", "timeframe", "datetime_utc"]


# =========================================================
# PROVEEDOR DE DATOS
# =========================================================
DATA_PROVIDER = "binance"

BINANCE_REST_BASE_URL = "https://api.binance.com"
BINANCE_WS_BASE_URL = "wss://data-stream.binance.vision/ws"

BINANCE_TESTNET_REST_BASE_URL = "https://testnet.binance.vision"
BINANCE_TESTNET_WS_BASE_URL = "wss://stream.testnet.binance.vision/ws"


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


# =========================================================
# DESCARGA HISTÓRICA / API
# =========================================================
KLINES_LIMIT = 1000
INITIAL_BACKFILL_DAYS = 365
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
BINANCE_API_KEY = None
BINANCE_API_SECRET = None

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
    "hour_sin",
    "hour_cos",
]

LOOKAHEAD_BARS = 6
TP_MULTIPLIER = 1.5
SL_MULTIPLIER = 1.0

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