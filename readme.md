# Plataforma autónoma de trading algorítmico para Binance Spot

Sistema local-first para investigación, ingesta, validación temporal, paper trading multi-modelo y preparación segura para trading real en Binance Spot. El proyecto busca trazabilidad y evaluación estadística; **no garantiza rentabilidad**.

## Seguridad por defecto

Por defecto no puede enviar órdenes reales:

```env
DRY_RUN=true
ENABLE_TESTNET_PAPER_TRADING=true
ENABLE_LOCAL_SIMULATED_PAPER=true
ENABLE_LIVE_TRADING=false
ENABLE_REAL_ORDER_EXECUTION=false
ENABLE_REAL_BINANCE_ACCOUNT=false
ALLOW_AUTO_PROMOTE_TO_REAL=false
```

Una orden real solo puede pasar si todas estas variables están activadas explícitamente:

```env
ENABLE_LIVE_TRADING=true
ENABLE_REAL_ORDER_EXECUTION=true
ENABLE_REAL_BINANCE_ACCOUNT=true
DRY_RUN=false
```

Además, toda ruta real pasa por `KillSwitch`, `RiskManager` y `LiveTradingEngine`. No guardes `.env`, claves, bases SQLite, modelos, logs ni reportes en Git.

## Arquitectura

- Datos: `download_data.py`, `realtime_ingestor.py`, `data_loader.py`, `data_quality_service.py`, `data_check.py`, `data_gap_fill.py`
- Features/labels: `features.py`, `technical_patterns.py`, `labels.py`, `feature_store.py`
- Modelos: `train.py`, `validate_model.py`, `backtest.py`, `strategy_evaluator.py`, `model_registry.py`, `model_pool_manager.py`, `model_maintenance.py`
- Trading: `broker_client.py`, `execution_engine.py`, `paper_trading_engine.py`, `portfolio_manager.py`, `risk_manager.py`, `kill_switch.py`, `live_trading_engine.py`, `trading_bot.py`
- Evaluación paper: `paper_model_evaluator.py`
- Dashboard: `dashboard.py`

## Configuración

```bash
cp .env.example .env
```

Configura en `.env`:

- `SYMBOLS`, `TIMEFRAME`
- credenciales `BINANCE_TESTNET_API_KEY/SECRET` para Spot Testnet/Demo Mode
- credenciales `BINANCE_REAL_API_KEY/SECRET` solo para real, desactivado por defecto
- límites de riesgo y criterios de paper validation
- `SQLITE_DB_PATH`

## Base de datos

Inicializar/migrar esquema:

```bash
python src/db_utils.py --init
python src/db_utils.py --check-schema
```

Tablas operativas: `prices`, `data_coverage`, `data_gaps`, `features`, `model_registry`, `signals`, `orders`, `fills`, `positions`, `portfolio_snapshots`, `paper_model_metrics`, `model_lifecycle_events`, `bot_events`, `risk_events`.

`positions`, `orders`, `signals`, `fills` y `portfolio_snapshots` incluyen `model_id`; las posiciones se separan por `(model_id, symbol, timeframe, account_mode)`.

## Workflow

### 1. Descargar histórico

```bash
python src/download_data.py --mode full
python src/data_loader.py --gap-check --no-prompt
```

### 2. Ingesta realtime/incremental

```bash
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

### 3. Calidad de datos

```bash
python src/data_quality_service.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

### 4. Feature store

```bash
python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

### 5. Mantener pool de modelos

```bash
python src/model_maintenance.py --target-accepted-models 5 --max-attempts 50
python src/model_pool_manager.py
```

El entrenamiento usa splits temporales, no random split. La ventana reciente configurada por `TRAINING_CUTOFF_HOURS_BEFORE_NOW` y `VALIDATION_WINDOW_HOURS` queda fuera del entrenamiento.

### 6. Paper trading multi-modelo

Modo por defecto: `per-model`.

```bash
python src/trading_bot.py --mode paper --paper-mode per-model --run-once
python src/trading_bot.py --mode paper --paper-mode per-model --loop
```

Si hay credenciales Testnet y `ENABLE_TESTNET_PAPER_TRADING=true`, usa Binance Spot Testnet/Demo Mode (`testnet_paper`). Si falla, registra el error; no cae nunca a real. Para paper local simulado usa `local_paper`.

Modo ensemble opcional:

```bash
python src/trading_bot.py --mode paper --paper-mode ensemble --run-once
```

### 7. Evaluar paper y promocionar

```bash
python src/paper_model_evaluator.py --evaluate-active
```

Respeta muestra mínima (`MIN_PAPER_VALIDATION_DAYS` o `MIN_PAPER_VALIDATION_TRADES`). Si pasa criterios pasa a `paper_validated` y `real_ready`. Si falla con muestra suficiente pasa a `paper_rejected`.

### 8. Trading real preparado pero apagado

Inspeccionar gates:

```bash
python src/live_trading_engine.py --model-id MODEL_ID
```

No activa real salvo flags explícitos y `ALLOW_AUTO_PROMOTE_TO_REAL=true`.

### 9. Dashboard

```bash
streamlit run src/dashboard.py
```

Muestra overview, registry, paper portfolios, órdenes, fills, performance, data quality y risk events desde SQLite.

## Verificación local mínima

```bash
python src/db_utils.py --init --check-schema
python src/platform_checks.py
python src/broker_client.py --healthcheck --symbol BTCUSDT
```

Comprueba que el JSON de `platform_checks.py` contiene `real_orders_blocked_by_default: true`.

## Ejecución sugerida en servidor

Procesos separados:

```bash
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --loop
python src/model_maintenance.py --target-accepted-models 5 --max-attempts 50
python src/trading_bot.py --mode paper --paper-mode per-model --loop
python src/paper_model_evaluator.py --evaluate-active
streamlit run src/dashboard.py --server.address 0.0.0.0
```

Usa systemd o un supervisor. Mantén `SQLITE_DB_PATH`, `logs/`, `models/` y `reports/` en almacenamiento persistente y fuera de Git.

## Consultas SQLite útiles

```sql
SELECT status, COUNT(*) FROM model_registry GROUP BY status;
SELECT model_id, account_mode, symbol, quantity, avg_price FROM positions;
SELECT model_id, account_mode, status, COUNT(*) FROM orders GROUP BY model_id, account_mode, status;
SELECT * FROM paper_model_metrics ORDER BY evaluated_at_utc DESC LIMIT 20;
```

## Troubleshooting

- Sin modelos activos: ejecuta feature store, entrenamiento, validación/backtest y `model_pool_manager.py`.
- Testnet falla: revisa `BINANCE_TESTNET_API_KEY/SECRET`; el sistema no usará real como fallback.
- Dashboard vacío: inicializa DB y ejecuta ingesta/feature store.
- Orden rechazada: mira `risk_events` y límites en `.env`.
- Real bloqueado: esperado por defecto; revisa los cuatro flags obligatorios si realmente quieres activar real.
