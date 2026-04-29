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

Dashboard operativo profesional en Streamlit + Plotly. Es de solo lectura: abre SQLite en modo `mode=ro` y no ejecuta órdenes ni cambia el estado del bot.

Datos usados:

- SQLite configurado por `SQLITE_DB_PATH` / `config.DB_FILE`.
- Tablas: `prices`, `data_coverage`, `data_gaps`, `model_registry`, `signals`, `orders`, `fills`, `positions`, `portfolio_snapshots`, `paper_model_metrics`, `bot_events`, `risk_events`, `bot_status`.
- Fallbacks de `reports/`: `backtest_oos_equity*.csv`, `backtest_oos_signals*.csv`, `backtest_oos_summary*.json`, `validation_equity*.csv`, `validation_summary*.json`.
- Logs filtrados en `logs/*.log` solo para líneas útiles: `ERROR`, `WARNING`, rejected, risk, gap, failed, blocked.

Secciones:

1. **Overview**: header operativo, modo actual, estado del bot, KPIs, equity curve, benchmark buy & hold si existe, precio + señales.
2. **Portfolio**: equity/drawdown, PnL por operación/backtest, posiciones actuales y exposición.
3. **Signals**: últimas señales con modelo, símbolo, timeframe y confianza.
4. **Orders**: órdenes recientes y fills.
5. **Models**: registry, métricas OOS/backtest, accepted/rejected y resumen de reportes.
6. **Data Quality**: cobertura, gaps abiertos y row counts.
7. **Logs / Ops**: límites de riesgo, risk events y logs críticos.

Si falta una tabla o la DB no existe, el dashboard no debe crashear; muestra el path esperado y comandos sugeridos para generar datos.

Comandos recomendados antes de abrirlo:

```bash
python src/db_utils.py --init --check-schema
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
python src/model_maintenance.py --target-accepted-models 5 --max-attempts 50
python src/trading_bot.py --mode paper --paper-mode per-model --run-once
python src/paper_model_evaluator.py --evaluate-active
streamlit run src/dashboard.py
```

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

## Tools install/run

Preparar el proyecto completo en local:

```powershell
.tools\install.cmd
```

Esto crea directorios persistentes, conserva `.env` si ya existe, inicializa/migra SQLite y valida el esquema.

Lanzar operación autónoma completa, incluyendo dashboard:

```powershell
.tools\run.cmd
```

Procesos lanzados por el runner:

- `realtime_ingestor`
- `trading_bot` en `--mode paper --paper-mode per-model`
- `paper_model_evaluator`
- `model_maintenance`
- `dashboard` con Streamlit

Ver estado desde terminal:

```powershell
.tools\status.cmd
```

El dashboard muestra `Bot: RUNNING` si el `autonomous_runner` mantiene heartbeats recientes. Si no hay heartbeat o está stale, muestra `OFF/STALE`.

## Entrenamiento multi-cripto vs individual por cripto

El entrenamiento soporta dos scopes explícitos:

- `multi_symbol`: un único modelo se entrena con todos los símbolos configurados. Usa `symbol_code` como feature para distinguir cripto dentro del mismo modelo.
- `per_symbol`: entrena un modelo separado por cada cripto. Cada artifact mantiene `symbol_code` por compatibilidad, pero en la práctica el modelo solo ve un símbolo.

Config por defecto en `.env`:

```env
TRAINING_SCOPE=both
```

### Entrenar ambos modos autom?ticamente

```powershell
python src/train.py --training-scope both --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
python src/model_maintenance.py --training-scope both --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --target-accepted-models 5 --max-attempts 50
```

### Entrenar un modelo multi-symbol

```powershell
python src/train.py --training-scope multi-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
python src/validate_model.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --model-id MODEL_ID
python src/backtest.py --mode oos --timeframe 1h --model-id MODEL_ID
```

Mantenimiento automático del pool multi-symbol:

```powershell
python src/model_maintenance.py --training-scope multi-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --target-accepted-models 5 --max-attempts 50
```

### Entrenar modelos individuales por cripto

```powershell
python src/train.py --training-scope per-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

Esto genera un modelo por símbolo. Para validar/backtest individualmente:

```powershell
python src/validate_model.py --symbols BTCUSDT --timeframe 1h --model-id MODEL_ID_BTC
python src/backtest.py --mode oos --timeframe 1h --model-id MODEL_ID_BTC
```

Mantenimiento automático del pool per-symbol:

```powershell
python src/model_maintenance.py --training-scope per-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --target-accepted-models 2 --max-attempts 20
```

### Comparar ambos modos en paper trading

Paper con modelos multi-symbol:

```powershell
python src/trading_bot.py --mode paper --paper-mode per-model --training-scope multi-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --run-once
```

Paper con modelos per-symbol:

```powershell
python src/trading_bot.py --mode paper --paper-mode per-model --training-scope per-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --run-once
```

Paper comparando/ejecutando ambos scopes elegibles:

```powershell
python src/trading_bot.py --mode paper --paper-mode per-model --training-scope both --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --run-once
```

El `model_registry` guarda `symbol_scope`, `training_scope`, `symbols_json`, `timeframe` y `selection_score`. La selecci?n usa m?tricas observadas de validaci?n/backtest/paper para ordenar candidatos; esto busca el mejor edge observado, no garantiza rentabilidad futura.
