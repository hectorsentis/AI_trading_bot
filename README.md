# Plataforma autÃ³noma de trading algorÃ­tmico para Binance Spot

Sistema local-first para investigaciÃ³n, ingesta, validaciÃ³n temporal, paper trading multi-modelo y preparaciÃ³n segura para trading real en Binance Spot. El proyecto busca trazabilidad y evaluaciÃ³n estadÃ­stica; **no garantiza rentabilidad**.

## Seguridad por defecto

Por defecto no puede enviar Ã³rdenes reales:

```env
DRY_RUN=true
ENABLE_TESTNET_PAPER_TRADING=true
ENABLE_LOCAL_SIMULATED_PAPER=true
ENABLE_LIVE_TRADING=false
ENABLE_REAL_ORDER_EXECUTION=false
ENABLE_REAL_BINANCE_ACCOUNT=false
ALLOW_AUTO_PROMOTE_TO_REAL=false
```

Una orden real solo puede pasar si todas estas variables estÃ¡n activadas explÃ­citamente:

```env
ENABLE_LIVE_TRADING=true
ENABLE_REAL_ORDER_EXECUTION=true
ENABLE_REAL_BINANCE_ACCOUNT=true
DRY_RUN=false
```

AdemÃ¡s, toda ruta real pasa por `KillSwitch`, `RiskManager` y `LiveTradingEngine`. No guardes `.env`, claves, bases SQLite, modelos, logs ni reportes en Git.

## Arquitectura

- Datos: `download_data.py`, `realtime_ingestor.py`, `data_loader.py`, `data_quality_service.py`, `data_check.py`, `data_gap_fill.py`
- Features/labels: `features.py`, `technical_patterns.py`, `labels.py`, `feature_store.py`
- Modelos: `train.py`, `validate_model.py`, `backtest.py`, `strategy_evaluator.py`, `model_registry.py`, `model_pool_manager.py`, `model_maintenance.py`
- Trading: `broker_client.py`, `execution_engine.py`, `paper_trading_engine.py`, `portfolio_manager.py`, `risk_manager.py`, `kill_switch.py`, `live_trading_engine.py`, `trading_bot.py`
- EvaluaciÃ³n paper: `paper_model_evaluator.py`
- Dashboard: `dashboard.py`

## ConfiguraciÃ³n

```bash
cp .env.example .env
```

Configura en `.env`:

- `SYMBOLS`, `TIMEFRAME` y opcionalmente `TIMEFRAMES=15m,1h,4h`
- `LOOKAHEAD_BARS`, `TP_MULTIPLIER`, `SL_MULTIPLIER`, `REQUIRE_TP_SL_ON_ENTRY=true`
- credenciales `BINANCE_TESTNET_API_KEY/SECRET` para Spot Testnet/Demo Mode
- credenciales `BINANCE_REAL_API_KEY/SECRET` solo para real, desactivado por defecto
- lÃ­mites de riesgo y criterios de paper validation
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

### 1. Descargar histÃ³rico

```bash
python src/download_data.py --mode full
python src/data_loader.py --gap-check --no-prompt
```

### 2. Ingesta realtime/incremental

```bash
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

Multi-timeframe:

```bash
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h
python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h
python src/model_maintenance.py --training-scope per_symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h --target-accepted-models 2 --max-attempts 50
python src/trading_bot.py --mode paper --paper-mode per-model --training-scope per_symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 15m 1h 4h --run-once
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

El entrenamiento usa splits temporales, no random split. La ventana reciente configurada por `TRAINING_CUTOFF_HOURS_BEFORE_NOW` y `VALIDATION_WINDOW_HOURS` queda fuera del entrenamiento. AdemÃ¡s se aplica un embargo de `LOOKAHEAD_BARS` antes de validaciÃ³n/OOS para que las etiquetas triple-barrier del train no vean velas de validaciÃ³n.

Politica obligatoria de take-profit / stop-loss:

- Las etiquetas de entrenamiento son `triple_barrier_tp_sl_v2`: una clase `LONG` representa una entrada cuyo take-profit ATR se alcanza antes del stop-loss ATR dentro de `LOOKAHEAD_BARS`.
- `TP_MULTIPLIER` y `SL_MULTIPLIER` definen esas barreras sobre `atr_14`; esos mismos multiplicadores se reutilizan al generar senales paper/live.
- `REQUIRE_TP_SL_ON_ENTRY=true` hace que `RiskManager` rechace cualquier entrada long sin `take_profit_price` y `stop_loss_price` validos.
- En `testnet_paper`, tras una compra se intenta colocar una orden protectora OCO de venta con TP/SL. En `local_paper`, la orden queda auditada con bracket local registrado.
- Binance Spot no abre shorts reales; las senales `SHORT` siguen siendo etiqueta de investigacion/risk-off y se ejecutan como reducir/cerrar posicion.

Notas de auditorÃ­a de entrenamiento:

- Los features se calculan por sÃ­mbolo; no se mezclan rolling windows entre criptos.
- `TRAINING_SCOPE=per_symbol` es el modo recomendado para un bot aislado por sÃ­mbolo.
- Las mÃ©tricas econÃ³micas por defecto respetan Binance Spot: una seÃ±al `SHORT` se evalÃºa como `FLAT/exit`, no como short real.
- Los modelos rechazados quedan en `model_registry` con razones; no se ocultan ni se sobrescriben.
- Si no se encuentra edge OOS suficiente, el sistema debe rechazar candidatos en vez de activar bots dÃ©biles.

### 6. Paper trading multi-modelo

Modo por defecto: `per-model`.

```bash
python src/trading_bot.py --mode paper --paper-mode per-model --run-once
python src/trading_bot.py --mode paper --paper-mode per-model --loop
```

Prueba controlada de orden en Binance Spot Demo/Testnet, sin usar real:

```bash
python src/paper_demo_probe.py --symbol BTCUSDT --timeframe 1h
```

Esta prueba usa `BinanceSpotClient.testnet_execution_client()`, `RiskManager`, `ExecutionEngine`, `PortfolioManager` y registra `orders`/`fills`/`positions`/`portfolio_snapshots` en SQLite. Se niega a correr si `DRY_RUN=false` o si cualquier flag real estÃ¡ activo.

Si hay credenciales Testnet y `ENABLE_TESTNET_PAPER_TRADING=true`, usa Binance Spot Testnet/Demo Mode (`testnet_paper`). Si falla, registra el error; no cae nunca a real. Para paper local simulado usa `local_paper`.

Toda entrada long queda guardada en `signals` y `orders` con `take_profit_price`, `stop_loss_price`, `risk_reward`, `protection_required` y `protection_status` para auditoria en SQLite/dashboard.

Modo ensemble opcional:

```bash
python src/trading_bot.py --mode paper --paper-mode ensemble --run-once
```

### 7. Evaluar paper y promocionar

```bash
python src/paper_model_evaluator.py --evaluate-active
```

Respeta muestra mÃ­nima (`MIN_PAPER_VALIDATION_DAYS` o `MIN_PAPER_VALIDATION_TRADES`). Si pasa criterios pasa a `paper_validated` y `real_ready`. Si falla con muestra suficiente pasa a `paper_rejected`.

### 8. Trading real preparado pero apagado

Inspeccionar gates:

```bash
python src/live_trading_engine.py --model-id MODEL_ID
```

No activa real salvo flags explÃ­citos y `ALLOW_AUTO_PROMOTE_TO_REAL=true`.

### 9. Dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard operativo profesional en Streamlit + Plotly. Es un panel de control simplificado para operar el bot desde un VPS:

- Lee datos operativos reales desde SQLite y `reports/`.
- Muestra estado global, balance/equity, PnL, drawdown, posiciones, se?ales, ?rdenes, modelos, calidad de datos y logs cr?ticos.
- Tiene login configurable.
- Permite ejecutar acciones operativas desde el VPS y registrar auditor?a en SQLite.
- **No ejecuta compras/ventas manuales y no puede saltarse los flags externos para trading real.**
- La UI evita bloques JSON crudos: m?tricas, configuraci?n, acciones y reportes se presentan como tablas/indicadores legibles.

#### Autenticaci?n

Por defecto el dashboard requiere login:

```env
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD_HASH=
DASHBOARD_SECRET_KEY=
LIVE_TRADING_ALLOWED=false
DASHBOARD_ALLOW_SERVER_ACTIONS=true
```

Generar hash PBKDF2:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from dashboard_auth import generate_password_hash; print(generate_password_hash('CAMBIA_ESTA_PASSWORD'))"
```

Copia el resultado a `DASHBOARD_PASSWORD_HASH` en `.env`. Usa un `DASHBOARD_SECRET_KEY` aleatorio y no expongas el panel en servidor sin HTTPS/reverse proxy.

Datos usados:

- SQLite configurado por `SQLITE_DB_PATH` / `config.DB_FILE`.
- Tablas le?das: `prices`, `data_coverage`, `data_gaps`, `ingestion_log`, `model_registry`, `signals`, `orders`, `fills`, `positions`, `portfolio_snapshots`, `paper_model_metrics`, `bot_events`, `risk_events`, `bot_status`, `validation_predictions`.
- Tablas creadas por el dashboard si la DB existe: `bot_control_actions`, `model_control`, `runtime_config`, `runtime_config_audit`.
- Fallbacks de `reports/`: `backtest_oos_equity*.csv`, `backtest_oos_signals*.csv`, `backtest_oos_summary*.json`, `validation_equity*.csv`, `validation_summary*.json`.
- Logs filtrados en `logs/*.log` solo para l?neas ?tiles: `ERROR`, `WARNING`, rejected, risk, gap, failed, blocked.

Secciones:

1. **Control**: KPIs principales, gr?fico de velas con se?ales/?rdenes, kill switch visible, start/stop, pausa/reanudaci?n de se?ales, paper trading, refresh, data check, entrenamiento, ?ltimas se?ales y ?rdenes.
2. **Portfolio**: equity/drawdown, PnL, exposici?n y posiciones.
3. **Models**: registry, m?tricas OOS/backtest/paper, detalle por modelo y controles auditables para activar/desactivar se?ales o paper trading.
4. **Data / Logs**: calidad de datos, gaps, ingestion log, risk events, logs cr?ticos y configuraci?n runtime.

Controles disponibles:

- `Activate model` / `Deactivate model`: actualiza `model_registry.is_active`, `model_control.signal_enabled` y registra acci?n auditable.
- `Enable/Disable paper trading`: actualiza `model_control.paper_enabled` y registra acci?n auditable.
- `Kill switch`: detiene procesos registrados y desactiva signals, paper y live runtime.
- `Request new training`: lanza `model_maintenance.py` en background en el servidor y registra `REQUEST_RETRAIN` en `bot_control_actions`; no bloquea la UI.
- Configuraci?n editable: universe/timeframe, pol?tica de modelos, `min_confidence`, retraining, l?mites de riesgo runtime, `dry_run`, paper trading, supuestos de fees/slippage, ingesta y gap checks.

Trading real:

- `execution.live_trading_enabled` est? bloqueado por defecto y puede activarse desde el dashboard solo con reautenticaci?n y si el VPS ya tiene activados expl?citamente `LIVE_TRADING_ALLOWED=true`, `ENABLE_LIVE_TRADING=true`, `ENABLE_REAL_ORDER_EXECUTION=true`, `ENABLE_REAL_BINANCE_ACCOUNT=true` y `DRY_RUN=false`.
- El dashboard no puede saltarse esos flags de entorno.
- Aunque se active live runtime, siguen aplicando `RiskManager` y `KillSwitch`.

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

Probar login/logout:

1. Configura `DASHBOARD_AUTH_ENABLED=true`, `DASHBOARD_USERNAME` y `DASHBOARD_PASSWORD_HASH`.
2. Ejecuta `streamlit run src/dashboard.py`.
3. Entra con las credenciales y usa `Logout` en el sidebar.

Probar acciones auditables:

```sql
SELECT * FROM bot_control_actions ORDER BY requested_at_utc DESC LIMIT 20;
SELECT * FROM model_control ORDER BY updated_at_utc DESC;
SELECT * FROM runtime_config_audit ORDER BY updated_at_utc DESC LIMIT 20;
```

## VerificaciÃ³n local mÃ­nima

```bash
python src/db_utils.py --init --check-schema
python src/platform_checks.py
python src/broker_client.py --healthcheck --symbol BTCUSDT
```

Comprueba que el JSON de `platform_checks.py` contiene `real_orders_blocked_by_default: true`.

## EjecuciÃ³n sugerida en servidor

Procesos separados:

```bash
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --loop
python src/model_maintenance.py --target-accepted-models 5 --max-attempts 50
python src/trading_bot.py --mode paper --paper-mode per-model --loop
python src/paper_model_evaluator.py --evaluate-active
streamlit run src/dashboard.py --server.address 0.0.0.0
```

Usa systemd o un supervisor. MantÃ©n `SQLITE_DB_PATH`, `logs/`, `models/` y `reports/` en almacenamiento persistente y fuera de Git.

## Consultas SQLite Ãºtiles

```sql
SELECT status, COUNT(*) FROM model_registry GROUP BY status;
SELECT model_id, account_mode, symbol, quantity, avg_price FROM positions;
SELECT model_id, account_mode, status, COUNT(*) FROM orders GROUP BY model_id, account_mode, status;
SELECT * FROM paper_model_metrics ORDER BY evaluated_at_utc DESC LIMIT 20;
```

## Troubleshooting

- Sin modelos activos: ejecuta feature store, entrenamiento, validaciÃ³n/backtest y `model_pool_manager.py`.
- Testnet falla: revisa `BINANCE_TESTNET_API_KEY/SECRET`; el sistema no usarÃ¡ real como fallback.
- Dashboard vacÃ­o: inicializa DB y ejecuta ingesta/feature store.
- Orden rechazada: mira `risk_events` y lÃ­mites en `.env`.
- Real bloqueado: esperado por defecto; revisa los cuatro flags obligatorios si realmente quieres activar real.

## Tools install/run

Preparar el proyecto completo en local:

```powershell
.tools\install.cmd
```

Esto crea directorios persistentes, conserva `.env` si ya existe, inicializa/migra SQLite y valida el esquema.

Lanzar operaciÃ³n autÃ³noma completa, incluyendo dashboard:

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

El dashboard muestra `Bot: RUNNING` si el `autonomous_runner` mantiene heartbeats recientes. Si no hay heartbeat o estÃ¡ stale, muestra `OFF/STALE`.

## Entrenamiento multi-cripto vs individual por cripto

El entrenamiento soporta dos scopes explÃ­citos:

- `multi_symbol`: un Ãºnico modelo se entrena con todos los sÃ­mbolos configurados. Usa `symbol_code` como feature para distinguir cripto dentro del mismo modelo.
- `per_symbol`: entrena un modelo separado por cada cripto. Cada artifact mantiene `symbol_code` por compatibilidad, pero en la prÃ¡ctica el modelo solo ve un sÃ­mbolo.

Config recomendada por defecto en `.env` para separaciÃ³n estricta por sÃ­mbolo:

```env
TRAINING_SCOPE=per_symbol
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

Mantenimiento automÃ¡tico del pool multi-symbol:

```powershell
python src/model_maintenance.py --training-scope multi-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --target-accepted-models 5 --max-attempts 50
```

### Entrenar modelos individuales por cripto

```powershell
python src/train.py --training-scope per-symbol --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

Esto genera un modelo por sÃ­mbolo. Para validar/backtest individualmente:

```powershell
python src/validate_model.py --symbols BTCUSDT --timeframe 1h --model-id MODEL_ID_BTC
python src/backtest.py --mode oos --timeframe 1h --model-id MODEL_ID_BTC
```

Mantenimiento automÃ¡tico del pool per-symbol:

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


