# Trading Bot Research Base (Binance Spot)

Base local-first para investigacion cuantitativa, validacion temporal estricta, gating de modelos y paper trading autonomo en Binance Spot.

Este repositorio esta preparado para operar como **bot autonomo en paper/dry-run** con datos reales actuales, pero **NO envia ordenes reales** en el estado actual.

## Seguridad obligatoria

Valores por defecto en `src/config.py`:

```python
ENABLE_TRADING = False
DRY_RUN = True
```

Reglas de seguridad:

- No hay envio de ordenes reales en este step.
- `broker_client.py` separa roles de cliente: market data, account read y simulated execution.
- Las credenciales nunca se hardcodean: solo entorno o `.env`.
- Para Spot paper trading no se abren cortos: una senal research `SHORT` se interpreta como bajista/risk-off y se traduce a reducir/cerrar largo o permanecer fuera.
- Un modelo entrenado no queda aceptado ni activo automaticamente: debe pasar gating con evidencia temporal/OOS.

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En PowerShell, si aparecen problemas de salida Unicode, usa:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## Credenciales Binance seguras

Copia `.env.example` a `.env` y rellena solo si necesitas account info:

```env
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_RECV_WINDOW_MS=5000
ENABLE_TRADING=False
DRY_RUN=True
```

Recomendado:

- Credenciales mainnet **read-only** para inspeccion de cuenta/balances.
- Credenciales separadas de Binance Spot Testnet para futuras pruebas de ejecucion.
- No concedas permisos de trading a credenciales de solo datos.

Healthcheck publico y prueba de permisos:

```bash
python src/broker_client.py --healthcheck --ticker --recent-klines 2 --account-info --symbol BTCUSDT
```

Si `account-info` falla por permisos, el paper trading puede seguir usando market data publica.

## Modulos principales

### Datos

- `src/download_data.py`: descarga historico Binance Spot a snapshots raw.
- `src/data_loader.py`: consolida snapshots en SQLite (`prices`) con upsert idempotente.
- `src/data_check.py`: detecta gaps y actualiza `data_gaps`.
- `src/data_gap_fill.py`: intenta rellenar huecos detectados.
- `src/coverage_report.py`: reporte final de cobertura.

### Datos derivados

- `src/features.py`: calculo de features.
- `src/labels.py`: etiquetado triple barrier multiclass.
- `src/feature_store.py`: persistencia incremental/full de features + labels.

### Modelo / validacion

- `src/train.py`: entrenamiento LightGBM baseline y registro en `model_registry`.
- `src/validate_model.py`: walk-forward temporal estricto y persistencia OOS en `validation_predictions`.
- `src/backtest.py`: backtest economico `in_sample` u `oos`.
- `src/strategy_evaluator.py`: gating formal accepted/candidate/rejected.
- `src/model_registry.py`: inspeccion/estado de modelos.
- `src/predict.py`: senal actual desde modelo apto para inferencia.

### Paper trading autonomo

- `src/broker_client.py`: REST wrapper seguro Binance Spot.
- `src/signal_engine.py`: convierte probabilidades en senal operativa filtrada.
- `src/risk_manager.py`: reglas de riesgo Spot paper.
- `src/execution_engine.py`: ordenes y fills simulados, sin live orders.
- `src/portfolio_manager.py`: cash, equity, posiciones y PnL simulados.
- `src/trading_bot.py`: loop autonomo local.

## SQLite como fuente de verdad

BD: `data/db/market_data.sqlite`

Tablas principales:

- `prices`
- `ingestion_log`
- `data_gaps`
- `data_coverage`
- `features`
- `model_registry`
- `validation_predictions`
- `signals`
- `orders`
- `positions`
- `portfolio_snapshots`

Auditoria rapida:

```bash
python -c "import sqlite3,pandas as pd; conn=sqlite3.connect('data/db/market_data.sqlite'); [print(t, pd.read_sql_query(f'SELECT COUNT(*) n FROM {t}', conn).iloc[0,0]) for t in ['prices','features','model_registry','signals','orders','positions','portfolio_snapshots']]; conn.close()"
```

## Flujo operativo completo

### 1) Descargar historicos solidos

Full backfill 1h para el scope inicial:

```bash
python src/download_data.py --mode full --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

Incremental diario/periodico:

```bash
python src/download_data.py --mode incremental --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

Preparado para otros timeframes sin romper 1h:

```bash
python src/download_data.py --mode incremental --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 15m
python src/download_data.py --mode incremental --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 4h
```

### 2) Consolidar snapshots en SQLite

```bash
python src/data_loader.py --no-prompt --gap-check
```

### 3) Intentar rellenar gaps y volver a consolidar

```bash
python src/data_gap_fill.py
python src/data_loader.py --no-prompt --gap-check
```

Algunos gaps historicos de Binance pueden ser mantenimiento real del exchange y no recuperarse por API. Deben quedar visibles como pendientes, no ocultos.

### 4) Reporte de cobertura

```bash
python src/coverage_report.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 1h --output reports/coverage_final_1h.csv
```

Campos:

- `min_datetime_utc`
- `max_datetime_utc`
- `row_count`
- `expected_rows`
- `coverage_pct`
- `gaps_detected`
- `gaps_resolved`
- `gaps_pending`

### 5) Recalcular feature store + labels

Full rebuild despues de ampliar historico:

```bash
python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --full-rebuild
```

Incremental normal:

```bash
python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

### 6) Entrenar baseline LightGBM

```bash
python src/train.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

No uses splits aleatorios. El script usa holdout temporal.

### 7) Validacion walk-forward OOS

```bash
python src/validate_model.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --model-id <MODEL_ID> --max-folds 20
```

### 8) Backtest OOS estricto

```bash
python src/backtest.py --mode oos --timeframe 1h --model-id <MODEL_ID>
```

El backtest OOS usa `validation_predictions`, no predicciones in-sample.

### 9) Inspeccionar modelos

```bash
python src/model_registry.py --limit 20
```

Estados:

- `accepted`: cumple criterios configurados.
- `candidate`: falta evidencia suficiente pero no hay fallo duro.
- `rejected`: falla criterios minimos.
- `active`: modelo aceptado marcado para inferencia preferente.

## Paper trading autonomo local

Run once con datos actuales de Binance, feature refresh y ejecucion simulada:

```bash
python src/trading_bot.py \
  --run-once \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --timeframe 1h \
  --paper-initial-cash 10000 \
  --sync-latest-from-binance \
  --refresh-features \
  --log-level INFO
```

Loop continuo local:

```bash
python src/trading_bot.py \
  --loop \
  --poll-seconds 60 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --timeframe 1h \
  --paper-initial-cash 10000 \
  --sync-latest-from-binance \
  --refresh-features \
  --log-level INFO
```

Propiedades del loop:

- Se puede parar con Ctrl+C.
- No duplica ordenes para la misma vela/modelo/simbolo.
- Si no cambia la senal o ya esta en target, registra `SKIPPED`.
- Persiste `signals`, `orders`, `portfolio_snapshots` y posiciones cuando existan fills.

## SHORT research vs Spot execution

El modelo trabaja en espacio research:

- `SHORT`
- `FLAT`
- `LONG`

Pero Binance Spot ejecutable en paper usa:

- `BUY`
- `HOLD`
- `SELL`
- `FLAT`

Mapeo seguro:

- `LONG`: abrir o mantener largo.
- `FLAT`: cerrar largo o mantenerse fuera.
- `SHORT`: senal bajista investigativa; cerrar/reducir largo o quedarse fuera. **No abre corto Spot**.

## Activacion futura

Para evolucionar hacia trading real faltan deliberadamente controles adicionales:

- paper trading prolongado con auditoria.
- reconciliacion robusta contra balances reales.
- soporte explicito Spot Testnet para ejecucion no simulada.
- aprobacion manual del operador.
- controles de permisos separados por cliente.
- revision de modelos aceptados con evidencia OOS suficiente.

Hasta entonces, el sistema debe permanecer en `DRY_RUN=True` y `ENABLE_TRADING=False`.


## Features de figuras / analisis tecnico

`FEATURE_VERSION = "v2_ta_patterns"` añade proxies cuantitativos sin mirar al futuro:

- Ratios de cuerpo y mechas: `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`.
- Velas: `is_doji`, `is_hammer`, `is_shooting_star`.
- Patrones de 2 velas: `bullish_engulfing`, `bearish_engulfing`, `inside_bar`, `outside_bar`.
- Rupturas: `breakout_20`, `breakdown_20`, `ma_cross_5_20`.
- Figuras aproximadas: `double_top_proxy`, `double_bottom_proxy`.

Son señales numéricas para el modelo; no implican edge por sí solas. Siempre hay que validar OOS.

## Demo trading / Testnet

Si tus claves son de Binance Spot Testnet o demo, configura:

```env
BINANCE_USE_TESTNET=True
```

El bot sigue bloqueado para real trading:

```python
ENABLE_TRADING = False
DRY_RUN = True
```

El cliente público de market data sigue usando Binance mainnet para velas reales. El cliente account/read usa testnet cuando `BINANCE_USE_TESTNET=True`.

## Arrancar bot desde Windows

Lanzador creado:

```powershell
.tools\run_trading_bot_loop.cmd
```

También puedes arrancar manualmente:

```powershell
cmd.exe /c start "TradingV02PaperBot" /min "E:\Trading\v02\.tools\run_trading_bot_loop.cmd"
```

Logs:

- `logs/trading_bot.log`
- `logs/trading_bot_scheduled.stderr.log`
- `logs/trading_bot_scheduled.stdout.log`

## Pool autonomo de modelos aceptados

Variable principal:

```env
TARGET_ACCEPTED_MODELS=3
ENABLE_MODEL_POOL_MAINTENANCE=True
MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE=4
MODEL_POOL_VALIDATION_MAX_FOLDS=5
MODEL_POOL_TRAINING_ENABLED_IN_BOT=True
```

Comportamiento:

- El sistema cuenta modelos con `acceptance_status='accepted'` para el timeframe.
- Si hay menos de `TARGET_ACCEPTED_MODELS`, `src/model_maintenance.py` entrena variantes LightGBM con distintos hiperparametros/thresholds.
- Cada intento ejecuta: train temporal -> walk-forward OOS -> backtest OOS -> gating.
- Solo modelos que pasan gating quedan `accepted`.
- Los rechazados quedan registrados como `rejected`; no se usan para paper trading.
- El bot usa un ensemble de modelos aceptados disponibles y promedia probabilidades antes del `signal_engine`.
- Si no alcanza el objetivo en un ciclo, vuelve a intentarlo en ciclos posteriores, acotado por `MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE` para evitar bucles infinitos.

Ejecutar mantenimiento manual:

```powershell
python src/model_maintenance.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --target-accepted-models 3 --max-attempts 4 --validation-max-folds 5
```

Arrancar bot con mantenimiento del pool:

```powershell
python src/trading_bot.py --loop --poll-seconds 60 --symbols BTCUSDT,ETHUSDT,SOLUSDT --timeframe 1h --sync-latest-from-binance --refresh-features --target-accepted-models 3
```

Para probar el bot sin entrenar en ese ciclo:

```powershell
python src/trading_bot.py --run-once --skip-model-maintenance --symbols BTCUSDT,ETHUSDT,SOLUSDT --timeframe 1h
```

El loop no reentrena en cada poll: usa `MODEL_POOL_MAINTENANCE_INTERVAL_SECONDS` para espaciar el mantenimiento del pool. Por defecto son 3600 segundos. Si el pool cae por debajo de `TARGET_ACCEPTED_MODELS`, en la siguiente ventana de mantenimiento intentara nuevos modelos hasta `MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE`.
