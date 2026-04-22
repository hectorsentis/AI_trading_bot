# Trading Bot Research Base (Binance Spot)

Base local para pipeline cuantitativo con SQLite y validacion temporal estricta.

## Modulos principales

- `src/download_data.py`: descarga historico Binance Spot a snapshots raw.
- `src/data_loader.py`: consolida snapshots en SQLite (`prices`) con upsert idempotente.
- `src/data_check.py`: detecta gaps en la serie y actualiza `data_gaps`.
- `src/data_gap_fill.py`: descarga rangos faltantes basados en `data_gaps`.
- `src/feature_store.py`: calcula features + labels triple barrier y persiste `features`.
- `src/train.py`: entrena LightGBM multiclass y registra el modelo en `model_registry`.
- `src/validate_model.py`: validacion walk-forward temporal (sin split aleatorio).
- `src/backtest.py`: backtest economico LONG/FLAT/SHORT con costes.
- `src/predict.py`: genera senal actual desde el ultimo modelo.

## Estructura de datos

- `data/raw/`: snapshots CSV por simbolo.
- `data/db/market_data.sqlite`: base SQLite local.
- `models/`: artefactos de modelos entrenados (`.joblib`).
- `reports/`: reportes JSON/CSV de train, validacion y backtest.
- `logs/data_quality/`: reportes de calidad (gaps).

## Flujo recomendado

1. Descargar historico:

```bash
python src/download_data.py --mode incremental
```

2. Cargar snapshots en SQLite:

```bash
python src/data_loader.py
```

3. Revisar gaps (opcional/manual):

```bash
python src/data_check.py
python src/data_gap_fill.py
python src/data_loader.py
```

4. Construir feature store:

```bash
python src/feature_store.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

5. Entrenar modelo:

```bash
python src/train.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

6. Validar con walk-forward:

```bash
python src/validate_model.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

7. Backtest economico:

```bash
python src/backtest.py --timeframe 1h
```

8. Generar senal actual:

```bash
python src/predict.py --timeframe 1h
```

## Seguridad

- `ENABLE_TRADING = False`
- `DRY_RUN = True`

No se envian ordenes reales en este estado.
