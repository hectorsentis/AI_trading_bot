
Trabaja DIRECTAMENTE sobre esta carpeta de proyecto y conviértela en la base de un sistema de trading algorítmico AUTÓNOMO, modular y escalable, orientado a producción real.

OBJETIVO FINAL DEL PROYECTO
El objetivo final es un bot autónomo que pueda operar con dinero real en Binance Spot mediante su API, con validación estricta, control de riesgo y activación progresiva desde local hasta servidor.

1. Descargue y consolide datos históricos y en tiempo real.
2. Construya datasets limpios y persistentes.
3. Detecte gaps y recupere huecos de datos automáticamente.
4. Genere features útiles de mercado.
5. Entrene modelos de ML robustos y consistentes.
6. Valide dichos modelos de forma estricta, sin fuga de datos y sin autoengaño estadístico.
7. Encuentre un modelo con expectativa matemática positiva (edge estadístico real o potencial).
8. Se reentrene periódicamente para adaptarse y consolidar el modelo.
9. Genere señales operativas en tiempo real.
10. Ejecute órdenes reales vía API del broker/exchange.
11. Tenga gestión de riesgo, control de posiciones y monitorización.
12. Pueda ejecutarse primero en local y más adelante migrarse a un servidor.
13. Use Git/GitHub como parte del flujo de desarrollo:
    - commits estructurados
    - branches por funcionalidad
    - posibilidad de hacer pull/update del proyecto
    - flujo de trabajo limpio y versionado

IMPORTANTE
NO prometas rentabilidad garantizada ni asumas que un modelo es rentable por defecto.
Diseña el sistema para:
- buscar edge estadístico real
- rechazar modelos mediocres
- evitar sobreajuste
- reentrenar con criterio
- medir robustez
- filtrar señales débiles
- evolucionar hacia producción real

MENTALIDAD DE TRABAJO
No quiero una demo académica.
No quiero un juguete.
No quiero scripts sueltos sin arquitectura.
Quiero una BASE SERIA para un BOT AUTÓNOMO DE TRADING.

Trabaja como si tuvieras que dejar el proyecto listo para ser:
- funcional en local ahora
- desplegable en servidor después
- ampliable a tiempo real y ejecución real sin rehacerlo todo

RESTRICCIONES
- Trabaja directamente sobre esta carpeta.
- Reutiliza lo que ya exista y mejóralo si tiene sentido.
- Si una pieza ya existe pero está incompleta, complétala.
- Si falta una pieza crítica, créala.
- No rompas la arquitectura.
- No metas secretos reales en el código.
- Usa rutas relativas con pathlib.
- Mantén separación clara entre:
  - descarga de datos
  - consolidación en BD
  - chequeo de calidad
  - relleno de gaps
  - feature engineering
  - entrenamiento
  - validación
  - backtest
  - predicción
  - tiempo real
  - ejecución
- El sistema debe quedar preparado para Windows local ahora y migración a servidor Linux después.

ARQUITECTURA OBJETIVO

CAPA 1 — DATOS
- download_data.py
  Descarga histórico desde Binance Spot.
- data_loader.py
  Inserta snapshots raw en SQLite con upsert.
- data_check.py
  Detecta gaps.
- data_gap_fill.py
  Rellena gaps.
- realtime_ingestor.py
  (crear si no existe) escucha stream en tiempo real y añade velas/nuevos datos a la BD.

CAPA 2 — DATOS DERIVADOS
- features.py
  Construcción de features.
- labels.py
  Etiquetado tipo triple barrier.
- feature_store.py
  Calcula y persiste features en la BD.

CAPA 3 — MODELO
- train.py
  Entrenamiento serio desde BD.
- validate_model.py
  (crear si no existe) validación robusta walk-forward / temporal.
- model_registry.py
  (crear si no existe) guardar metadata del modelo, métricas, fecha, parámetros, universo, timeframe.

CAPA 4 — INVESTIGACIÓN / ROBUSTEZ
- backtest.py
  Backtest económico realista.
- strategy_evaluator.py
  (crear si no existe) ranking de modelos/estrategias por métricas robustas.
- retrain_scheduler.py
  (crear si no existe) lógica de reentrenamiento periódico.

CAPA 5 — OPERATIVA REAL
- predict.py
  Señal actual.
- signal_engine.py
  (crear si no existe) genera señal final filtrada.
- risk_manager.py
  (crear si no existe) controla tamaño, exposición, stops, pérdidas máximas, filtros.
- execution_engine.py
  (crear si no existe) prepara y envía órdenes.
- broker_client.py
  (crear si no existe) wrapper de Binance Spot API.
- portfolio_manager.py
  (crear si no existe) posiciones, PnL, cash, exposición.
- trading_bot.py
  (crear si no existe) orquestador autónomo.

CAPA 6 — OPERACIÓN / DEVOPS
- Git/GitHub workflow
- logs/
- reports/
- models/
- config.py
- .env.example
- requirements.txt
- README.md

OBJETIVO TÉCNICO INMEDIATO
El PRIMER STEP debe dejar el sistema FUNCIONAL EN LOCAL:
1. descargar datos históricos de Binance
2. guardarlos como snapshots raw
3. consolidarlos en SQLite
4. detectar gaps
5. rellenar gaps
6. calcular features
7. entrenar modelo desde la BD
8. validar el modelo
9. hacer backtest
10. generar señal actual

OBJETIVO TÉCNICO POSTERIOR
Dejar el proyecto PREPARADO para:
1. streaming en tiempo real por WebSocket
2. inserción incremental en la BD
3. reentrenamiento programado
4. paper trading
5. ejecución real vía Binance Spot
6. migración a servidor

DATOS
Usar Binance Spot.
Símbolos iniciales:
- BTCUSDT
- ETHUSDT
- SOLUSDT

Timeframe inicial:
- 1h

Debe quedar preparado para:
- varios símbolos
- varios timeframes
- entrenamiento por símbolo o multiactivo

BASE DE DATOS
Usar SQLite local como fuente de verdad.

Tablas mínimas:
- prices
- ingestion_log
- data_gaps
- data_coverage
- features
- model_registry (crear)
- signals (crear)
- orders (crear)
- positions (crear si necesario)

PRICES
Clave lógica:
(symbol, timeframe, datetime_utc)

DATA COVERAGE
Guardar:
- symbol
- timeframe
- min_datetime_utc
- max_datetime_utc
- row_count
- updated_at_utc

DATA GAPS
Guardar:
- symbol
- timeframe
- gap_start_utc
- gap_end_utc
- missing_bars
- detected_at_utc

MODEL REGISTRY
Guardar:
- model_id
- symbol_scope
- timeframe
- train_start
- train_end
- test_start
- test_end
- feature_version
- label_version
- model_path
- training_ts_utc
- metrics_json
- params_json
- status

GITHUB / GIT
Quiero que el proyecto quede preparado para trabajar bien con Git.
Haz lo siguiente:
- añade o corrige .gitignore
- no subir data/ ni logs/ pesados ni .venv/
- deja README útil
- deja requirements.txt actualizado
- si propones flujo de trabajo, usa branches por funcionalidad:
  - feature/data-pipeline
  - feature/model-training
  - feature/realtime-bot
  - feature/execution-engine

No ejecutes GitHub tú mismo si no puedes, pero deja el repo preparado profesionalmente para:
- pull
- branch
- merge
- evolución ordenada

DESCARGA DE DATOS
La descarga debe:
- crear snapshots raw por símbolo
- no sobreescribir históricos previos
- soportar modos:
  - incremental
  - full
  - recent
  - range
- usar CLI útil
- leer la última fecha existente en BD
- tener un pequeño solape para no perder velas

DATA QUALITY
La calidad de datos es crítica.
No quiero un modelo entrenado sobre basura.
Asegura:
- deduplicación
- timestamps correctos
- orden temporal correcto
- chequeo de gaps
- cobertura temporal guardada
- logs de data quality exportables

FEATURE ENGINEERING
Construir al menos:
- retornos
- ATR
- volatilidad rolling
- volumen relativo
- z-score de volumen
- distancia a medias
- slope de medias
- distancia a máximos/mínimos rolling
- RSI
- componentes cíclicas de hora

Debe quedar preparado para añadir:
- features cross-asset
- features de régimen
- features de volatilidad de mercado
- market structure
- features de order flow si en el futuro hay datos más finos

LABELING
Implementar triple barrier multiclase:
- SHORT
- FLAT
- LONG

Parametrizable.
Sin mirar el futuro incorrectamente.
Conservador en casos ambiguos.

MODELO
No quiero un modelo flojo ni naive.
Quiero una primera base consistente y bien entrenada.

Empieza con:
- LightGBM multiclase
porque es sólido para tabular y rápido.

Pero deja el proyecto preparado para probar:
- XGBoost
- RandomForest
- modelos por símbolo
- modelos multiactivo
- ensembles

VALIDACIÓN
MUY IMPORTANTE:
No uses validación basura.
No uses split aleatorio.
No uses métricas engañosas.

Quiero:
- split temporal correcto
- validación walk-forward o rolling
- métricas de clasificación
- métricas económicas
- comparación contra baseline / buy & hold
- análisis de robustez

BACKTEST
El backtest debe mirar dinero, no solo accuracy.

Debe incluir:
- señales históricas
- posiciones
- cambios de posición
- coste de trading
- retorno acumulado
- comparación con buy & hold
- soporte para LONG / FLAT / SHORT
- reportes exportables

REENTRENAMIENTO
Quiero dejar la base preparada para reentrenamiento periódico.
No necesariamente full online learning desde el día 1, pero sí:
- reentrenamiento batch programado
- registro de versiones de modelo
- comparación de modelos
- posibilidad de reemplazar el modelo activo si el nuevo supera criterios mínimos

CRITERIOS DE CALIDAD DEL MODELO
No aceptar un modelo simplemente porque “entrena”.
El sistema debe facilitar filtrar modelos malos.
Diseña la lógica para que en el futuro se puedan definir criterios como:
- profit factor mínimo
- drawdown máximo
- Sharpe/Sortino mínimo
- estabilidad por ventana temporal
- consistencia entre train/validation/test
- número mínimo de trades útiles
- edge neto tras costes

TIEMPO REAL
No hace falta dejarlo perfecto ya, pero sí la arquitectura lista.
Debe poder evolucionar hacia:
- escuchar WebSocket de Binance
- escribir nuevas velas a la BD
- recalcular features recientes
- generar señal actualizada
- ejecutar órdenes automáticamente

EJECUCIÓN REAL
Debe quedar preparado para Binance Spot.
No derivados por ahora.
No margin por ahora.
No futures por ahora.

La lógica futura debe permitir:
- tamaño fijo o por riesgo
- órdenes market/limit
- validación previa de balance
- registro de órdenes
- registro de fills
- gestión de posiciones
- kill switch / modo dry run

SEGURIDAD
No ejecutar órdenes reales por defecto.
Todo debe quedar con:
- DRY_RUN = True
- ENABLE_TRADING = False
hasta que se active explícitamente.

ENTREGABLES QUE QUIERO DEJAR EN ESTA FASE
Quiero que dejes operativo:
1. config.py bien estructurado
2. download_data.py funcional con Binance
3. data_loader.py funcional con SQLite y prompt para revisar gaps de símbolos recién cargados
4. data_check.py funcional preguntando por símbolo(s)
5. data_gap_fill.py funcional leyendo data_gaps
6. features.py funcional
7. labels.py funcional
8. feature_store.py funcional
9. train.py leyendo desde la BD
10. backtest.py leyendo desde la BD
11. predict.py leyendo desde la BD
12. requirements.txt correcto
13. .gitignore correcto
14. README.md útil y orientado al flujo real

PRIORIDAD DE EJECUCIÓN
Prioridad 1:
- pipeline de datos sólido
- BD limpia
- gap detection / fill
- feature store

Prioridad 2:
- entrenamiento y validación desde BD
- backtest económico
- señal actual

Prioridad 3:
- preparación seria para:
  - realtime
  - reentrenamiento
  - ejecución real
  - Git/GitHub workflow

ESTILO DE IMPLEMENTACIÓN
- Código claro, modular y mantenible
- Comentarios donde aporten valor
- No sobreingeniería absurda
- No refactorizaciones cosméticas sin impacto
- No dejar piezas críticas a medias si pueden quedar operativas ya


EJECUCIÓN REAL EN BROKER / EXCHANGE

El sistema debe quedar preparado para operar con dinero real en Binance usando su API completa de Spot Trading.

No quiero una maqueta desconectada de la operativa. Quiero una arquitectura que pueda evolucionar a live trading real.

Implementa o deja preparada la siguiente capa:

- broker_client.py
  Wrapper limpio de Binance API con soporte para:
  - market data
  - account info
  - balances
  - exchange info / filters
  - order placement
  - order status
  - open orders
  - cancel order
  - trade/fill history

- execution_engine.py
  Debe:
  - recibir señal operativa
  - pedir validación al risk_manager
  - traducir la señal a una orden real
  - enviar la orden al broker_client
  - guardar la orden y su estado en BD

- risk_manager.py
  Debe validar:
  - balance disponible
  - tamaño mínimo permitido por Binance
  - step size / lot size
  - min notional
  - exposición máxima por activo
  - pérdida máxima diaria
  - número máximo de operaciones
  - si el sistema está en DRY_RUN o LIVE

- portfolio_manager.py
  Debe llevar:
  - posiciones spot actuales
  - cash / USDT disponible
  - exposición por símbolo
  - PnL realizado y no realizado
  - estado sincronizado con Binance

- trading_bot.py
  Debe ser el orquestador real:
  - leer nuevos datos
  - actualizar features
  - cargar modelo activo
  - generar señal
  - pasar por risk_manager
  - ejecutar por execution_engine
  - registrar todo en logs y BD

MODO DE SEGURIDAD
Por defecto:
- ENABLE_TRADING = False
- DRY_RUN = True

El sistema NO debe enviar órdenes reales salvo activación explícita.

OBJETIVO DE PRODUCCIÓN
La arquitectura debe permitir pasar de:
- local research bot
a
- live trading bot en Binance Spot
sin rehacer el proyecto.



IMPORTANTE FINAL
No te limites a “montar scripts”.
Construye la base de un SISTEMA AUTÓNOMO DE TRADING.
Debe estar diseñado para evolucionar desde:
- local research bot
a
- bot autónomo desplegable en servidor
con:
- data ingestion
- model training
- model validation
- realtime inference
- execution

Trabaja directamente sobre la carpeta, implementa, corrige incoherencias, y deja el proyecto en un estado funcional y serio.