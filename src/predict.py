import argparse
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from config import DB_FILE, FEATURES_TABLE, SYMBOLS, TIMEFRAME, SIGNALS_TABLE
from db_utils import init_research_tables, ensure_project_directories
from model_registry import get_latest_model
from modeling_utils import POSITION_TO_NAME, probabilities_to_signal


def parse_args():
    parser = argparse.ArgumentParser(description="Generate current signal from latest feature rows.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model artifact (.joblib)")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    return parser.parse_args()


def resolve_model_path(model_path_arg: str | None, timeframe: str) -> Path:
    if model_path_arg:
        path = Path(model_path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        return path

    latest = get_latest_model(timeframe=timeframe)
    if latest and Path(latest["model_path"]).exists():
        return Path(latest["model_path"])

    candidates = sorted(Path("models").glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained model found in registry or models directory.")
    return candidates[0]


def load_latest_feature_row(symbol: str, timeframe: str, feature_columns: list[str]) -> pd.DataFrame:
    not_null_cols = " AND ".join([f"{col} IS NOT NULL" for col in feature_columns])

    conn = sqlite3.connect(DB_FILE)
    try:
        query = f"""
        SELECT symbol, timeframe, datetime_utc, {", ".join(feature_columns)}
        FROM {FEATURES_TABLE}
        WHERE symbol = ? AND timeframe = ? AND {not_null_cols}
        ORDER BY datetime_utc DESC
        LIMIT 1
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    finally:
        conn.close()

    if df.empty:
        return df

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime_utc"] + feature_columns).copy()
    return df


def save_signal(
    model_id: str,
    symbol: str,
    timeframe: str,
    datetime_utc: pd.Timestamp,
    signal_position: int,
    prob_short: float,
    prob_flat: float,
    prob_long: float,
) -> None:
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {SIGNALS_TABLE} (
                model_id,
                symbol,
                timeframe,
                datetime_utc,
                signal_position,
                signal_label,
                prob_short,
                prob_flat,
                prob_long,
                created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                symbol,
                timeframe,
                datetime_utc.isoformat(),
                int(signal_position),
                POSITION_TO_NAME[int(signal_position)],
                float(prob_short),
                float(prob_flat),
                float(prob_long),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def main():
    args = parse_args()
    ensure_project_directories()
    init_research_tables()

    model_path = resolve_model_path(args.model_path, args.timeframe)
    artifact = joblib.load(model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    short_threshold = float(artifact.get("short_threshold", 0.55))
    long_threshold = float(artifact.get("long_threshold", 0.55))
    symbol_mapping = artifact["symbol_mapping"]
    model_id = str(artifact.get("model_id", model_path.stem))

    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    rows = []

    for symbol in symbols:
        if symbol not in symbol_mapping:
            print(f"[{symbol}] skipped: symbol not present in model mapping.")
            continue

        latest = load_latest_feature_row(symbol, args.timeframe, feature_columns=artifact["base_feature_columns"])
        if latest.empty:
            print(f"[{symbol}] skipped: no latest complete feature row.")
            continue

        latest = latest.copy()
        latest["symbol_code"] = symbol_mapping[symbol]
        X = latest[feature_columns]

        probas = model.predict_proba(X)
        signal_position = probabilities_to_signal(
            probas=probas,
            short_threshold=short_threshold,
            long_threshold=long_threshold,
        )[0]

        dt = latest.iloc[0]["datetime_utc"]
        p_short = float(probas[0, 0])
        p_flat = float(probas[0, 1])
        p_long = float(probas[0, 2])

        save_signal(
            model_id=model_id,
            symbol=symbol,
            timeframe=args.timeframe,
            datetime_utc=dt,
            signal_position=int(signal_position),
            prob_short=p_short,
            prob_flat=p_flat,
            prob_long=p_long,
        )

        rows.append(
            {
                "symbol": symbol,
                "timeframe": args.timeframe,
                "datetime_utc": dt.isoformat(),
                "signal_position": int(signal_position),
                "signal_label": POSITION_TO_NAME[int(signal_position)],
                "prob_short": p_short,
                "prob_flat": p_flat,
                "prob_long": p_long,
            }
        )

    if not rows:
        print("No signals generated.")
        return

    out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    print(f"Signals generated with model_id={model_id}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
