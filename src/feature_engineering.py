"""Create technical features for the ETF trading simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "0050_daily.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "0050_features.csv"


def build_features(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Read raw OHLCV data and write a feature-enriched CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"找不到資料檔：{input_path}，請先執行 python -m src.data_loader")

    df = pd.read_csv(input_path, parse_dates=["Date"])
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"資料缺少必要欄位：{missing}")

    df = df.sort_values("Date").reset_index(drop=True)
    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["ma_ratio"] = (df["ma_5"] / df["ma_20"]) - 1.0
    df["volatility_20"] = df["return_1d"].rolling(window=20).std()
    df["volume_change"] = df["Volume"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    try:
        df = build_features()
    except Exception as exc:
        print(f"特徵工程失敗：{exc}")
        raise SystemExit(1) from exc

    print(f"已儲存 {len(df)} 筆特徵資料到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
