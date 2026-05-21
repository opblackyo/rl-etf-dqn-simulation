"""Download historical 0050 ETF daily data for offline simulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "0050_daily.csv"

TICKER = "0050.TW"
START_DATE = "2003-01-01"


def download_0050_daily(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Download 0050.TW daily OHLCV data and save it as a CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        TICKER,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("yfinance returned an empty dataset for 0050.TW.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing required columns: {missing}")

    data = data[required_columns].dropna()
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    return data


def main() -> None:
    try:
        data = download_0050_daily()
    except Exception as exc:
        print("無法透過 yfinance 下載 0050.TW 歷史資料。")
        print(f"錯誤訊息：{exc}")
        print(f"你可以手動放入 CSV 檔案：{OUTPUT_PATH}")
        print("欄位至少需包含：Date, Open, High, Low, Close, Volume")
        raise SystemExit(1) from exc

    print(f"已儲存 {len(data)} 筆資料到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
