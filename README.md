# 強化學習於台股 ETF 交易決策模擬與視覺化系統

本專案以台股 0050 ETF 歷史日資料為研究資料，建立一個簡化的交易模擬環境，並使用 Stable-Baselines3 的 DQN agent 模擬 Buy / Sell / Hold 決策。成果重點是資料處理、DQN 訓練流程、baseline 比較、Streamlit 視覺化與學習反思。

## 免責聲明

本專案僅作為強化學習與金融資料分析的自發學習成果，不涉及真實下單，也不構成任何投資建議。專案不串接券商 API，也不包含任何真實交易功能。模型回測結果只代表歷史資料上的模擬表現，不代表未來績效。

## 專案架構

```text
rl-etf-dqn-simulation/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── trading_env.py
│   ├── train_dqn.py
│   ├── backtest.py
│   └── metrics.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 安裝方式

建議使用 Python 3.10 以上版本，並在虛擬環境中安裝套件。

```bash
pip install -r requirements.txt
```

## 執行流程

請依序執行：

```bash
pip install -r requirements.txt
python -m src.data_loader
python -m src.feature_engineering
python -m src.train_dqn
python -m src.backtest
streamlit run app.py
```

如果 yfinance 暫時無法下載資料，可以手動放入 `data/0050_daily.csv`，欄位至少需包含：

```text
Date, Open, High, Low, Close, Volume
```

## 輸出成果

執行完整流程後，會產生以下檔案：

```text
data/0050_daily.csv
data/0050_features.csv
models/dqn_0050.zip
reports/training_rewards.csv
reports/dqn_equity_curve.csv
reports/buy_hold_equity_curve.csv
reports/random_equity_curve.csv
reports/trades.csv
reports/performance_summary.csv
```

Streamlit Dashboard 會展示：

- 0050 價格走勢
- DQN 交易訊號
- DQN、Buy and Hold、Random Policy 淨值曲線
- DQN 訓練 reward 曲線
- 策略績效表格
- 專題免責聲明

## 學習重點

- 使用 yfinance 取得 ETF 歷史資料並整理 OHLCV 欄位。
- 以 pandas 建立報酬率、均線、波動度與成交量變化等特徵。
- 依照 Gymnasium Env API 實作自訂交易模擬環境。
- 使用 Stable-Baselines3 DQN 訓練離散動作策略。
- 建立 Buy and Hold 與 Random Policy baseline，避免只看單一模型結果。
- 使用 Streamlit 與 Plotly 製作互動式成果展示頁面。
- 從回測結果反思模型限制，而不是宣稱模型可以穩定獲利。

## 已知限制

- 0050 日資料量有限，對強化學習而言樣本數偏少。
- 此專案只做歷史模擬，不代表未來績效。
- 交易成本、滑價與成交限制皆為簡化模型。
- DQN 可能無法打敗 Buy and Hold，這仍然是合理的研究結果。
- Random Policy 使用固定 seed 以便重現，但仍不代表實際交易行為。
- 回測沒有做 walk-forward validation，也沒有針對不同市場狀態做嚴格穩健性測試。
