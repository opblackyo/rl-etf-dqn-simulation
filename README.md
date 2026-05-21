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
│   ├── run_seed_experiment.py
│   ├── run_reward_experiment.py
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
python -m src.run_seed_experiment
python -m src.run_reward_experiment
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
reports/seed_experiment_summary.csv
reports/reward_experiment_summary.csv
```

Streamlit Dashboard 會展示：

- 0050 價格走勢
- DQN 交易訊號
- DQN、Buy and Hold、Random Policy 淨值曲線
- DQN 訓練 reward 曲線
- 策略績效表格
- 多 seed 實驗表格
- reward function 比較表格
- 專題免責聲明

## 研究完整度實驗

本專案新增兩個研究完整度實驗，目的是觀察模型訓練結果的穩定性與 reward function 設計的影響，不是為了調整出更好的投資績效。

多 seed 實驗：

```bash
python -m src.run_seed_experiment
```

此實驗使用相同 80% train / 20% test 時間序列切分，分別以 `42`、`123`、`999` 三個 random seed 訓練 DQN，並將測試集回測結果輸出到 `reports/seed_experiment_summary.csv`。這可以用來觀察 DQN 在不同隨機初始化與探索過程下，結果是否穩定。

reward function 比較：

```bash
python -m src.run_reward_experiment
```

此實驗比較 `equity_change` 與 `cost_penalty` 兩種 reward mode。`equity_change` 使用前後資產淨值變化率，`cost_penalty` 則在發生 Buy / Sell 時額外加入小懲罰。結果會輸出到 `reports/reward_experiment_summary.csv`，可用來觀察 reward function 對交易行為與回測結果的影響。

以上實驗仍然只使用歷史資料進行模擬，不涉及真實下單、券商 API 或即時行情，也不構成投資建議。

## 學習重點

- 使用 yfinance 取得 ETF 歷史資料並整理 OHLCV 欄位。
- 以 pandas 建立報酬率、均線、波動度與成交量變化等特徵。
- 依照 Gymnasium Env API 實作自訂交易模擬環境。
- 使用 Stable-Baselines3 DQN 訓練離散動作策略。
- 建立 Buy and Hold 與 Random Policy baseline，避免只看單一模型結果。
- 透過多 seed 實驗觀察 DQN 訓練結果是否穩定。
- 透過 reward function 比較理解 reward 設計會影響 agent 行為。
- 使用 Streamlit 與 Plotly 製作互動式成果展示頁面。
- 從回測結果反思模型限制，而不是宣稱模型可以穩定獲利。

## 已知限制

- 0050 日資料量有限，對強化學習而言樣本數偏少。
- 此專案只做歷史模擬，不代表未來績效。
- 交易成本、滑價與成交限制皆為簡化模型。
- DQN 可能無法打敗 Buy and Hold，這仍然是合理的研究結果。
- Random Policy 使用固定 seed 以便重現，但仍不代表實際交易行為。
- 回測沒有做 walk-forward validation，也沒有針對不同市場狀態做嚴格穩健性測試。
