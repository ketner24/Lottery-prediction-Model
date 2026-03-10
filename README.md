# 📊 Lottery Statistical Auditor & AI Predictor

A Streamlit-powered tool that downloads historical lottery data, runs rigorous statistical fairness tests, and trains an XGBoost machine learning model to generate number suggestions — with **honest backtest reporting** so you can see exactly how (in)effective prediction really is.

> **⚠️ Disclaimer:** Lottery draws are independent random events by design. No statistical model can reliably predict future outcomes. This tool is built for **educational and analytical purposes** — to explore probability, hypothesis testing, and ML classification. Please gamble responsibly.

---

## Supported Lotteries

| Lottery | Data Source | Numbers | Bonus Ball | Matrix Since |
|---------|------------|---------|------------|-------------|
| **Mega Millions** | data.ny.gov (API) | 5 of 70 | 1 of 25 | 2017-10-28 |
| **Powerball** | data.ny.gov (API) | 5 of 69 | 1 of 26 | 2015-10-07 |
| **SuperLotto Plus** | lottery.net (scraped) | 5 of 47 | 1 of 27 | 2000-06-03 |

---

## Features

### 📥 Built-In Data Scraper
- One-click download for Powerball and Mega Millions from official NY Open Data APIs
- Playwright-based scraper for SuperLotto Plus (full history since 1986)
- Shows last-download timestamps so you know how fresh your data is
- HTTP response caching to avoid hammering servers on repeated runs

### ⚖️ Fairness Hypothesis Testing
- **Chi-square goodness-of-fit** tests for both main numbers and bonus balls
- Warns when expected cell counts are too low for reliable chi-square results
- Interactive **frequency distribution bar charts** with expected-value reference lines

### 🔍 Single-Number Deep Dive
- **Wilson score confidence intervals** for any number's appearance rate
- **Geometric expected wait time** vs. actual average gap between appearances
- **Poisson model** for yearly appearance probability

### 🔥 Hot & Cold Analysis
- Adjustable look-back window (10–200 draws)
- Visual comparison of most and least frequently drawn numbers

### 🤖 XGBoost ML Predictor
- Time-series feature engineering: gap since last seen, rolling 3-draw and 10-draw frequency
- **Honest 80/20 backtest split** reporting accuracy, precision, and random baseline
- Generates 1–5 weighted ticket suggestions
- Top-15 probability table with gradient highlighting

---

## How Predictive Is It Really?

**Short answer: not very, and that's the point.**

Lottery machines are engineered to produce independent, uniformly distributed outcomes. The XGBoost model looks for short-term patterns in gap lengths and rolling frequencies. In backtesting, you'll typically see:

- **Overall accuracy ~93%** — sounds impressive, but this is because ~93% of number slots are 0 (not drawn), so predicting "not drawn" for everything gets you there.
- **Precision ~5–8%** — the model's "yes, this number will appear" predictions hit at roughly the same rate as random selection (5/69 ≈ 7.2% for Powerball).
- **Conclusion:** The model finds no exploitable edge. The draws are fair.

This is actually a valuable result — it **confirms the lottery is working as designed**. The statistical auditing features (chi-square, CI, Poisson) are the genuinely useful parts of this tool.

---

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/lottery-auditor.git
cd lottery-auditor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Lottery_Tool.py
```

### SuperLotto Plus Support (Optional)

SuperLotto Plus data is scraped from lottery.net using Playwright. If you want this lottery:

```bash
pip install playwright
playwright install chromium
```

Powerball and Mega Millions work without Playwright (they use direct API downloads).

### Docker (Alternative)

```bash
docker build -t lottery-auditor .
docker run -p 8501:8501 lottery-auditor
```

Then open http://localhost:8501

---

## Usage

1. **Launch the app** with `streamlit run Lottery_Tool.py`
2. **Download data** using the 📥 Data Manager in the left sidebar
3. **Select a lottery** from the dropdown
4. **Explore the tabs:**
   - ⚖️ Fairness Tests — are the draws statistically uniform?
   - 🔍 Number Deep Dive — drill into any single number
   - 🔥 Hot / Cold — recent frequency leaders and laggards
   - 🤖 AI Predictor — run the ML model and see backtest results

---

## Project Structure

```
lottery-auditor/
├── Lottery_Tool.py          # Main Streamlit application (all-in-one)
├── requirements.txt         # Python dependencies
├── requirements-full.txt    # Including Playwright for SuperLotto
├── Dockerfile               # Container deployment
├── .dockerignore
├── .gitignore
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── tests/
│   └── test_features.py     # Unit tests for feature engineering
├── data/                    # CSV files will be saved here (gitignored)
└── README.md
```

---

## Technical Details

### Feature Engineering

For each number N at each historical draw index T, the model computes:

| Feature | Description |
|---------|-------------|
| `Number` | The ball number itself (1–70) |
| `Gap` | Draws since N was last seen |
| `Freq_Last_10` | Times N appeared in the previous 10 draws |
| `Freq_Last_3` | Times N appeared in the previous 3 draws |

**Target:** Binary — did number N appear in draw T+1?

### Model Configuration

- XGBoost binary classifier
- 100 estimators, max depth 4, learning rate 0.05
- Log-loss evaluation metric
- 80/20 temporal split for backtesting (no data leakage)

### Statistical Tests

- **Chi-square (χ²):** Tests whether observed frequencies deviate significantly from uniform expectation. Warns when expected counts < 5.
- **Wilson score CI:** Confidence interval for a single number's draw rate. More accurate than Wald intervals for small proportions.
- **Poisson model:** Expected appearances per year assuming independence.
- **Geometric wait time:** Theoretical vs. actual gap between appearances.

---

## Contributing

Contributions are welcome! Some ideas:

- Add more lotteries (EuroMillions, UK Lotto, etc.)
- Implement additional ML models (LSTM, random forest) for comparison
- Add consecutive/sequential number pattern analysis
- Build a historical "what if" simulator (how would your numbers have done?)
- Improve Mega Ball prediction with its own XGBoost sub-model

---

## License

MIT License — see [LICENSE](LICENSE) for details.
