# Smart Beta ETF Builder
### Designing, Backtesting, and Reporting on an Institutional-Grade Factor ETF

**Live dashboard:** https://brandonkuenne7.pythonanywhere.com/ — data refreshes automatically each month.

---

## 1. Project Overview

The Smart Beta ETF Builder is a self-contained Python research engine that constructs and
backtests multiple rules-based ("smart beta") equity portfolios — **Equal Weight, Minimum
Volatility, Momentum, Quality, Value, and Multi-Factor** — and benchmarks them against a
market-cap-weighted proxy (SPY).

The system is built as a set of composable classes (`DataPipeline`, `FactorEngine`,
`PortfolioConstructor`, `Backtester`, `PerformanceAnalytics`) so that any one piece — the
universe, the factors, the weighting scheme, or the rebalancing rules — can be swapped out
independently. The output is a full research stack: cleaned data → factor scores → portfolio
weights → historical simulation → risk/return report → charts suitable for a GitHub README or an
investment memo.

---

## 2. Real-World Finance Use Case

Smart beta is a multi-trillion-dollar segment of institutional asset management that sits between
passive indexing and active stock-picking:

- **BlackRock (iShares)** runs factor ETFs for Minimum Volatility, Value, and Quality exposures.
- **AQR Capital Management** built its entire research program around Value, Momentum, Quality,
  and Defensive (low-vol) factors.
- **Goldman Sachs Asset Management, JPMorgan, Morgan Stanley** embed factor models in equity
  research, wealth management, and asset allocation.
- **Quant hedge funds** (Two Sigma, Point72, D.E. Shaw, Citadel, Renaissance) use factor signals as
  inputs into larger statistical/ML return models.

This project reproduces the core research workflow those teams use: turn raw price/fundamental
data into standardized factor scores, translate those scores into portfolio weights subject to
constraints, and rigorously backtest the result against a benchmark with transaction costs.

---

## 3. System Architecture

```
                ┌───────────────────┐
                │   Data Pipeline    │  yfinance prices + fundamentals
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │   Data Cleaning    │  missing data, outliers, alignment
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │   Factor Engine    │  Momentum / Value / Quality / LowVol
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │ Portfolio          │  EqualWeight / FactorWeighted /
                │ Constructor        │  MinVariance / MultiFactor
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │    Backtester      │  monthly rebalance, txn costs, turnover
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │ Performance /      │  CAGR, Sharpe, Sortino, Max DD, IR,
                │ Risk Analytics     │  rolling Sharpe, factor exposures
                └─────────┬─────────┘
                          │
                ┌─────────▼─────────┐
                │  Visualization /   │  equity curve, drawdown, allocation,
                │  Reporting Layer   │  correlation matrix, risk dashboard
                └───────────────────┘
```

Each stage consumes and returns a plain `pandas` object (Series/DataFrame), so the pipeline can
be paused, inspected, or re-run from any stage without recomputing earlier steps.

---

## 4. Required APIs and Data Sources

| Purpose | Source | Notes |
|---|---|---|
| Adjusted prices, volume | **Yahoo Finance** via `yfinance` | Free, no API key, used for the reference implementation |
| Fundamentals (P/E, P/B, ROE, D/E) | `yfinance` `.info` (best-effort) | Free tier is adequate for a prototype; swap in **Financial Modeling Prep** or **Alpha Vantage** for production-grade point-in-time fundamentals |
| Factor benchmarks | **Kenneth French Data Library** | Optional — used to validate constructed factors against academic Fama-French factors |
| Risk-free rate | **FRED** (`^IRX` via yfinance, or `fredapi`) | Used in Sharpe/Sortino calculations |
| SEC filings (stretch) | **SEC EDGAR** | For point-in-time fundamentals / bias removal |

> **Production note:** `yfinance` fundamentals are current-snapshot only (not point-in-time),
> which introduces look-ahead bias. Section 15 (Potential Upgrades) addresses this.

---

## 5. Required Python Libraries

```bash
pip install yfinance pandas numpy scipy matplotlib seaborn plotly quantstats statsmodels
```

| Library | Role |
|---|---|
| `pandas`, `numpy` | Data wrangling, numerical computation |
| `yfinance` | Price + fundamentals data |
| `scipy.optimize` | Minimum-variance portfolio optimization |
| `matplotlib`, `seaborn` | Static charts |
| `plotly` | Interactive charts (optional, Colab-friendly) |
| `quantstats` | Convenience performance/tearsheet metrics (optional) |
| `statsmodels` | Fama-French regression (optional, stretch) |

---

## 6. Folder / File Structure

Even though the reference build runs inside Google Colab, structure it like a real repo so it
drops cleanly into GitHub:

```
Smart-Beta-ETF-Builder/
│
├── README.md
├── Smart_Beta_ETF.ipynb          # Colab notebook (cells mirror the .py sections)
├── requirements.txt
│
├── data/
│   ├── raw/                      # cached price/fundamental pulls
│   └── processed/                # cleaned panels, factor scores
│
├── src/
│   ├── data_pipeline.py          # DataPipeline class
│   ├── factors.py                # FactorEngine class
│   ├── portfolio.py              # PortfolioConstructor class
│   ├── backtester.py             # Backtester class
│   └── analytics.py              # PerformanceAnalytics class
│
├── results/
│   ├── charts/                   # exported PNGs
│   └── performance_report.pdf
│
└── smart_beta_etf_builder.py     # single-file reference implementation (this deliverable)
```

---

## 7. Step-by-Step Build Guide

1. **Define the universe** — start with a liquid large-cap subset (demo uses ~30 tickers); scale
   to full S&P 500 once the pipeline is validated.
2. **Pull price history** (`Cell 2`) and **fundamentals** (`Cell 3`) via `yfinance`.
3. **Clean the panel** (`Cell 4`) — forward-fill small gaps, drop tickers with excessive missing
   data, align all series to a common trading calendar.
4. **Engineer factors** (`Cell 5`) — Momentum, Value, Quality, Low-Volatility, each converted to a
   cross-sectional z-score so factors are comparable and combinable.
5. **Construct portfolios** (`Cell 6`) — Equal Weight, single-factor tilts, Minimum-Variance
   (via constrained optimization), and a blended Multi-Factor portfolio.
6. **Backtest** (`Cell 7`) — monthly rebalancing loop with transaction-cost drag and turnover
   tracking.
7. **Compute performance metrics** (`Cell 8`) — CAGR, volatility, Sharpe, Sortino, max drawdown,
   information ratio vs. SPY.
8. **Visualize** (`Cell 9`) — equity curves, drawdown chart, rolling Sharpe, allocation pie,
   correlation matrix, risk dashboard.
9. **Write the investment memo** — strategy thesis, historical results, risks, limitations
   (template included at the end of the script's output).

---

## 8. Data Collection Pipeline

`DataPipeline.get_prices()` batch-downloads adjusted close prices for the full universe plus the
benchmark in a single `yfinance.download()` call (fast, avoids rate-limiting), then
`DataPipeline.get_fundamentals()` loops per-ticker over `.info` with try/except guards, since any
single ticker's fundamentals call can fail or return incomplete fields.

## 9. Data Cleaning & Feature Engineering

- Drop tickers with more than a configurable missing-data threshold (default 10%).
- Forward-fill short gaps only (max 5 trading days) to avoid manufacturing stale-price returns.
- Winsorize factor scores at the 1st/99th percentile before z-scoring to limit outlier influence.
- All factor scores are recomputed **on a rolling basis at each rebalance date**, using only data
  available as of that date, to avoid look-ahead bias in the backtest loop.

## 10. Core Models/Algorithms

- **Momentum**: 12-month return skipping the most recent month (standard 12-1 momentum).
- **Value**: composite of earnings yield (1/P·E) and 1/P·B.
- **Quality**: composite z-score of ROE, profit margin, and (inverse) debt-to-equity.
- **Low Volatility**: inverse of trailing 12-month realized volatility.
- **Multi-Factor**: equal-weighted blend of the four standardized factor scores.
- **Minimum Variance**: `scipy.optimize.minimize` (SLSQP) solving `min wᵀΣw` subject to
  `Σw = 1`, `0 ≤ w ≤ max_weight`.

## 11. Visualizations & Dashboard Components

Equity curve (all strategies vs. SPY), drawdown chart, rolling 6-month Sharpe, current portfolio
allocation (pie), factor exposure bar chart, correlation matrix heatmap, and a summary risk
dashboard table (CAGR / Vol / Sharpe / Sortino / Max DD / Turnover per strategy).

## 12. Performance Metrics

CAGR, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio,
information ratio vs. SPY, average annual turnover, and estimated cost drag.

## 13. Final Deliverables

- Cleaned price/fundamental panels
- Factor score history
- Six backtested portfolios with weight histories
- Performance tearsheet (table + charts)
- Investment memo template (auto-populated with computed metrics)

## 14. Resume Description

> Designed and backtested a multi-factor smart beta ETF engine in Python (Value, Momentum,
> Quality, Low-Volatility, and a blended Multi-Factor portfolio), including a constrained
> minimum-variance optimizer, monthly rebalancing backtester with transaction-cost modeling, and
> a full risk/performance reporting suite benchmarked against SPY (CAGR, Sharpe, Sortino, max
> drawdown, information ratio).

## 15. Potential Upgrades

1. Replace `yfinance` snapshot fundamentals with **point-in-time** data (FMP/EDGAR) to eliminate
   look-ahead bias.
2. Reconstruct historical index membership to remove survivorship bias.
3. Add sector/position-size constraints to the optimizer (`cvxpy` + `PyPortfolioOpt`).
4. Add a machine-learning factor (XGBoost/LightGBM) predicting forward returns from the same
   feature set.
5. Add regime-aware dynamic factor timing (e.g., tilt toward Quality/Low-Vol in high-VIX regimes).
6. Wrap the reporting layer in a Plotly Dash app for an interactive dashboard.
7. Add a Fama-French 3/5-factor regression to report each portfolio's factor loadings (α, β).

---

## 16. Interactive Dashboard (`dash_app.py`)

A Plotly Dash web app sits on top of the same engine — no logic is duplicated. It:

- Runs `run_pipeline()` **once** at startup and caches the result in memory.
- Lets you toggle which strategies are plotted, switch tabs between Equity Curve /
  Drawdown / Rolling Sharpe / Allocation / Correlation Matrix, and pick which strategy's
  current holdings to view as a pie chart.
- Renders the same risk/performance dashboard and Fama-French attribution table as
  sortable data tables, plus the auto-generated investment memo.

**Run locally:**
```bash
pip install dash plotly pandas-datareader statsmodels yfinance
python dash_app.py
# open http://127.0.0.1:8050
```

**Run in Colab:** `pip install dash jupyter-dash` then use `app.run(jupyter_mode="external")`
(or `"inline"`) instead of the bare `app.run(debug=True)` at the bottom of the file.

**Deploy it for a resume link** (all have free tiers): **Render**, **Railway**, **Fly.io**, or
**PythonAnywhere**. Push `dash_app.py`, `smart_beta_etf_builder.py`, and a `requirements.txt` to a
GitHub repo, connect the repo to the host, and set the start command to
`gunicorn dash_app:server`. You get a public URL you can drop straight into a resume, LinkedIn
project section, or a "Live Demo" badge on the GitHub README.

## 17. Fama-French Factor Attribution

`FamaFrenchAnalytics` (in `smart_beta_etf_builder.py`) pulls monthly factor returns from
Kenneth French's data library via `pandas_datareader` and regresses each strategy's monthly
excess return on the FF5 factors (Mkt-RF, SMB, HML, RMW, CMA):

```
R_p - RF = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + ε
```

It reports annualized alpha, each factor beta with its t-stat, and R². This tells you, for
example, whether your "Quality" portfolio's outperformance is genuine quality-factor exposure
(RMW loading) or is actually just a disguised bet on Momentum or Value. Requires internet access
to French's data library; both `smart_beta_etf_builder.py` and `dash_app.py` degrade gracefully
(print a warning, skip the table) if that connection isn't available.

## 18. Reproducibility Note

The stock **universe** is a fixed, hardcoded list, so it never changes between runs. What *can*
drift is which names get **selected and weighted**, because Value/Quality fundamentals are pulled
as a live current-day snapshot via `yfinance`. To keep results stable across reruns, fundamentals
are cached to `/tmp/fundamentals_snapshot.csv` by default (`Config.use_cached_fundamentals =
True`). Delete that file, or set the flag to `False`, whenever you want a fresh pull (e.g., to
show the portfolio drift over time as a feature rather than a bug).

---

## 19. Automated Monthly Refresh

The live site runs on PythonAnywhere's free tier, which can't reach Yahoo Finance. So the data is
rebuilt on GitHub's servers and pushed into PythonAnywhere by
`.github/workflows/refresh_data.yml`, on the 3rd of every month:

1. `precompute_cache.py` reruns the full pipeline through the last completed month, using the
   pinned stack in `requirements-refresh.txt`.
2. `validate_cache.py` checks the result before anything is published: all seven strategies
   present, no missing values, data is current, Sharpe/CAGR in a plausible range, latest
   weights sum to 100%, Fama-French attribution complete. If any check fails, the run stops and
   the live site keeps the previous month's data.
3. The new `pipeline_cache.json` is committed, along with a gzipped snapshot in `history/`.
   Those snapshots build a month-by-month, out-of-sample record of what each strategy held.
4. The file is uploaded through the PythonAnywhere API, the web app is reloaded, and the
   workflow confirms the live site is serving the new data.

**One-time setup:** create an API token (PythonAnywhere → Account → API Token) and add it as a
repository secret named `PA_API_TOKEN` (GitHub → Settings → Secrets and variables → Actions).

**Run it manually:** Actions tab → *Monthly data refresh* → *Run workflow*.

**Deploying code changes:** after changing `dash_app.py`, `assets/style.css` or the engine, run
*Deploy dashboard code* from the Actions tab. It smoke-tests the app against the server's Dash
versions, backs up the live files to `mysite/backups/<timestamp>/`, uploads, reloads, and checks the
site.

**Run it locally:** `pip install -r requirements-refresh.txt && python precompute_cache.py`
(optionally `--end-date YYYY-MM-DD`), then `python validate_cache.py pipeline_cache.json`.

---

## 20. Point-in-Time Universe (Hindsight-Bias Fix)

The original universe was a fixed list of 30 large caps chosen in 2026, which quietly bakes
hindsight into the backtest (it includes NVIDIA, Broadcom, Tesla, etc., because we now know they
won). The engine now defaults to `Config.universe_mode = "pit"`:

- **Membership:** `data/sp500_membership.csv.gz`, every S&P 500 composition change since 2012
  (from [fja05680/sp500](https://github.com/fja05680/sp500); refreshed by
  `scripts/update_membership.py` at the start of each monthly run).
- **Selection:** at each month-end, take that date's members with ~13 months of price history,
  rank by trailing 3-month average dollar volume, keep the top 50 (`Config.pit_top_n`).
- **Scoring:** every factor is computed cross-sectionally within that month's universe.
- **Comparison:** `precompute_cache.py` also runs the original fixed list
  (`universe_mode="fixed"`); the dashboard's *Hindsight-Bias Check* shows both.

Backtest timing was tightened at the same time: rebalances happen on the last **trading** day of
each month, new weights start earning the **next** day (no same-day look-ahead), and price history
before `start_date` is used only for factor lookbacks, so every strategy is invested from day one.

**Known remaining biases:** free price data is missing for some delisted/acquired members, and
fundamentals are a current snapshot rather than point-in-time.

---

*See `smart_beta_etf_builder.py` for the complete, runnable implementation (Colab-cell-structured)
and `dash_app.py` for the interactive dashboard.*
