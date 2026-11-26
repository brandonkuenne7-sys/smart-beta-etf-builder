# Project Overview: Smart Beta ETF Builder
### (Context document — written for handoff to another AI assistant/model)

## 1. What This Project Is

A Python research engine + interactive web dashboard that constructs and backtests six
rules-based "smart beta" equity portfolios (Equal Weight, Momentum, Minimum Volatility, Value,
Quality, Multi-Factor, plus an optimizer-based Minimum Variance portfolio) and benchmarks them
against SPY (S&P 500 ETF proxy). It's a portfolio/resume piece for quantitative finance,
demonstrating factor investing, backtesting, portfolio optimization, and Fama-French factor
attribution — deployed as a live, public, interactive web app.

**Live URL:** https://brandonkuenne7.pythonanywhere.com/

## 2. File Inventory & Roles

```
smart_beta_etf_builder.py   Core engine — all classes/functions, ~920 lines
dash_app.py                 Interactive Plotly Dash web dashboard, ~450 lines
precompute_cache.py         Run locally to pull live data & generate pipeline_cache.json
pipeline_cache.json         Generated data snapshot (NOT in the code repo/zip — user-generated,
                             uploaded separately to the host)
requirements.txt            pip dependencies
Procfile                    `web: gunicorn dash_app:server` (for Render-style hosts)
.gitignore
README.md                   Full project spec/writeup (15-section format: overview, use case,
                             architecture, APIs, libraries, folder structure, build guide, etc.)
DEPLOY.md                   Render.com deployment guide (live-data path)
DEPLOY_PYTHONANYWHERE.md    PythonAnywhere deployment guide (cached-data path, free tier)
.github/workflows/
  refresh_data.yml          Optional GitHub Actions workflow: reruns precompute_cache.py
                             weekly and auto-commits the refreshed pipeline_cache.json
                             (not yet wired up to auto-deploy on PythonAnywhere — see §7)
```

## 3. Core Engine Architecture (`smart_beta_etf_builder.py`)

Structured as Colab-cell-style sections (`# %% CELL N` markers) but written as a proper
importable module. **Important structural detail:** all class/function *definitions* live at
module level (safe to import elsewhere with no side effects), but the actual pipeline
*execution* (running the backtest, generating matplotlib charts, printing results) is wrapped
inside `def main(): ...` guarded by `if __name__ == "__main__":`. This was a deliberate
refactor so `dash_app.py` can `import` this module's classes/functions without triggering a
full re-run of the backtest and without popping matplotlib windows.

**Pipeline stages (each a class):**

1. `Config` (dataclass) — central settings: stock universe (fixed list of ~30 large-cap
   tickers), benchmark ticker (SPY), date range (2015-01-01 to 2026-01-01), rebalance
   frequency (monthly), position caps, transaction cost (10 bps), fundamentals caching flags.
2. `DataPipeline` — pulls prices (`yfinance.download`, batched) and fundamentals
   (`yfinance.Ticker(t).info`, per-ticker, best-effort with try/except). Fundamentals are
   cached to `/tmp/fundamentals_snapshot.csv` by default for run-to-run reproducibility (since
   they're a live snapshot, not point-in-time — re-pulling on a different day could shift
   Value/Quality rankings otherwise).
3. `DataCleaner` — drops tickers with excessive missing data, forward-fills short gaps only
   (max 5 days), drops any remaining NaNs.
4. `FactorEngine` — computes z-scored factor signals:
   - Momentum = 12-month return, skipping the most recent month (12-1 momentum)
   - Low-Vol = inverse trailing 12-month realized volatility
   - Value = average(z-score(1/P·E), z-score(1/P·B))
   - Quality = average(z-score(ROE), z-score(profit margin), z-score(-debt/equity))
   - Multi-Factor = mean of the above four z-scores
5. `PortfolioConstructor`:
   - `equal_weight()` — flat 1/N
   - `factor_weighted(scores)` — selects top 50% by score, weights proportional to
     (shifted-positive) score
   - `min_variance(returns)` — `scipy.optimize.minimize` (SLSQP), minimizes `wᵀΣw` subject to
     `Σw=1`, `0≤w≤10%` per position; falls back to equal-weight if the optimizer fails to
     converge
6. `Backtester` — simulates monthly rebalancing day-by-day, applies transaction cost drag
   (bps × turnover) at each rebalance, returns equity curve + weight history + turnover
   history per strategy
7. `PerformanceAnalytics` — CAGR, annualized volatility, Sharpe, Sortino, Max Drawdown,
   Calmar, Information Ratio (vs. SPY), rolling Sharpe, drawdown series
8. `FamaFrenchAnalytics` — pulls FF5 factors (Mkt-RF, SMB, HML, RMW, CMA) via
   `pandas_datareader` from Kenneth French's data library, regresses each strategy's monthly
   excess return via `statsmodels.OLS`, reports annualized alpha, factor betas, t-stats, R²

**Module-level orchestration functions:**
- `run_pipeline(cfg)` → returns `{"results": {strategy_name: {"equity_curve", "weight_history",
  "turnover_history"}}, "benchmark_curve", "universe_prices", "fundamentals"}`
- `generate_investment_memo(report_df)` → auto-populated text memo (strategy overview,
  headline Sharpe winner, risks/limitations, future improvements)
- `generate_education_content()` → markdown explainer covering what the dashboard does, data
  sources, why this specific stock universe, **the exact formula/inputs behind each strategy's
  weighting**, how the backtest mechanics work, a metric glossary, and an honest note on data
  freshness

## 4. Dashboard (`dash_app.py`)

Built with Plotly **Dash** (not Streamlit). Key architectural choice: it does **not** call
`run_pipeline()` on every page load. Instead, at module load time it checks for a
`pipeline_cache.json` file next to it:
- **If present** (production/PythonAnywhere path): loads and deserializes it into the same
  pandas Series/DataFrame shapes `run_pipeline()` would have produced.
- **If absent** (local dev / Colab): calls `run_pipeline()` live.

**Why JSON, not pickle:** an earlier version cached via `pickle`, which broke in production
with `ModuleNotFoundError: No module named 'numpy._core'` because the pickle was created with a
different numpy major version (2.x) than the one installed on the server (1.x) — pickle
embeds numpy's internal object layout. Switched to plain JSON (`[date_string, float]` pairs for
Series, nested dicts for weight histories) specifically because it has zero coupling to
numpy/pandas internal versions.

**Tabs in the deployed app:**
1. **How It Works** — renders `generate_education_content()` via `dcc.Markdown`
2. **Strategy Explorer** — interactive: `dcc.RadioItems` to pick a strategy, callback renders
   (a) a monospace "formula chip" + plain-language description from a `STRATEGY_EXPLANATIONS`
   dict, and (b) a live horizontal bar chart of that strategy's **actual current top-10
   holdings**, pulled from the real `weight_history` data (not mocked)
3. **Equity Curve**, **Drawdown**, **Rolling Sharpe** — line charts, togglable per-strategy via
   a checklist that filters all three simultaneously
4. **Allocation** — dropdown-selected strategy → pie chart of current holdings
5. **Correlation Matrix** — heatmap of strategy return correlations
6. Below the tabs: sortable **Risk/Performance DataTable** and **Fama-French Attribution
   DataTable**, both with **hover tooltips on column headers** (`METRIC_DEFINITIONS` /
   `FACTOR_DEFINITIONS` dicts feeding a `tooltip_header` helper) explaining each stat in plain
   English
7. **Investment Memo** — rendered as preformatted text

`server = app.server` is exposed at module level so a WSGI server (gunicorn on Render, or
PythonAnywhere's WSGI config) can serve the underlying Flask app directly.

## 5. Deployment (current state: LIVE on PythonAnywhere free tier)

**Why the cached-data architecture exists at all:** PythonAnywhere's free tier restricts
outbound internet to a domain whitelist that does NOT include Yahoo Finance or Kenneth
French's data library. So the live site cannot call `yfinance`/`pandas_datareader` itself —
all data pulling happens offline (`precompute_cache.py`, run by the user locally, with full
internet access), producing `pipeline_cache.json`, which is then manually uploaded.

**Deployment mechanics on PythonAnywhere (no git involved — manual file upload):**
- Files live at `/home/brandonkuenne7/mysite/`
- A virtualenv `mysite-venv` was created via `mkvirtualenv --python=/usr/bin/python3.10`, with
  `requirements.txt` installed into it
- The **Web tab** manual-configuration app points: Source code & Working directory →
  `/home/brandonkuenne7/mysite`; Virtualenv → `/home/brandonkuenne7/.virtualenvs/mysite-venv`
- The **WSGI config file** (`/var/www/brandonkuenne7_pythonanywhere_com_wsgi.py`) is a short
  script that adds the project dir to `sys.path` and does `from dash_app import server as
  application`
- Updates are currently done by manually opening files in PythonAnywhere's browser-based code
  editor, pasting new content, saving, and clicking "Reload" on the Web tab

**Render.com path also exists** (`DEPLOY.md`) as an alternative that supports either cached or
*live* data (Render's free tier has no outbound whitelist restriction), using
`gunicorn dash_app:server` per the `Procfile`. Not currently in use — PythonAnywhere is live.

## 6. Data Refresh Cadence (currently MANUAL)

There is no automatic refresh in production right now. The cycle to update numbers is:
1. Locally: `python3 precompute_cache.py` (needs full internet — pulls yfinance + Fama-French
   data) → regenerates `pipeline_cache.json`
2. Manually upload that file to PythonAnywhere, overwriting the old one
3. Click "Reload" on the Web tab

The dashboard header displays "Data last refreshed: [timestamp]" (stamped into the JSON cache
at generation time) so visitors can see snapshot age.

## 7. Known Gaps / Possible Next Steps (not yet built)

- `.github/workflows/refresh_data.yml` exists to auto-run `precompute_cache.py` weekly and
  commit the refreshed JSON back to a GitHub repo — but **the project isn't on GitHub yet**,
  and even once it is, PythonAnywhere won't auto-pull from GitHub without either a paid-tier
  Scheduled Task running `git pull`, or switching hosts to Render (which auto-redeploys on
  push). This is flagged to the user as an open TODO, not done.
- Fundamentals are a live snapshot (not point-in-time), introducing look-ahead bias — flagged
  explicitly in both the Investment Memo and education content as a known limitation.
- Universe (~30 large-caps) is not survivorship-bias-free (no historical index reconstruction).
- No sector/position-size constraints beyond the flat 10% cap in the optimizer.

## 8. Notable Bugs Already Fixed During Development (for context on fragility points)

1. `generate_investment_memo` was accidentally nested inside `main()` during an earlier
   refactor, making it unimportable from `dash_app.py` → moved to module level.
2. A stray typo (`from smart_beta_engine import` instead of `from smart_beta_etf_builder
   import`) in an intermediate version of `dash_app.py`.
3. The pickle→numpy version mismatch described in §4.
4. A `str_replace` edit once accidentally deleted the `def main():` line itself while inserting
   a new function above it — caught immediately via `grep` + recompile, not left in.

## 9. Tech Stack Summary

Python 3.10, pandas, numpy, scipy (SLSQP optimizer), matplotlib/seaborn (for the
Colab/console-run charts), Plotly + Dash (web dashboard), yfinance (price/fundamentals data),
pandas-datareader + statsmodels (Fama-French factor regression), gunicorn (WSGI server for
production hosting).
