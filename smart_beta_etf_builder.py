# =============================================================================
# SMART BETA ETF BUILDER
# Designing, Backtesting, and Reporting on an Institutional-Grade Factor ETF
#
# Structured as Google Colab cells. Each "# %% CELL N" marker below is meant
# to be pasted into its own Colab cell (Colab recognizes the "# %%" syntax as
# a cell divider if you use "Insert > Code cell" / paste-per-block, or you can
# split manually). Run top to bottom.
#
# NOTE ON STRUCTURE: Cells 1-8 (and the Fama-French cell) define classes and
# functions only — importing this file elsewhere (e.g. dash_app.py) does NOT
# trigger a full backtest run. Cells 9-16 (the actual pipeline execution,
# charts, and memo) live inside main(), which only runs when you execute this
# file directly (`python smart_beta_etf_builder.py`) or call main() yourself
# from the last Colab cell. This is what lets dash_app.py reuse run_pipeline()
# and PerformanceAnalytics without re-triggering matplotlib windows.
# =============================================================================

# %% CELL 1 — Install & Imports
# -----------------------------------------------------------------------------
"""
!pip install -q yfinance pandas numpy scipy matplotlib seaborn plotly quantstats
"""

import os
import warnings
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# The libraries below (yfinance, matplotlib, seaborn, scipy, statsmodels,
# pandas_datareader) are ONLY needed for the data-generation side of this
# project — DataPipeline (live Yahoo Finance pulls), the Colab/CLI charting
# in main(), PortfolioConstructor.min_variance's optimizer, and
# FamaFrenchAnalytics's regression. None of them are needed by the deployed
# Dash dashboard, which only reads the already-generated pipeline_cache.json.
#
# They're imported defensively (try/except, no hard failure) so that
# `import smart_beta_etf_builder` — and therefore `import dash_app` — still
# works on a minimal server install that only has pandas/numpy/plotly/dash.
# Run `pip install -r requirements.txt` (the full list) on your own machine
# before running precompute_cache.py or main(), which DO need these.
# -----------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:
    plt = mticker = sns = None
    _HAS_PLOTTING = False

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")
if _HAS_PLOTTING:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


# %% CELL 2 — Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Central configuration for the entire pipeline."""

    # Universe: a liquid large-cap subset for a fast, reliable demo run.
    # Swap in a full S&P 500 ticker list to scale up (expect slower fundamentals
    # pulls and more missing-data handling).
    universe: list = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM",
        "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD", "MRK", "ABBV", "COST", "PEP",
        "KO", "AVGO", "WMT", "BAC", "CSCO", "ADBE", "CRM", "ACN", "MCD", "LIN",
    ])
    benchmark: str = "SPY"

    # Universe mode:
    #   "pit"   — point-in-time S&P 500. At each rebalance, the `pit_top_n` most
    #             liquid stocks (trailing average dollar volume) among the
    #             index's members ON THAT DATE. No hindsight in stock selection.
    #   "fixed" — the hardcoded `universe` list above (the original approach).
    #             Kept as a comparison run to measure how much picking today's
    #             winners in hindsight inflates results.
    universe_mode: str = "pit"
    pit_top_n: int = 50
    liquidity_lookback: int = 63           # trading days of dollar volume for the liquidity rank
    min_history_coverage: float = 0.95     # share of the factor lookback a stock needs prices for
    membership_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "data", "sp500_membership.csv.gz")
    history_buffer_days: int = 420         # extra calendar days of prices before start_date so
                                           # factor lookbacks are full from the first rebalance

    start_date: str = "2015-01-01"
    end_date: str = "2026-01-01"

    rebalance_freq: str = "M"          # monthly rebalancing
    top_pct: float = 0.5               # select top 50% of universe by factor score
    max_weight: float = 0.10           # single-name cap for optimized portfolios
    transaction_cost_bps: float = 10.0  # 10 bps per unit of one-way turnover
    max_missing_pct: float = 0.10      # drop tickers missing more than this fraction of data
    risk_free_annual: float = 0.03     # fallback risk-free rate if FRED/T-bill pull unavailable

    momentum_lookback: int = 252
    momentum_skip: int = 21
    vol_lookback: int = 252

    # Reproducibility: fundamentals pulled from yfinance are a LIVE snapshot,
    # so re-running the notebook on a different day can shift Value/Quality
    # rankings even though the price history and universe are unchanged.
    # Caching to disk freezes the snapshot so results are identical across
    # reruns until you delete the cache file or set use_cached_fundamentals
    # = False to force a fresh pull.
    use_cached_fundamentals: bool = True
    fundamentals_cache_path: str = "/tmp/fundamentals_snapshot.csv"


CFG = Config()


# %% CELL 3 — Data Pipeline (Prices + Fundamentals)
# -----------------------------------------------------------------------------
class DataPipeline:
    """Handles all external data retrieval with defensive error handling."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_prices(self, tickers: Optional[list] = None, with_volume: bool = False,
                   batch_size: int = 100):
        """
        Download adjusted close prices (and optionally volume) for `tickers`
        (default: the fixed universe + benchmark), in batches. Prices start
        `history_buffer_days` before start_date so momentum / volatility
        lookbacks are already full on the first rebalance.
        """
        if yf is None:
            raise ImportError(
                "yfinance isn't installed. This is only needed for live data pulls "
                "(precompute_cache.py) — run `pip install yfinance` in that environment. "
                "The deployed dashboard shouldn't hit this path since it reads from "
                "pipeline_cache.json instead."
            )
        if tickers is None:
            tickers = list(dict.fromkeys(self.cfg.universe + [self.cfg.benchmark]))
        start = (pd.Timestamp(self.cfg.start_date)
                 - pd.Timedelta(days=self.cfg.history_buffer_days)).strftime("%Y-%m-%d")
        print(f"Downloading price history for {len(tickers)} tickers "
              f"({start} -> {self.cfg.end_date}, incl. lookback buffer) ...")

        closes, volumes, missing = {}, {}, []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                raw = yf.download(
                    batch, start=start, end=self.cfg.end_date,
                    auto_adjust=True,   # adjusted close folded into 'Close'
                    progress=False, group_by="ticker", threads=True,
                )
            except Exception as e:
                raise RuntimeError(f"Price download failed: {e}") from e
            for t in batch:
                try:
                    sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                    close = sub["Close"]
                    if close.notna().sum() == 0:
                        raise KeyError(t)
                    closes[t] = close
                    volumes[t] = sub["Volume"]
                except Exception:
                    missing.append(t)

        if missing:
            preview = ", ".join(sorted(missing)[:15])
            more = f" (+{len(missing) - 15} more)" if len(missing) > 15 else ""
            print(f"  [warn] no price data for {len(missing)} tickers "
                  f"(typically delisted/acquired): {preview}{more}")

        prices = pd.DataFrame(closes).sort_index().dropna(axis=1, how="all")
        volume = pd.DataFrame(volumes).sort_index().reindex(columns=prices.columns)
        print(f"Retrieved prices for {prices.shape[1]} tickers, "
              f"{prices.shape[0]} trading days.")
        return (prices, volume) if with_volume else prices

    def get_fundamentals(self, tickers: Optional[list] = None,
                         cache_path: Optional[str] = None) -> pd.DataFrame:
        """
        Best-effort fundamentals snapshot via yfinance `.info`.
        NOTE: this is a CURRENT snapshot, not point-in-time history, which
        introduces look-ahead bias in a rigorous backtest. Acceptable for a
        research prototype; see README Section 15 for the production fix
        (Financial Modeling Prep / SEC EDGAR point-in-time fundamentals).

        For run-to-run REPRODUCIBILITY, this snapshot is cached to disk by
        default (see Config.use_cached_fundamentals) so Value/Quality
        rankings don't silently drift if you re-run the notebook on a
        different day. Delete the cache file (or set the flag to False) to
        force a fresh pull.
        """
        tickers = list(tickers) if tickers is not None else list(self.cfg.universe)
        cache_path = cache_path or self.cfg.fundamentals_cache_path
        if self.cfg.use_cached_fundamentals and os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, index_col="ticker")
            if set(tickers).issubset(set(cached.index)):
                print(f"Loaded cached fundamentals snapshot from "
                      f"{cache_path} (delete this file, or set "
                      f"use_cached_fundamentals=False, to refresh).")
                return cached.reindex(tickers)

        records = []
        if yf is None:
            raise ImportError(
                "yfinance isn't installed and no cached fundamentals snapshot was found. "
                "Run `pip install yfinance` in the environment where you run "
                "precompute_cache.py."
            )
        print(f"Pulling fundamentals for {len(tickers)} tickers ...")
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                records.append({
                    "ticker": t,
                    "trailingPE": info.get("trailingPE", np.nan),
                    "priceToBook": info.get("priceToBook", np.nan),
                    "returnOnEquity": info.get("returnOnEquity", np.nan),
                    "debtToEquity": info.get("debtToEquity", np.nan),
                    "profitMargins": info.get("profitMargins", np.nan),
                })
            except Exception as e:
                print(f"  [warn] fundamentals failed for {t}: {e}")
                records.append({"ticker": t})
        fdf = pd.DataFrame(records).set_index("ticker")

        if self.cfg.use_cached_fundamentals:
            fdf.to_csv(cache_path)
            print(f"Cached fundamentals snapshot to {cache_path}")

        return fdf


# Ticker renames where Yahoo Finance keeps the full price history under the
# NEW symbol only. The membership file records the symbol in use at the time,
# so without this map, e.g., Facebook would be missing before it became META.
# Only renames of the same company (not mergers into a different company).
TICKER_RENAMES = {
    "FB": "META", "PCLN": "BKNG", "ANTM": "ELV", "UTX": "RTX", "ABC": "COR",
    "HRS": "LHX", "BBT": "TFC", "SYMC": "GEN", "NLOK": "GEN", "KORS": "CPRI",
    "WLTW": "WTW", "DWDP": "DD", "ADS": "BFH", "FBHS": "FBIN", "PKI": "RVTY",
    "RE": "EG", "FLT": "CPAY", "CDAY": "DAY", "COG": "CTRA",
}


def load_sp500_membership(path: str) -> pd.Series:
    """
    Historical S&P 500 membership, one row per date the index changed
    (source: github.com/fja05680/sp500). Returns a Series indexed by date
    whose values are frozensets of tickers in Yahoo format (BRK.B -> BRK-B,
    renamed symbols mapped to their current one).
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return pd.Series(
        [frozenset(TICKER_RENAMES.get(t.strip(), t.strip()).replace(".", "-")
                   for t in row.split(",") if t.strip())
         for row in df["tickers"]],
        index=pd.DatetimeIndex(df["date"]),
    )


class UniverseSelector:
    """
    Decides which stocks are investable on a given date, using only
    information available on that date.

    fixed mode: every stock in the (pre-cleaned) fixed universe.
    pit mode:   S&P 500 members as of that date, with enough price history for
                the factor lookbacks, ranked by trailing average dollar volume;
                the top `pit_top_n` are kept.
    """

    def __init__(self, cfg: Config, prices: pd.DataFrame,
                 volume: Optional[pd.DataFrame] = None,
                 membership: Optional[pd.Series] = None):
        self.cfg = cfg
        self.prices = prices
        self.membership = membership
        need = cfg.momentum_lookback + cfg.momentum_skip
        self._coverage = prices.notna().rolling(need, min_periods=1).mean()
        self._dollar_vol = None
        if volume is not None:
            self._dollar_vol = (prices * volume).rolling(
                cfg.liquidity_lookback, min_periods=int(cfg.liquidity_lookback * 0.8)).mean()
        self._cache = {}
        self.coverage_log = {}   # date -> [index members, members with prices, selected]

    def members_at(self, date: pd.Timestamp) -> frozenset:
        i = self.membership.index.searchsorted(date, side="right") - 1
        return self.membership.iloc[i] if i >= 0 else frozenset()

    def eligible(self, date: pd.Timestamp) -> list:
        if date in self._cache:
            return self._cache[date]
        has_price = self.prices.loc[date].notna()
        has_history = self._coverage.loc[date] >= self.cfg.min_history_coverage
        usable = has_price & has_history

        if self.cfg.universe_mode == "pit":
            members = self.members_at(date)
            in_panel = [t for t in members if t in self.prices.columns]
            priced = [t for t in in_panel if has_price[t]]
            candidates = [t for t in in_panel if usable[t]]
            dv = self._dollar_vol.loc[date, candidates].dropna().sort_values(ascending=False)
            names = sorted(dv.index[: self.cfg.pit_top_n])
            self.coverage_log[date] = [len(members), len(priced), len(names)]
        else:
            names = [t for t in self.prices.columns if usable[t]]

        self._cache[date] = names
        return names


# %% CELL 4 — Data Cleaning
# -----------------------------------------------------------------------------
class DataCleaner:
    """Cleans the raw price panel prior to factor computation and backtesting."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        missing_pct = prices.isna().mean()
        keep = missing_pct[missing_pct <= self.cfg.max_missing_pct].index
        dropped = set(prices.columns) - set(keep)
        if dropped:
            print(f"Dropping {len(dropped)} tickers with excessive missing data: "
                  f"{sorted(dropped)}")
        cleaned = prices[keep].copy()

        # Forward-fill only short gaps (e.g., holidays/glitches), not extended
        # halts or delistings, to avoid manufacturing artificial flat returns.
        cleaned = cleaned.ffill(limit=5)
        cleaned = cleaned.dropna(how="any")  # ensure a fully aligned panel
        return cleaned


# %% CELL 5 — Factor Engine
# -----------------------------------------------------------------------------
class FactorEngine:
    """Computes cross-sectional factor scores from price and fundamental data."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    def momentum_score(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """12-1 momentum: 12-month return, skipping the most recent month."""
        window = prices.loc[:as_of].tail(
            self.cfg.momentum_lookback + self.cfg.momentum_skip)
        if len(window) < self.cfg.momentum_lookback:
            return pd.Series(dtype=float)
        p_end = window.iloc[-self.cfg.momentum_skip - 1]
        p_start = window.iloc[0]
        raw = (p_end / p_start) - 1
        return self._zscore(raw)

    def low_vol_score(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Inverse realized volatility -> higher score = lower risk."""
        window = prices.loc[:as_of].tail(self.cfg.vol_lookback)
        if len(window) < 30:
            return pd.Series(dtype=float)
        rets = window.pct_change().dropna()
        vol = rets.std()
        return self._zscore(-vol)  # negate: low vol -> high score

    def value_score(self, fundamentals: pd.DataFrame) -> pd.Series:
        pe = fundamentals["trailingPE"].replace(0, np.nan)
        pb = fundamentals["priceToBook"].replace(0, np.nan)
        earnings_yield = 1.0 / pe
        book_yield = 1.0 / pb
        composite = pd.concat(
            [self._zscore(earnings_yield), self._zscore(book_yield)], axis=1
        ).mean(axis=1)
        return composite

    def quality_score(self, fundamentals: pd.DataFrame) -> pd.Series:
        roe = self._zscore(fundamentals["returnOnEquity"])
        margin = self._zscore(fundamentals["profitMargins"])
        leverage = self._zscore(-fundamentals["debtToEquity"])  # lower D/E is better
        composite = pd.concat([roe, margin, leverage], axis=1).mean(axis=1)
        return composite

    def multi_factor_score(self, *scores: pd.Series) -> pd.Series:
        aligned = pd.concat(scores, axis=1)
        return aligned.mean(axis=1)


# %% CELL 6 — Portfolio Constructor
# -----------------------------------------------------------------------------
class PortfolioConstructor:
    """Translates factor scores (or none, for equal weight) into portfolio weights."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def equal_weight(self, tickers: list) -> pd.Series:
        n = len(tickers)
        return pd.Series(1.0 / n, index=tickers)

    def factor_weighted(self, scores: pd.Series) -> pd.Series:
        """Select top `top_pct` by score, weight proportionally to score rank."""
        scores = scores.dropna()
        n_select = max(1, int(len(scores) * self.cfg.top_pct))
        top = scores.sort_values(ascending=False).head(n_select)
        shifted = top - top.min() + 1e-6  # ensure strictly positive weights
        weights = shifted / shifted.sum()
        return weights

    def min_variance(self, returns: pd.DataFrame) -> pd.Series:
        """
        Constrained minimum-variance optimization:
            minimize   w' Σ w
            subject to sum(w) = 1, 0 <= w_i <= max_weight
        """
        if minimize is None:
            raise ImportError(
                "scipy isn't installed. This is only needed for live pipeline runs "
                "(precompute_cache.py) — run `pip install scipy` in that environment."
            )
        cov = returns.cov().values
        n = cov.shape[0]
        tickers = returns.columns

        def objective(w):
            return w @ cov @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.cfg.max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            print(f"  [warn] min-variance optimizer did not converge cleanly: "
                  f"{result.message}. Falling back to equal weight.")
            return self.equal_weight(list(tickers))
        weights = pd.Series(result.x, index=tickers)
        weights[weights < 1e-4] = 0.0
        return weights / weights.sum()


# %% CELL 7 — Backtesting Engine
# -----------------------------------------------------------------------------
class Backtester:
    """
    Simulates monthly rebalancing for a given weighting strategy, applying
    transaction costs proportional to one-way turnover.

    Timing (no look-ahead):
      - Weights are decided at the CLOSE of a rebalance date, using data up to
        and including that date, and start earning returns on the NEXT trading day.
      - Rebalance dates are the last TRADING day of each month.
      - The initial portfolio is formed at the close of the last trading day
        before start_date, so every strategy is invested from day one (the
        price history before start_date is only used for factor lookbacks).
    """

    def __init__(self, cfg: Config, prices: pd.DataFrame):
        self.cfg = cfg
        self.prices = prices
        self.daily_returns = prices.pct_change(fill_method=None)
        start = pd.Timestamp(cfg.start_date)
        self.sim_index = self.daily_returns.index[self.daily_returns.index >= start]
        if len(self.sim_index) == 0:
            raise ValueError("No price data on or after start_date.")
        before = prices.index[prices.index < self.sim_index[0]]
        self.formation_date = before[-1] if len(before) else self.sim_index[0]

    def rebalance_dates(self) -> pd.DatetimeIndex:
        idx = self.sim_index
        month_ends = idx.to_series().groupby(idx.to_period("M")).max()
        return pd.DatetimeIndex(month_ends.values)

    @staticmethod
    def _turnover(new: pd.Series, old: pd.Series) -> float:
        names = new.index.union(old.index)
        return float((new.reindex(names, fill_value=0) - old.reindex(names, fill_value=0)).abs().sum() / 2.0)

    def _safe_weights(self, weight_func, date, current: pd.Series, name: str) -> pd.Series:
        try:
            w = weight_func(date)
            w = w[w.index.isin(self.prices.columns)].astype(float)
            w = w[w > 0]
            if w.sum() <= 0:
                raise ValueError("no positive weights")
            return w / w.sum()
        except Exception as e:
            print(f"  [warn] {name}: weight_func failed on {date.date()}: {e} — keeping prior weights")
            return current

    def run(self, weight_func, name: str) -> dict:
        """
        weight_func(as_of_date) -> pd.Series of weights (tickers -> weight)
        Runs the full historical simulation and returns equity curve, weight
        history, and turnover history.
        """
        rebal = set(self.rebalance_dates())
        weight_history, turnover_history = {}, {}
        current = pd.Series(dtype=float)
        portfolio_value = 1.0

        def rebalance(date):
            nonlocal current, portfolio_value
            new = self._safe_weights(weight_func, date, current, name)
            turnover = self._turnover(new, current)
            portfolio_value *= 1 - turnover * (self.cfg.transaction_cost_bps / 10000.0)
            current = new
            weight_history[date] = new
            turnover_history[date] = turnover

        rebalance(self.formation_date)
        values = []
        for date in self.sim_index:
            if len(current):
                day_ret = self.daily_returns.loc[date, current.index].fillna(0.0)
                portfolio_value *= 1 + float((day_ret * current).sum())
            values.append(portfolio_value)
            if date in rebal:
                rebalance(date)   # decided at today's close, earns from tomorrow

        return {
            "name": name,
            "equity_curve": pd.Series(values, index=self.sim_index),
            "weight_history": weight_history,
            "turnover_history": pd.Series(turnover_history),
        }


# %% CELL 8 — Performance Analytics
# -----------------------------------------------------------------------------
class PerformanceAnalytics:
    """Computes standard risk/return metrics from an equity curve."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_metrics(self, equity_curve: pd.Series, benchmark_curve: Optional[pd.Series] = None) -> dict:
        rets = equity_curve.pct_change().dropna()
        n_years = len(rets) / 252.0
        rf_daily = self.cfg.risk_free_annual / 252.0

        cagr = equity_curve.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() - rf_daily) / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan

        downside = rets[rets < 0]
        sortino = ((rets.mean() - rf_daily) / downside.std() * np.sqrt(252)
                   if len(downside) and downside.std() > 0 else np.nan)

        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        metrics = {
            "CAGR": cagr,
            "Ann. Volatility": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": max_dd,
            "Calmar": calmar,
        }

        if benchmark_curve is not None:
            b_rets = benchmark_curve.pct_change().dropna()
            aligned = pd.concat([rets, b_rets], axis=1, join="inner")
            aligned.columns = ["strategy", "benchmark"]
            active = aligned["strategy"] - aligned["benchmark"]
            info_ratio = (active.mean() / active.std() * np.sqrt(252)
                          if active.std() > 0 else np.nan)
            metrics["Information Ratio"] = info_ratio

        return metrics

    def rolling_sharpe(self, equity_curve: pd.Series, window: int = 126) -> pd.Series:
        rets = equity_curve.pct_change().dropna()
        rf_daily = self.cfg.risk_free_annual / 252.0
        roll_mean = rets.rolling(window).mean() - rf_daily
        roll_std = rets.rolling(window).std()
        return (roll_mean / roll_std) * np.sqrt(252)

    def drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        return equity_curve / equity_curve.cummax() - 1


# %% CELL 9 — Orchestration: Run the Full Pipeline
# -----------------------------------------------------------------------------
def run_pipeline(cfg: Config):
    pipeline = DataPipeline(cfg)
    cleaner = DataCleaner(cfg)
    factors = FactorEngine(cfg)
    constructor = PortfolioConstructor(cfg)
    bench = cfg.benchmark

    if cfg.universe_mode == "pit":
        membership = load_sp500_membership(cfg.membership_path)
        first = pd.Timestamp(cfg.start_date) - pd.Timedelta(days=cfg.history_buffer_days)
        last = pd.Timestamp(cfg.end_date)
        i0 = max(membership.index.searchsorted(first, side="right") - 1, 0)
        in_window = membership.iloc[i0:][membership.index[i0:] < last]
        candidates = sorted(set().union(*in_window.values))
        print(f"Point-in-time universe: {len(candidates)} tickers were S&P 500 members "
              f"at some point between {first.date()} and {last.date()}.")
        raw_prices, raw_volume = pipeline.get_prices(candidates + [bench], with_volume=True)
        benchmark_prices = raw_prices[bench].dropna()
        # Short gaps only — never fill across a delisting.
        universe_prices = raw_prices.drop(columns=[bench]).ffill(limit=5)
        volume = raw_volume.drop(columns=[bench]).reindex_like(universe_prices)
        selector = UniverseSelector(cfg, universe_prices, volume, membership)
    else:
        raw_prices = pipeline.get_prices()
        benchmark_prices = raw_prices[bench].dropna()
        universe_prices = cleaner.clean_prices(raw_prices.drop(columns=[bench]))
        selector = UniverseSelector(cfg, universe_prices)

    bt = Backtester(cfg, universe_prices)
    decision_dates = [bt.formation_date] + list(bt.rebalance_dates())
    ever_eligible = sorted(set().union(*(selector.eligible(d) for d in decision_dates)))
    print(f"{len(ever_eligible)} distinct stocks are investable at some rebalance.")

    if cfg.universe_mode == "pit":
        fundamentals = pipeline.get_fundamentals(
            ever_eligible, cache_path=cfg.fundamentals_cache_path.replace(".csv", "_pit.csv"))
    else:
        fundamentals = pipeline.get_fundamentals()
    fundamentals = fundamentals.reindex(universe_prices.columns)

    E = selector.eligible

    def px(d):
        return universe_prices[E(d)]

    def value_scores(d):
        return factors.value_score(fundamentals.reindex(E(d)))

    def quality_scores(d):
        return factors.quality_score(fundamentals.reindex(E(d)))

    results = {}

    # --- Equal Weight ---
    results["Equal Weight"] = bt.run(lambda d: constructor.equal_weight(E(d)), "Equal Weight")

    # --- Momentum ---
    results["Momentum"] = bt.run(
        lambda d: constructor.factor_weighted(factors.momentum_score(px(d), d)), "Momentum")

    # --- Low Volatility ---
    results["Min Volatility"] = bt.run(
        lambda d: constructor.factor_weighted(factors.low_vol_score(px(d), d)), "Min Volatility")

    # --- Value / Quality (fundamentals snapshot, ranked within each date's universe) ---
    results["Value"] = bt.run(lambda d: constructor.factor_weighted(value_scores(d)), "Value")
    results["Quality"] = bt.run(lambda d: constructor.factor_weighted(quality_scores(d)), "Quality")

    # --- Multi-Factor blend ---
    def multi_factor_weights(d):
        prices_d = px(d)
        combined = factors.multi_factor_score(
            factors.momentum_score(prices_d, d), factors.low_vol_score(prices_d, d),
            value_scores(d), quality_scores(d))
        return constructor.factor_weighted(combined)

    results["Multi-Factor"] = bt.run(multi_factor_weights, "Multi-Factor")

    # --- Minimum Variance (optimized) ---
    min_names = int(np.ceil(1.0 / cfg.max_weight))

    def min_var_weights(d):
        hist = (px(d).loc[:d].pct_change(fill_method=None)
                .tail(cfg.vol_lookback).dropna(axis=1, how="any"))
        if len(hist) < 60 or hist.shape[1] < min_names:
            return constructor.equal_weight(E(d))
        return constructor.min_variance(hist)

    results["Minimum Variance (Optimized)"] = bt.run(min_var_weights, "Minimum Variance (Optimized)")

    # --- Benchmark equity curve (buy & hold SPY), same simulation window ---
    bench_rets = benchmark_prices.pct_change().reindex(bt.sim_index).fillna(0.0)
    bench_curve = (1 + bench_rets).cumprod()
    bench_curve.iloc[0] = 1.0

    universe_log = [[str(d.date()), *selector.coverage_log[d]]
                    for d in sorted(selector.coverage_log)]

    return {
        "results": results,
        "benchmark_curve": bench_curve,
        "universe_prices": universe_prices,
        "fundamentals": fundamentals,
        "universe_mode": cfg.universe_mode,
        "universe_log": universe_log,
    }



# %% CELL 17 — Fama-French Factor Regression (Alpha / Beta Attribution)
# -----------------------------------------------------------------------------
"""
!pip install -q pandas-datareader statsmodels
"""
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

try:
    import pandas_datareader.data as pdr
except ImportError:
    pdr = None


class FamaFrenchAnalytics:
    """
    Pulls Fama-French factor returns from Kenneth French's data library and
    regresses each strategy's monthly excess return on them to report
    annualized alpha, factor betas, t-stats, and R-squared.

    FF3: Mkt-RF, SMB, HML
    FF5: Mkt-RF, SMB, HML, RMW, CMA
    """

    FF3_DATASET = "F-F_Research_Data_Factors"
    FF5_DATASET = "F-F_Research_Data_5_Factors_2x3"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_factors(self, model: str = "FF5") -> pd.DataFrame:
        """Returns monthly factor data (decimal, not %) with columns like
        ['Mkt-RF','SMB','HML','RMW','CMA','RF'], indexed by month-end date."""
        if pdr is None:
            raise ImportError(
                "pandas_datareader is required: `pip install pandas-datareader`"
            )
        dataset = self.FF5_DATASET if model == "FF5" else self.FF3_DATASET
        try:
            raw = pdr.DataReader(dataset, "famafrench",
                                  start=self.cfg.start_date, end=self.cfg.end_date)[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to pull Fama-French data ({dataset}) from Kenneth "
                f"French's library: {e}"
            ) from e
        factors = raw / 100.0  # French's files are in percent
        factors.index = factors.index.to_timestamp(how="end")
        return factors

    def regress(self, equity_curve: pd.Series, factors: pd.DataFrame) -> dict:
        """
        Resamples the strategy's equity curve to monthly returns, aligns with
        factor data, and runs OLS: (R_p - RF) = alpha + sum(beta_i * factor_i).
        """
        if sm is None:
            raise ImportError(
                "statsmodels is required: `pip install statsmodels`"
            )
        monthly_curve = equity_curve.resample("M").last()
        monthly_rets = monthly_curve.pct_change().dropna()
        monthly_rets.index = monthly_rets.index.to_period("M").to_timestamp(how="end")

        aligned = factors.join(monthly_rets.rename("strategy"), how="inner").dropna()
        if len(aligned) < 12:
            return {"error": "Not enough overlapping monthly observations to regress."}

        y = aligned["strategy"] - aligned["RF"]
        factor_cols = [c for c in aligned.columns if c not in ("strategy", "RF")]
        X = sm.add_constant(aligned[factor_cols])

        model = sm.OLS(y, X).fit()

        result = {
            "Annualized Alpha": model.params["const"] * 12,
            "Alpha t-stat": model.tvalues["const"],
            "R-squared": model.rsquared,
            "N Observations": int(model.nobs),
        }
        for col in factor_cols:
            result[f"Beta ({col})"] = model.params[col]
            result[f"t-stat ({col})"] = model.tvalues[col]
        return result

    def run_all(self, results: dict, model: str = "FF5") -> pd.DataFrame:
        """Runs the regression for every strategy in `results` and returns a
        tidy summary table."""
        factors = self.get_factors(model)
        rows = []
        for name, res in results.items():
            r = self.regress(res["equity_curve"], factors)
            r["Strategy"] = name
            rows.append(r)
        return pd.DataFrame(rows).set_index("Strategy")


def generate_investment_memo(report_df: pd.DataFrame) -> str:
    best_sharpe = report_df["Sharpe"].idxmax()
    best_row = report_df.loc[best_sharpe]

    if CFG.universe_mode == "pit":
        universe_risk = (
            f"- The universe is point-in-time: each month, the {CFG.pit_top_n} most-traded\n"
            "  stocks that were in the S&P 500 on that date. Free price data is missing\n"
            "  for some former members (mostly acquired or delisted companies), so a\n"
            "  small amount of survivorship bias remains.")
        universe_next = ("- Fill price gaps for delisted members (e.g. CRSP / Norgate) to remove\n"
                         "  the remaining survivorship bias.")
    else:
        universe_risk = (
            "- The universe (~30 large-cap names) is not survivorship-bias-free; expanding\n"
            "  to full historical S&P 500 membership would materially change results.")
        universe_next = "- Full historical index membership to remove survivorship bias."

    memo = f"""
INVESTMENT MEMO — SMART BETA ETF RESEARCH SUMMARY
==================================================

Strategy Overview
------------------
This research evaluates six systematic (smart beta) portfolio construction
methodologies — Equal Weight, Momentum, Minimum Volatility, Value, Quality,
and a blended Multi-Factor approach — alongside an optimized Minimum Variance
portfolio, benchmarked against SPY over {CFG.start_date} to {CFG.end_date}.

Headline Result
---------------
The best risk-adjusted performer by Sharpe ratio was "{best_sharpe}"
(Sharpe: {best_row['Sharpe']:.2f}, CAGR: {best_row['CAGR']:.2%},
Max Drawdown: {best_row['Max Drawdown']:.2%}).

Risks & Limitations
--------------------
- Fundamentals are a current-day snapshot (yfinance), not point-in-time,
  which introduces look-ahead bias versus a production-grade backtest.
{universe_risk}
- Transaction cost assumptions ({CFG.transaction_cost_bps} bps per unit
  turnover) are simplified and do not model market impact or bid-ask spread
  dynamics for less liquid names.
- Factor definitions are standard academic proxies; crowding and factor decay
  are real risks for any of these strategies deployed at scale.

Future Improvements
--------------------
- Point-in-time fundamentals (FMP / SEC EDGAR) to remove look-ahead bias.
{universe_next}
- Sector and position-size constraints in the optimizer.
- Regime-aware dynamic factor allocation.

For a plain-language walkthrough of how this dashboard works, where the data comes from,
and what each performance metric means, see the "How It Works" tab.
"""
    return memo


def generate_education_content() -> str:
    """
    Plain-language explainer of how the dashboard works, where the data comes
    from, what each strategy does, and what the performance metrics mean.
    Written once here so it can be reused both as printed output (Colab/CLI)
    and rendered as a dashboard tab (via dcc.Markdown in dash_app.py).
    """
    universe_list = ", ".join(CFG.universe)
    if CFG.universe_mode == "pit":
        universe_section = f"""## Which Stocks Are Eligible?

The universe is rebuilt every month using **only information available on that date**:

1. Start from the companies that were in the **S&P 500 on that date** — not today's list.
   Historical membership comes from a maintained public record of every index change since
   1996 (github.com/fja05680/sp500).
2. Keep the ones with enough price history for the factor calculations (about 13 months).
3. Rank them by **average daily dollar volume over the prior 3 months** and keep the
   **{CFG.pit_top_n} most-traded**. That's the investable universe for the month.

Why this matters: an earlier version of this project used a fixed list of 30 companies chosen
in 2026 (Apple, Microsoft, NVIDIA and other names we now know became huge winners). That
quietly bakes hindsight into every backtest. The Performance tab shows both versions side by
side so you can see how much that bias inflated the results.

**Remaining limitation:** free price data isn't available for some former index members —
mostly companies that were acquired or delisted. Those can't be selected, so a small amount of
survivorship bias remains."""
        n_equal = CFG.pit_top_n
    else:
        universe_section = f"""## Why These Specific Stocks?

The universe is a fixed list of {len(CFG.universe)} large, liquid, well-known U.S. companies:
{universe_list}.

These weren't picked to make any one strategy look good — they were chosen because they're
large-cap S&P 500 members with long, complete price histories and reliable data on Yahoo
Finance, which keeps the demo fast and avoids constant missing-data errors. It is **not** the
full S&P 500, and it is **not adjusted for survivorship bias** — every company on this list
still exists and is still doing fine today, which flatters every strategy's results somewhat
(a real-world backtest would also need to include companies that got delisted, went bankrupt,
or were acquired along the way). See the Investment Memo tab for more on this limitation."""
        n_equal = len(CFG.universe)
    return f"""
## What This Dashboard Does

This project builds six different rule-based ("smart beta") stock portfolios and compares
them against SPY, an ETF that tracks the S&P 500. Instead of picking stocks by hand, each
portfolio follows a fixed, repeatable rule — like "hold the stocks with the strongest recent
price gains" — and gets automatically re-evaluated and rebalanced once a month across the
full history from {CFG.start_date} to {CFG.end_date}.

## Where the Data Comes From

- **Prices** — daily stock prices, pulled from Yahoo Finance.
- **Fundamentals** — company financial ratios (like P/E ratio and profitability), also from
  Yahoo Finance, captured as a single snapshot in time.
- **Fama-French factors** — a well-known academic dataset (from Dartmouth's Kenneth French
  data library) used to check whether a strategy's returns are genuine skill or just a
  disguised bet on a known market factor.

{universe_section}

## How Each ETF Actually Decides Its Weights

Every factor strategy follows the same two-step recipe: **(1) score every stock, (2) keep only
the top {int(CFG.top_pct * 100)}% by score and size each position roughly in proportion to how
much better its score is than the rest of that top group.** A stock that doesn't make the cut
gets a 0% weight that month. Here's exactly what goes into each score:

- **Momentum score** — each stock's total return over the trailing 12 months, *skipping the
  most recent month* (a standard academic adjustment, since very recent moves tend to reverse
  short-term). Stocks with the strongest 12-month run get the biggest slices.
- **Value score** — an average of two ratios: **earnings yield** (1 ÷ P/E — how much profit
  you get per dollar of stock price) and **book yield** (1 ÷ P/B — how much accounting net
  worth you get per dollar of stock price). Higher earnings/book relative to price = "cheaper"
  = bigger slice.
- **Quality score** — an average of three signals: **Return on Equity** (how efficiently the
  company turns shareholder capital into profit), **profit margin** (how much of revenue
  becomes actual profit), and **debt-to-equity, inverted** (less debt scores higher). Companies
  that are profitable, efficient, and not over-leveraged get bigger slices.
- **Minimum Volatility score** — each stock's trailing 12-month return volatility (day-to-day
  choppiness), inverted so calmer stocks score higher. Note this is a *simple factor tilt*
  (rank-and-weight by low volatility), which is different from the next one.
- **Multi-Factor score** — the plain average of the Momentum, Low-Volatility, Value, and
  Quality scores above, all put on the same scale first (see "z-scores" below) so no single
  factor accidentally dominates just because its raw numbers are bigger.
- **Minimum Variance (Optimized)** — this one doesn't use a "score" at all. It feeds the last
  year of daily returns into a mathematical optimizer that directly searches for the specific
  combination of position sizes that minimizes the *portfolio's* overall volatility (not each
  stock's individual volatility) — including how the stocks move together, not just how much
  each one moves alone. A cap of {int(CFG.max_weight * 100)}% per stock keeps it from just
  dumping everything into one or two names.
- **Equal Weight** — no scoring at all; every stock simply gets 1 ÷ {n_equal} of the
  portfolio, rebalanced back to equal shares every month.

**A technical note on "scores":** raw numbers like P/E or ROE aren't directly comparable to
each other (a P/E of 20 and an ROE of 20% mean very different things), so every score is first
converted to a **z-score** — basically "how many standard deviations above or below the
group's average is this stock on this measure" — before stocks are ranked or blended together.

## How the Backtest Works

Each strategy is simulated month by month: at the close of the last trading day of each month,
the portfolio is recalculated using only data available up to that point, and the new weights
start earning returns the next trading day. They're held until the next rebalance.
A small trading cost ({CFG.transaction_cost_bps} basis points per trade) is subtracted each
time the portfolio changes, so the results reflect realistic friction rather than a
frictionless, idealized world.

## What the Numbers Mean

- **CAGR** — the average annual growth rate, smoothed over the whole period.
- **Volatility** — how much the portfolio's value swings up and down; higher means bumpier.
- **Sharpe Ratio** — return earned per unit of risk taken; higher is generally better.
- **Sortino Ratio** — like Sharpe, but only penalizes downside swings, not upside ones.
- **Max Drawdown** — the worst peak-to-trough loss the portfolio experienced.
- **Calmar Ratio** — annual return divided by the worst drawdown; a "return per unit of pain"
  measure.
- **Information Ratio** — how consistently a strategy beat SPY, adjusted for how much it
  deviated from SPY to do so.
- **Alpha** — the return a strategy earned that *can't* be explained by broad market
  movements or known factors; the "extra credit," positive or negative.
- **Beta** — how sensitive a strategy is to a given factor (e.g., a Beta of 1.2 on the market
  factor means it tends to move about 20% more than the market in either direction).

## Does This Data Update Automatically?

Yes — **once a month.** Live-pulling from Yahoo Finance on every visit would be slow and risk
rate limits, so the full backtest runs on a schedule instead: on the 3rd of each month, GitHub
Actions rebuilds the data through the last completed month, runs a set of sanity checks, and only
then publishes it here. If a check fails, the site keeps the previous month's data. The "Data
refreshed" badge at the top of the page shows when the current snapshot was generated.

## Important Caveats

This is a research prototype, not investment advice. It uses a snapshot of fundamentals
rather than true point-in-time data, free price data that is missing some delisted companies,
and simplified trading-cost assumptions. See the Investment Memo tab for the full list of
limitations and possible next steps.
"""


def main():
    print("Running Smart Beta ETF Builder pipeline ...")
    pipeline_output = run_pipeline(CFG)
    print("Pipeline complete.")


    # %% CELL 10 — Performance Report Table
    # -----------------------------------------------------------------------------
    analytics = PerformanceAnalytics(CFG)
    report_rows = []

    for name, res in pipeline_output["results"].items():
        m = analytics.compute_metrics(res["equity_curve"], pipeline_output["benchmark_curve"])
        m["Strategy"] = name
        m["Avg Turnover"] = res["turnover_history"].mean()
        report_rows.append(m)

    bench_metrics = analytics.compute_metrics(pipeline_output["benchmark_curve"])
    bench_metrics["Strategy"] = "SPY (Benchmark)"
    bench_metrics["Avg Turnover"] = 0.0
    report_rows.append(bench_metrics)

    report_df = pd.DataFrame(report_rows).set_index("Strategy")
    report_df = report_df[["CAGR", "Ann. Volatility", "Sharpe", "Sortino",
                            "Max Drawdown", "Calmar", "Information Ratio", "Avg Turnover"]]

    print("\n===== RISK / PERFORMANCE DASHBOARD =====\n")
    print(report_df.round(4))

    # --- Fama-French factor attribution (alpha/beta vs Mkt-RF, SMB, HML, RMW, CMA) ---
    ff_df = pd.DataFrame()
    try:
        ff = FamaFrenchAnalytics(CFG)
        ff_df = ff.run_all(pipeline_output["results"], model="FF5")
        print("\n===== FAMA-FRENCH 5-FACTOR ATTRIBUTION =====\n")
        print(ff_df.round(4))
    except Exception as e:
        print(f"\n[warn] Fama-French regression skipped: {e}")


    # %% CELL 11 — Visualization: Equity Curves
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots()
    for name, res in pipeline_output["results"].items():
        ax.plot(res["equity_curve"], label=name, linewidth=1.6)
    ax.plot(pipeline_output["benchmark_curve"], label="SPY (Benchmark)",
            color="black", linewidth=2.2, linestyle="--")
    ax.set_title("Smart Beta Strategies vs. SPY — Growth of $1")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
    plt.tight_layout()
    plt.savefig("/tmp/equity_curve.png", dpi=150)
    plt.show()


    # %% CELL 12 — Visualization: Drawdown Chart
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots()
    for name, res in pipeline_output["results"].items():
        dd = analytics.drawdown_series(res["equity_curve"])
        ax.plot(dd, label=name, linewidth=1.4)
    bench_dd = analytics.drawdown_series(pipeline_output["benchmark_curve"])
    ax.plot(bench_dd, label="SPY (Benchmark)", color="black", linewidth=2.0, linestyle="--")
    ax.fill_between(bench_dd.index, bench_dd.values, 0, color="black", alpha=0.05)
    ax.set_title("Drawdown: Smart Beta Strategies vs. SPY")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig("/tmp/drawdown.png", dpi=150)
    plt.show()


    # %% CELL 13 — Visualization: Rolling Sharpe Ratio
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots()
    for name, res in pipeline_output["results"].items():
        rs = analytics.rolling_sharpe(res["equity_curve"])
        ax.plot(rs, label=name, linewidth=1.3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Rolling 6-Month Sharpe Ratio")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("/tmp/rolling_sharpe.png", dpi=150)
    plt.show()


    # %% CELL 14 — Visualization: Current Allocation (Multi-Factor Portfolio)
    # -----------------------------------------------------------------------------
    mf_weights_history = pipeline_output["results"]["Multi-Factor"]["weight_history"]
    latest_date = max(mf_weights_history.keys())
    latest_weights = mf_weights_history[latest_date].sort_values(ascending=False)
    latest_weights = latest_weights[latest_weights > 0.001]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(latest_weights, labels=latest_weights.index, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 8})
    ax.set_title(f"Multi-Factor ETF Allocation as of {latest_date.date()}")
    plt.tight_layout()
    plt.savefig("/tmp/allocation_pie.png", dpi=150)
    plt.show()


    # %% CELL 15 — Visualization: Correlation Matrix
    # -----------------------------------------------------------------------------
    curves = {name: res["equity_curve"].pct_change() for name, res in pipeline_output["results"].items()}
    curves["SPY"] = pipeline_output["benchmark_curve"].pct_change()
    corr_df = pd.DataFrame(curves).dropna()
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Strategy Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig("/tmp/correlation_matrix.png", dpi=150)
    plt.show()


    # %% CELL 16 — Investment Memo (Auto-Populated Template)
    # -----------------------------------------------------------------------------
    memo_text = generate_investment_memo(report_df)
    print(memo_text)

    with open("/tmp/investment_memo.txt", "w") as f:
        f.write(memo_text)

    print("\nSaved charts and memo to /tmp/. Copy these into results/charts/ "
          "and results/performance_report.pdf for your GitHub repo.")


if __name__ == "__main__":
    main()
