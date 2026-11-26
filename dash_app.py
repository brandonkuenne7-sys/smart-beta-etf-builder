# =============================================================================
# SMART BETA ETF BUILDER — Interactive Dash Dashboard
#
# Run locally:      python dash_app.py        (then open http://127.0.0.1:8050)
# Run in Colab:     !pip install dash jupyter-dash
#                   then use app.run(mode="inline") or app.run(mode="external")
# Deploy for free:  Render.com, Railway, Fly.io, or PythonAnywhere all support
#                   Dash out of the box — push this file + requirements.txt +
#                   smart_beta_etf_builder.py to a repo and connect it.
#
# This app RE-USES the pipeline classes from smart_beta_etf_builder.py rather
# than duplicating logic. Importing that module does NOT auto-run the
# backtest (see the NOTE at the top of that file) — we call run_pipeline()
# once here, cache the result in memory, and let all the Dash callbacks read
# from that cache. Re-running the underlying backtest is expensive, so the
# UI filters/redraws already-computed results instead of recomputing them.
# =============================================================================

"""
!pip install -q dash plotly pandas-datareader statsmodels yfinance
"""

import os
import json

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme

from smart_beta_etf_builder import (
    CFG, run_pipeline, PerformanceAnalytics, FamaFrenchAnalytics,
    generate_investment_memo, generate_education_content,
)

# -----------------------------------------------------------------------------
# 1. Load pipeline results.
#
# If pipeline_cache.json exists (produced by precompute_cache.py), load it —
# this is the path used for free-tier PythonAnywhere hosting, which can't
# reach Yahoo Finance / Kenneth French's site directly. Otherwise, run the
# pipeline live (local dev, Colab, or a paid host with open internet access).
#
# JSON is used instead of pickle specifically because it has zero dependency
# on which numpy/pandas version is installed — plain numbers/strings load
# identically everywhere, unlike a pickled DataFrame.
# -----------------------------------------------------------------------------
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_cache.json")


def _pairs_to_series(pairs):
    """Reconstruct a datetime-indexed pandas Series from [str, float] pairs."""
    idx = pd.to_datetime([p[0] for p in pairs])
    vals = [p[1] for p in pairs]
    return pd.Series(vals, index=idx).sort_index()


if os.path.exists(CACHE_PATH):
    print(f"Loading cached pipeline results from {CACHE_PATH} ...")
    with open(CACHE_PATH) as f:
        _cache = json.load(f)

    PIPELINE = {"results": {}, "benchmark_curve": _pairs_to_series(_cache["benchmark_curve"])}
    for name, res in _cache["results"].items():
        weight_history = {
            pd.to_datetime(date): pd.Series(weights)
            for date, weights in res["weight_history"].items()
        }
        PIPELINE["results"][name] = {
            "equity_curve": _pairs_to_series(res["equity_curve"]),
            "turnover_history": _pairs_to_series(res["turnover_history"]),
            "weight_history": weight_history,
        }

    FF_DF_RAW = (pd.DataFrame(_cache["ff_attribution"]).set_index("Strategy")
                 if _cache["ff_attribution"] else pd.DataFrame())
    DATA_GENERATED_AT = _cache.get("generated_at", "unknown")

    # Newer caches record the backtest window they were built for. Use it so
    # the subtitle, Investment Memo and How It Works text describe the data
    # actually on screen (older caches fall back to Config's dates).
    CFG.start_date = _cache.get("start_date", CFG.start_date)
    CFG.end_date = _cache.get("end_date", CFG.end_date)

    # Which universe produced this cache (older caches = the fixed 30-stock list).
    UNIVERSE_INFO = _cache.get("universe") or {"mode": "fixed"}
    CFG.universe_mode = UNIVERSE_INFO.get("mode", "fixed")
    CFG.pit_top_n = UNIVERSE_INFO.get("top_n", CFG.pit_top_n)
    _comparison = _cache.get("comparison") or {}
    COMPARISON_LABEL = _comparison.get("label", "Hindsight universe")
    COMPARISON_CURVES = {name: _pairs_to_series(r["equity_curve"])
                         for name, r in (_comparison.get("results") or {}).items()}
else:
    print("No pipeline_cache.json found — running pipeline live (needs internet access).")
    PIPELINE = run_pipeline(CFG)
    try:
        FF_DF_RAW = FamaFrenchAnalytics(CFG).run_all(PIPELINE["results"], model="FF5")
    except Exception as e:
        print(f"[warn] Fama-French regression unavailable: {e}")
        FF_DF_RAW = pd.DataFrame()
    from datetime import datetime, timezone
    DATA_GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + " (live run)"
    UNIVERSE_INFO = {"mode": CFG.universe_mode, "top_n": CFG.pit_top_n,
                     "log": PIPELINE.get("universe_log", [])}
    COMPARISON_LABEL, COMPARISON_CURVES = "Hindsight universe", {}

ANALYTICS = PerformanceAnalytics(CFG)
STRATEGY_NAMES = list(PIPELINE["results"].keys())
BENCH_CURVE = PIPELINE["benchmark_curve"]

# Precompute the metrics dashboard table
_rows = []
for name, res in PIPELINE["results"].items():
    m = ANALYTICS.compute_metrics(res["equity_curve"], BENCH_CURVE)
    m["Strategy"] = name
    m["Avg Turnover"] = res["turnover_history"].mean()
    _rows.append(m)
bench_m = ANALYTICS.compute_metrics(BENCH_CURVE)
bench_m["Strategy"] = "SPY (Benchmark)"
bench_m["Avg Turnover"] = 0.0
_rows.append(bench_m)
METRICS_DF = pd.DataFrame(_rows).set_index("Strategy").round(4).reset_index()

FF_DF = FF_DF_RAW.round(4).reset_index() if not FF_DF_RAW.empty else FF_DF_RAW

MEMO_TEXT = generate_investment_memo(METRICS_DF.set_index("Strategy"))
EDUCATION_TEXT = generate_education_content()

COLORS = {
    "Equal Weight": "#4C72B0", "Momentum": "#DD8452", "Min Volatility": "#55A868",
    "Value": "#C44E52", "Quality": "#8172B2", "Multi-Factor": "#937860",
    "Minimum Variance (Optimized)": "#DA8BC3",
}

# One-line hover definitions shown when you hover over a column header.
METRIC_DEFINITIONS = {
    "Strategy": "The name of the portfolio construction rule being tested.",
    "CAGR": "Compound Annual Growth Rate — the average yearly growth rate, smoothed over the full backtest period.",
    "Ann. Volatility": "Annualized Volatility — how much the portfolio's value swings up and down over a year; higher means bumpier.",
    "Sharpe": "Sharpe Ratio — return earned per unit of total risk taken; higher is generally better.",
    "Sortino": "Sortino Ratio — like Sharpe, but only penalizes downside swings, not upside ones.",
    "Max Drawdown": "The worst peak-to-trough loss the portfolio experienced at any point in the backtest.",
    "Calmar": "Calmar Ratio — annual return divided by the worst drawdown; a 'return per unit of pain' measure.",
    "Information Ratio": "How consistently a strategy beat SPY, adjusted for how much it deviated from SPY to do so.",
    "Avg Turnover": "The average share of the portfolio replaced at each monthly rebalance; higher turnover means more trading costs.",
    "Annualized Alpha": "The yearly return that can't be explained by market movement or known factors — the strategy's 'extra credit.'",
    "Alpha t-stat": "A statistical confidence score for Alpha; roughly, values above ~2 or below ~-2 suggest the alpha isn't just noise.",
    "R-squared": "How much of the strategy's return is explained by the factors in this regression, from 0 (none) to 1 (all of it).",
    "N Observations": "The number of monthly return data points used in this regression.",
}
FACTOR_DEFINITIONS = {
    "Mkt-RF": "Market factor — the broad stock market's return above the risk-free rate.",
    "SMB": "Size factor ('Small Minus Big') — the historical tendency of small-cap stocks to outperform large-caps.",
    "HML": "Value factor ('High Minus Low') — the historical tendency of cheap (value) stocks to outperform expensive (growth) stocks.",
    "RMW": "Profitability factor ('Robust Minus Weak') — the tendency of highly profitable companies to outperform.",
    "CMA": "Investment factor ('Conservative Minus Aggressive') — the tendency of conservatively-investing companies to outperform.",
}
for _factor, _desc in FACTOR_DEFINITIONS.items():
    METRIC_DEFINITIONS[f"Beta ({_factor})"] = f"Sensitivity to the {_factor} factor. {_desc}"
    METRIC_DEFINITIONS[f"t-stat ({_factor})"] = f"Statistical confidence score for the {_factor} beta; roughly, values above ~2 or below ~-2 suggest it's not just noise."


# Used by the interactive "Strategy Explorer" tab — the formula/inputs behind
# each strategy's scoring, paired with a live chart of its actual current
# top holdings (pulled from the real backtest weight history, not a mockup).
STRATEGY_EXPLANATIONS = {
    "Equal Weight": {
        "formula": f"weight = 1 ÷ {CFG.pit_top_n if CFG.universe_mode == 'pit' else len(CFG.universe)} for every stock",
        "description": "No scoring at all. Every stock in the universe gets the exact same "
                        "slice of the portfolio, rebalanced back to equal shares every month. "
                        "This is the simplest possible baseline — useful for seeing whether "
                        "any of the fancier strategies actually add value.",
    },
    "Momentum": {
        "formula": "score = 12-month return, skipping the most recent month",
        "description": "Ranks stocks by how strongly they've risen over the trailing year "
                        "(excluding the most recent month, a standard adjustment since very "
                        "recent moves tend to reverse short-term). The strongest recent "
                        "performers get the biggest slices.",
    },
    "Min Volatility": {
        "formula": "score = −(trailing 12-month daily return volatility)",
        "description": "Ranks stocks by how calm and steady their day-to-day price moves have "
                        "been over the past year, favoring the least jumpy names. This is a "
                        "simple rank-and-weight tilt — different from the optimizer-based "
                        "Minimum Variance strategy below.",
    },
    "Value": {
        "formula": "score = average( 1 ÷ P/E , 1 ÷ P/B )",
        "description": "Averages two 'cheapness' ratios: earnings yield (profit per dollar of "
                        "stock price) and book yield (accounting net worth per dollar of stock "
                        "price). Stocks that look cheap relative to their fundamentals get the "
                        "biggest slices.",
    },
    "Quality": {
        "formula": "score = average( ROE , profit margin , −debt/equity )",
        "description": "Averages three financial-health signals: Return on Equity (how "
                        "efficiently the company turns shareholder capital into profit), "
                        "profit margin (how much of revenue becomes actual profit), and "
                        "debt-to-equity (inverted, so less debt scores higher). Financially "
                        "strong, efficient, low-leverage companies get the biggest slices.",
    },
    "Multi-Factor": {
        "formula": "score = average( Momentum z-score, Low-Vol z-score, Value z-score, "
                   "Quality z-score )",
        "description": "Blends all four factor scores above into one signal, after first "
                        "putting each on the same scale (a z-score) so no single factor "
                        "dominates just because its raw numbers happen to be bigger. The idea: "
                        "combining independent signals is often more reliable than betting on "
                        "any single one.",
    },
    "Minimum Variance (Optimized)": {
        "formula": f"minimize wᵀΣw subject to Σw = 1, 0 ≤ w ≤ {int(CFG.max_weight * 100)}%",
        "description": "Doesn't use a 'score' at all. A mathematical optimizer searches the "
                        "last year of daily returns for the exact combination of position "
                        f"sizes (capped at {int(CFG.max_weight * 100)}% each) that minimizes "
                        "the portfolio's overall volatility — accounting for how stocks move "
                        "together, not just how much each one moves alone.",
    },
}


def _tooltip_header(columns):
    """Build the tooltip_header dict Dash expects, only for columns we have a definition for."""
    return {
        c: {"value": METRIC_DEFINITIONS[c], "type": "markdown"}
        for c in columns if c in METRIC_DEFINITIONS
    }


# ---- Table display formats (presentation only; underlying numbers unchanged) ----
# Formats are applied by the DataTable itself, so sorting still works on the
# raw numeric values.
_PCT_1 = Format(precision=1, scheme=Scheme.percentage)
_FIXED_2 = Format(precision=2, scheme=Scheme.fixed)
_INT = Format(precision=0, scheme=Scheme.fixed)

_METRIC_PCT_COLS = {"CAGR", "Ann. Volatility", "Max Drawdown", "Avg Turnover"}
_FF_PAIRS = [("Annualized Alpha", "Alpha t-stat")] + [
    (f"Beta ({f})", f"t-stat ({f})") for f in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
]
SIGNIFICANCE_T = 2.0
_STRATEGY_COL_LEFT = [{"if": {"column_id": "Strategy"}, "textAlign": "left", "fontWeight": 500}]


def _table_columns(columns, pct_cols=(), int_cols=()):
    out = []
    for c in columns:
        if c == "Strategy":
            out.append({"name": c, "id": c})
        elif c in pct_cols:
            out.append({"name": c, "id": c, "type": "numeric", "format": _PCT_1})
        elif c in int_cols:
            out.append({"name": c, "id": c, "type": "numeric", "format": _INT})
        else:
            out.append({"name": c, "id": c, "type": "numeric", "format": _FIXED_2})
    return out


def _ff_significance_styles():
    """Shade each coefficient and its t-stat when |t| >= 2; colour alpha by sign."""
    rules = []
    for coef, tcol in _FF_PAIRS:
        query = f"{{{tcol}}} >= {SIGNIFICANCE_T} || {{{tcol}}} <= -{SIGNIFICANCE_T}"
        for col in (coef, tcol):
            rules.append({"if": {"filter_query": query, "column_id": col},
                          "backgroundColor": "var(--accent-soft)", "fontWeight": 600,
                          "color": "var(--text-primary)"})
    rules += [
        {"if": {"filter_query": f"{{Alpha t-stat}} >= {SIGNIFICANCE_T}", "column_id": "Annualized Alpha"},
         "backgroundColor": "var(--positive-soft)", "color": "var(--positive)", "fontWeight": 600},
        {"if": {"filter_query": f"{{Alpha t-stat}} <= -{SIGNIFICANCE_T}", "column_id": "Annualized Alpha"},
         "backgroundColor": "var(--negative-soft)", "color": "var(--negative)", "fontWeight": 600},
    ]
    return rules


def _memo_to_components(text):
    """Turn the engine's plain-text memo (underlined headings, '- ' bullets,
    wrapped lines) into structured HTML. The memo text itself is unchanged."""
    lines = text.strip("\n").split("\n")
    blocks, para, bullets = [], [], []

    def flush():
        if para:
            blocks.append(html.P(" ".join(para), className="memo-p"))
            para.clear()
        if bullets:
            blocks.append(html.Ul([html.Li(b) for b in bullets], className="memo-list"))
            bullets.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line and nxt and set(nxt) <= {"=", "-"} and len(nxt) >= 3:
            flush()
            if set(nxt) == {"="}:
                pass  # document title — the card already has a heading
            else:
                blocks.append(html.H4(line.strip(), className="memo-h"))
            i += 2
            continue
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("- "):
            if para:
                flush()
            bullets.append(stripped[2:])
        elif bullets and line.startswith("  "):
            bullets[-1] += " " + stripped       # continuation of a wrapped bullet
        else:
            if bullets:
                flush()
            para.append(stripped)
        i += 1
    flush()
    return blocks

# -----------------------------------------------------------------------------
# 2. App shell, design tokens, and derived (KPI) data
#
# Everything below is presentation only — no new backend computation is
# introduced. All figures/tables are built from PIPELINE, METRICS_DF, FF_DF,
# STRATEGY_EXPLANATIONS, MEMO_TEXT, and EDUCATION_TEXT, all already computed
# above from the existing analytics engine.
# -----------------------------------------------------------------------------
import re

import dash_bootstrap_components as dbc
from dash import ctx

app = Dash(
    __name__,
    title="Smart Beta ETF Builder",
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # exposed for gunicorn / production WSGI deployment

PLOTLY_FONT = dict(family="Inter, -apple-system, sans-serif", size=12, color="#5C6472")
PLOTLY_TRANSITION = dict(duration=450, easing="cubic-in-out")
CHART_CONFIG = {"displayModeBar": False}


def _base_figure(title, yaxis_title, y_percent=False, height=440):
    """Shared Plotly styling: Inter font, soft gridlines, animated transitions,
    rich unified hover — this is what makes chart switches feel smooth instead
    of an abrupt redraw."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Inter", color="#10151F"), x=0.0),
        template="plotly_white",
        height=height,
        margin=dict(l=55, r=25, t=50, b=40),
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        yaxis_title=yaxis_title,
        font=PLOTLY_FONT,
        plot_bgcolor="white",
        paper_bgcolor="white",
        transition=PLOTLY_TRANSITION,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter", bordercolor="#E6E8EC"),
        xaxis=dict(showgrid=False, showspikes=True, spikemode="across",
                    spikecolor="#D6D9E0", spikethickness=1, spikedash="dot"),
        yaxis=dict(gridcolor="#F0F1F4", zeroline=False),
    )
    if y_percent:
        fig.update_yaxes(tickformat=".0%")
    return fig


def _kpi_card(icon, label, value, sub, positive=None, delay_class=""):
    value_class = "kpi-value" + (" kpi-positive" if positive is True else
                                  " kpi-negative" if positive is False else "")
    return html.Div([
        html.Div(html.I(className=f"bi {icon}"), className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div(value, className=value_class),
        html.Div(sub, className="kpi-sub"),
    ], className=f"kpi-card fade-in {delay_class}")


# ---- Headline KPI stats, derived from the already-computed METRICS_DF ----
_by_strategy = METRICS_DF[METRICS_DF["Strategy"] != "SPY (Benchmark)"].set_index("Strategy")
_best_sharpe_name = _by_strategy["Sharpe"].idxmax()
_best_sharpe_val = _by_strategy.loc[_best_sharpe_name, "Sharpe"]
_best_cagr_name = _by_strategy["CAGR"].idxmax()
_best_cagr_val = _by_strategy.loc[_best_cagr_name, "CAGR"]
_best_dd_name = _by_strategy["Max Drawdown"].idxmax()  # least negative = shallowest
_best_dd_val = _by_strategy.loc[_best_dd_name, "Max Drawdown"]
_spy_sharpe = METRICS_DF.set_index("Strategy").loc["SPY (Benchmark)", "Sharpe"]

kpi_row = html.Div([
    _kpi_card("bi-trophy", "Best Risk-Adjusted Return", _best_sharpe_name,
              f"Sharpe {_best_sharpe_val:.2f}", positive=True, delay_class="fade-in-1"),
    _kpi_card("bi-graph-up-arrow", "Highest CAGR", _best_cagr_name,
              f"{_best_cagr_val:.1%} / yr", positive=True, delay_class="fade-in-2"),
    _kpi_card("bi-shield-check", "Shallowest Drawdown", _best_dd_name,
              f"{_best_dd_val:.1%} worst decline", positive=True, delay_class="fade-in-3"),
    _kpi_card("bi-bar-chart-line", "SPY Benchmark Sharpe", f"{_spy_sharpe:.2f}",
              "S&P 500 buy-and-hold", delay_class="fade-in-4"),
    _kpi_card("bi-clock-history", "Strategies Tracked", str(len(STRATEGY_NAMES)),
              (f"Top {CFG.pit_top_n} point-in-time S&P 500" if CFG.universe_mode == "pit"
               else "vs. 1 benchmark"), delay_class="fade-in-5"),
], className="kpi-row")

header = html.Div([
    html.Div([
        html.H1([html.I(className="bi bi-graph-up-arrow", style={"color": "#1652F0",
                                                                   "marginRight": "10px"}),
                 "Smart Beta ETF Builder"], className="app-title"),
        html.P(f"Institutional-style factor ETF research — backtested {CFG.start_date} to "
               f"{CFG.end_date} vs. SPY" + (f", on a point-in-time S&P 500 universe."
                                             if CFG.universe_mode == "pit" else "."),
               className="app-subtitle"),
    ]),
    html.Div([html.Span(className="dot"), f" Data refreshed {DATA_GENERATED_AT}"],
              className="app-meta-badge"),
], className="app-header fade-in")


# -----------------------------------------------------------------------------
# 3. Static (non-interactive) figures — built once, no callback needed since
#    they don't depend on any user input.
# -----------------------------------------------------------------------------
def _build_correlation_figure():
    curves = {name: PIPELINE["results"][name]["equity_curve"].pct_change() for name in STRATEGY_NAMES}
    curves["SPY"] = BENCH_CURVE.pct_change()
    corr = pd.DataFrame(curves).dropna().corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(template="plotly_white", height=460, font=PLOTLY_FONT,
                       margin=dict(l=20, r=20, t=20, b=20),
                       plot_bgcolor="white", paper_bgcolor="white")
    return fig


def _build_alpha_figure():
    df = FF_DF.set_index("Strategy")["Annualized Alpha"].sort_values()
    colors = ["#16A34A" if v >= 0 else "#DC2626" for v in df.values]
    fig = go.Figure(go.Bar(
        x=df.values, y=df.index, orientation="h", marker_color=colors,
        text=[f"{v:.1%}" for v in df.values], textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#D6D9E0")
    fig.update_layout(template="plotly_white", height=360, font=PLOTLY_FONT,
                       margin=dict(l=170, r=50, t=20, b=40), xaxis_tickformat=".0%",
                       plot_bgcolor="white", paper_bgcolor="white")
    return fig


CORR_FIGURE = _build_correlation_figure()


# ---- Hindsight-bias check: point-in-time universe vs. the original fixed list ----
def _build_bias_table():
    rows = []
    for name in STRATEGY_NAMES:
        if name not in COMPARISON_CURVES:
            continue
        pit = ANALYTICS.compute_metrics(PIPELINE["results"][name]["equity_curve"], BENCH_CURVE)
        hind = ANALYTICS.compute_metrics(COMPARISON_CURVES[name], BENCH_CURVE)
        rows.append({
            "Strategy": name,
            "CAGR (point-in-time)": pit["CAGR"],
            "CAGR (hindsight list)": hind["CAGR"],
            "Inflation from hindsight": hind["CAGR"] - pit["CAGR"],
            "Sharpe (point-in-time)": pit["Sharpe"],
            "Sharpe (hindsight list)": hind["Sharpe"],
        })
    return pd.DataFrame(rows)


def _build_bias_figure(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Strategy"], y=df["CAGR (hindsight list)"], name=COMPARISON_LABEL,
                         marker_color="#D6D9E0",
                         text=[f"{v:.1%}" for v in df["CAGR (hindsight list)"]], textposition="outside"))
    fig.add_trace(go.Bar(x=df["Strategy"], y=df["CAGR (point-in-time)"],
                         name=f"Point-in-time top {CFG.pit_top_n} S&P 500", marker_color="#1652F0",
                         text=[f"{v:.1%}" for v in df["CAGR (point-in-time)"]], textposition="outside"))
    spy_cagr = ANALYTICS.compute_metrics(BENCH_CURVE)["CAGR"]
    fig.add_hline(y=spy_cagr, line_dash="dash", line_color="#10151F",
                  annotation_text=f"SPY {spy_cagr:.1%}", annotation_position="top left")
    fig.update_layout(barmode="group", template="plotly_white", height=400, font=PLOTLY_FONT,
                      yaxis_tickformat=".0%", yaxis_title="CAGR",
                      margin=dict(l=50, r=20, t=30, b=60), plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def _universe_coverage_text():
    log = UNIVERSE_INFO.get("log") or []
    if not log:
        return ""
    members = sum(r[1] for r in log) / len(log)
    priced = sum(r[2] / r[1] for r in log if r[1]) / len(log)
    return (f"Each month the universe is the {CFG.pit_top_n} most-traded stocks among that month's "
            f"S&P 500 members (~{members:.0f} names). On average {priced:.0%} of members had usable "
            f"free price data; the rest are mostly acquired or delisted companies, so a little "
            f"survivorship bias remains.")


BIAS_DF = _build_bias_table() if COMPARISON_CURVES else pd.DataFrame()


def _build_education_accordion():
    """Splits generate_education_content()'s markdown on '## ' headers into
    collapsible accordion sections, reusing the exact same text used in the
    Colab/console printout — no content is duplicated or rewritten here."""
    stripped = EDUCATION_TEXT.strip()
    sections = re.split(r"\n## ", stripped)
    if sections[0].startswith("## "):
        sections[0] = sections[0][3:]
    items = []
    for i, block in enumerate(sections):
        parts = block.strip().split("\n", 1)
        heading = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""
        items.append(dbc.AccordionItem(dcc.Markdown(body), title=heading, item_id=f"edu-{i}"))
    return items


# -----------------------------------------------------------------------------
# 4. Tab content
# -----------------------------------------------------------------------------
strategy_filter = dbc.Checklist(
    id="strategy-selector",
    options=[{"label": s, "value": s} for s in STRATEGY_NAMES],
    value=STRATEGY_NAMES,
    inline=True,
    className="strategy-filter",
)

chart_toggle = dbc.RadioItems(
    id="chart-type-toggle",
    options=[
        {"label": "Equity Curve", "value": "equity"},
        {"label": "Drawdown", "value": "drawdown"},
        {"label": "Rolling Sharpe", "value": "sharpe"},
    ],
    value="equity",
    inline=True,
    className="chart-toggle",
)

performance_tab = html.Div([
    html.Div([
        html.Div("Performance Explorer", className="section-title"),
        html.P("Toggle chart type, filter strategies, and click any point on the chart to "
               "drill into that month's holdings on the Allocation tab.",
               className="section-desc"),
        dbc.Row([
            dbc.Col(chart_toggle, width="auto"),
            dbc.Col(strategy_filter, width=True),
        ], className="mb-3", justify="start"),
        dcc.Loading(dcc.Graph(id="main-chart", config=CHART_CONFIG), type="circle", color="#1652F0"),
    ], className="section-card"),

    (html.Div([
        html.Div("Hindsight-Bias Check", className="section-title"),
        html.P(["The same seven strategies run twice: once on the ", html.B("original fixed list"),
                " of 30 companies chosen in 2026 (today's winners, picked with hindsight), and once on a ",
                html.B("point-in-time universe"), " built each month from that date's S&P 500 members. ",
                "The gap is how much hindsight flattered the original results."],
               className="section-desc"),
        dcc.Graph(figure=_build_bias_figure(BIAS_DF), config=CHART_CONFIG),
        dash_table.DataTable(
            id="bias-table",
            columns=_table_columns(BIAS_DF.columns, pct_cols={
                "CAGR (point-in-time)", "CAGR (hindsight list)", "Inflation from hindsight"}),
            data=BIAS_DF.to_dict("records"),
            style_cell={"padding": "10px 14px", "fontFamily": "Inter, Arial", "fontSize": 13,
                        "minWidth": "110px"},
            style_cell_conditional=_STRATEGY_COL_LEFT,
            style_data_conditional=[
                {"if": {"column_id": "Inflation from hindsight", "filter_query": "{Inflation from hindsight} > 0"},
                 "color": "var(--negative)", "fontWeight": 600},
            ],
            style_table={"overflowX": "auto"},
            style_as_list_view=True,
            sort_action="native",
        ),
        html.P(_universe_coverage_text(), className="section-desc",
               style={"marginTop": "14px", "marginBottom": 0}),
    ], className="section-card") if not BIAS_DF.empty else html.Div()),

    dbc.Accordion([
        dbc.AccordionItem(
            dash_table.DataTable(
                id="metrics-table",
                columns=_table_columns(METRICS_DF.columns, pct_cols=_METRIC_PCT_COLS),
                style_cell_conditional=_STRATEGY_COL_LEFT,
                data=METRICS_DF.to_dict("records"),
                tooltip_header=_tooltip_header(METRICS_DF.columns),
                tooltip_delay=0, tooltip_duration=None,
                style_cell={"padding": "10px 14px", "fontFamily": "Inter, Arial", "fontSize": 13,
                            "minWidth": "110px"},
                style_table={"overflowX": "auto"},
                style_as_list_view=True,
                sort_action="native",
            ),
            title="Full Risk / Performance Metrics Table", item_id="metrics-acc",
        ),
    ], start_collapsed=True, className="mt-1"),
], className="fade-in")

strategy_explorer_tab = html.Div([
    html.Div([
        html.Div("Strategy Explorer", className="section-title"),
        html.P("Select a strategy to see exactly how it scores stocks and what it's actually "
               "holding right now. Add a second strategy to compare them head-to-head.",
               className="section-desc"),
        dbc.Row([
            dbc.Col([
                html.Label("Strategy", className="kpi-label", style={"display": "block", "marginBottom": "6px"}),
                dcc.Dropdown(id="explorer-strategy",
                             options=[{"label": s, "value": s} for s in STRATEGY_EXPLANATIONS],
                             value="Momentum", clearable=False),
            ], md=6),
            dbc.Col([
                html.Label("Compare Against (optional)", className="kpi-label", style={"display": "block", "marginBottom": "6px"}),
                dcc.Dropdown(id="explorer-compare",
                             options=[{"label": "None", "value": "none"}] +
                                     [{"label": s, "value": s} for s in STRATEGY_EXPLANATIONS],
                             value="none", clearable=False),
            ], md=6),
        ], className="mb-3 g-3"),
        html.Div(id="explorer-description"),
        html.Div(id="explorer-compare-kpis"),
        dcc.Loading(dcc.Graph(id="explorer-holdings-chart", config=CHART_CONFIG),
                    type="circle", color="#1652F0"),
    ], className="section-card"),
], className="fade-in")

allocation_tab = html.Div([
    html.Div([
        html.Div("Portfolio Allocation", className="section-title"),
        html.P("Pick a strategy and view its holdings as a pie or treemap. Click a point on "
               "the Performance tab's chart to jump to that month's snapshot.",
               className="section-desc"),
        html.Div([
            html.Span(id="drilldown-text"),
            html.Button([html.I(className="bi bi-arrow-counterclockwise"), " Reset to Latest"],
                        id="reset-date-btn", className="btn-ghost"),
        ], id="drilldown-info", className="drilldown-banner", style={"display": "none"}),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="allocation-strategy",
                                  options=[{"label": s, "value": s} for s in STRATEGY_NAMES],
                                  value="Multi-Factor", clearable=False), md=8),
            dbc.Col(dbc.RadioItems(id="allocation-view-toggle",
                                    options=[{"label": "Pie", "value": "pie"},
                                             {"label": "Treemap", "value": "treemap"}],
                                    value="pie", inline=True, className="chart-toggle"), md=4),
        ], className="mb-3 g-2"),
        dcc.Loading(dcc.Graph(id="allocation-chart", config=CHART_CONFIG), type="circle", color="#1652F0"),
    ], className="section-card"),

    html.Div([
        html.Div("Strategy Correlation Matrix", className="section-title"),
        html.P("How similarly each strategy's monthly returns move relative to the others and to SPY.",
               className="section-desc"),
        dcc.Graph(figure=CORR_FIGURE, config=CHART_CONFIG),
    ], className="section-card"),

    dcc.Store(id="selected-date-store", data=None),
], className="fade-in")

attribution_tab = html.Div([
    html.Div([
        html.Div("Fama-French 5-Factor Attribution", className="section-title"),
        html.P("Annualized alpha and factor loadings vs. Mkt-RF, SMB, HML, RMW, CMA. Hover any "
               "column header for a definition.", className="section-desc"),
        html.Div([
            html.Span(className="sig-swatch sig-swatch-accent"), "Statistically significant (|t| ≥ 2)",
            html.Span(className="sig-swatch sig-swatch-positive"), "Significant positive alpha",
            html.Span(className="sig-swatch sig-swatch-negative"), "Significant negative alpha",
        ], className="sig-legend"),
        (dash_table.DataTable(
            id="ff-table",
            columns=_table_columns(FF_DF.columns, pct_cols={"Annualized Alpha"},
                                   int_cols={"N Observations"}),
            data=FF_DF.to_dict("records"),
            style_data_conditional=_ff_significance_styles(),
            style_cell_conditional=_STRATEGY_COL_LEFT,
            tooltip_header=_tooltip_header(FF_DF.columns),
            tooltip_delay=0, tooltip_duration=None,
            style_cell={"padding": "10px 14px", "fontFamily": "Inter, Arial", "fontSize": 13,
                        "minWidth": "100px"},
            style_table={"overflowX": "auto"},
            style_as_list_view=True,
            sort_action="native",
        ) if not FF_DF.empty else html.P(
            "Fama-French data wasn't available when this snapshot was generated.",
            style={"color": "var(--text-muted)"})),
    ], className="section-card"),

    (html.Div([
        html.Div("Annualized Alpha by Strategy", className="section-title"),
        html.P("Extra return not explained by market or factor exposure — positive means "
               "genuine outperformance beyond what the five factors predict.",
               className="section-desc"),
        dcc.Graph(figure=_build_alpha_figure(), config=CHART_CONFIG),
    ], className="section-card") if not FF_DF.empty else html.Div()),
], className="fade-in")

how_it_works_tab = html.Div([
    html.Div([
        dbc.Accordion(_build_education_accordion(), always_open=True, active_item="edu-0"),
    ], className="section-card"),
], className="fade-in")

memo_tab = html.Div([
    html.Div([
        html.Div("Investment Memo", className="section-title"),
        html.P("Auto-generated summary of strategy performance, risks, and next steps.",
               className="section-desc"),
        html.Div(_memo_to_components(MEMO_TEXT), className="memo-body"),
    ], className="section-card"),
], className="fade-in")

tabs = dbc.Tabs([
    dbc.Tab(performance_tab, label="Performance", tab_id="tab-performance"),
    dbc.Tab(strategy_explorer_tab, label="Strategy Explorer", tab_id="tab-explorer"),
    dbc.Tab(allocation_tab, label="Allocation & Correlation", tab_id="tab-allocation"),
    dbc.Tab(attribution_tab, label="Factor Attribution", tab_id="tab-attribution"),
    dbc.Tab(how_it_works_tab, label="How It Works", tab_id="tab-education"),
    dbc.Tab(memo_tab, label="Investment Memo", tab_id="tab-memo"),
], id="main-tabs", active_tab="tab-performance", className="custom-tabs")

footer = html.Div(
    "This is a research prototype for educational purposes — not investment advice.",
    style={"textAlign": "center", "color": "var(--text-muted)", "fontSize": "12px", "marginTop": "40px"},
)

app.layout = html.Div(html.Div([header, kpi_row, tabs, footer], className="app-shell"))


# -----------------------------------------------------------------------------
# 5. Callbacks
# -----------------------------------------------------------------------------
@app.callback(Output("main-chart", "figure"),
              Input("chart-type-toggle", "value"),
              Input("strategy-selector", "value"))
def update_main_chart(chart_type, selected):
    selected = selected or []

    if chart_type == "drawdown":
        fig = _base_figure("Drawdown", "Drawdown", y_percent=True)
        for name in selected:
            dd = ANALYTICS.drawdown_series(PIPELINE["results"][name]["equity_curve"])
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, mode="lines",
                                      line=dict(color=COLORS.get(name), width=2),
                                      hovertemplate="%{y:.1%}<extra>" + name + "</extra>"))
        bench_dd = ANALYTICS.drawdown_series(BENCH_CURVE)
        fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, name="SPY (Benchmark)",
                                  line=dict(color="#10151F", dash="dash", width=2),
                                  fill="tozeroy", fillcolor="rgba(16,21,31,0.04)",
                                  hovertemplate="%{y:.1%}<extra>SPY</extra>"))
    elif chart_type == "sharpe":
        fig = _base_figure("Rolling 6-Month Sharpe Ratio", "Sharpe Ratio")
        for name in selected:
            rs = ANALYTICS.rolling_sharpe(PIPELINE["results"][name]["equity_curve"])
            fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name=name, mode="lines",
                                      line=dict(color=COLORS.get(name), width=2),
                                      hovertemplate="%{y:.2f}<extra>" + name + "</extra>"))
        fig.add_hline(y=0, line_color="#D6D9E0", line_width=1)
    else:
        fig = _base_figure("Growth of $1 — Smart Beta Strategies vs. SPY", "Portfolio Value ($)")
        for name in selected:
            curve = PIPELINE["results"][name]["equity_curve"]
            fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=name, mode="lines",
                                      line=dict(color=COLORS.get(name), width=2),
                                      hovertemplate="%{y:$.2f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=BENCH_CURVE.index, y=BENCH_CURVE.values, name="SPY (Benchmark)",
                                  line=dict(color="#10151F", dash="dash", width=2),
                                  hovertemplate="%{y:$.2f}<extra>SPY</extra>"))

    fig.update_layout(hovermode="x unified")
    return fig


@app.callback(Output("selected-date-store", "data"),
              Input("main-chart", "clickData"),
              Input("reset-date-btn", "n_clicks"),
              prevent_initial_call=True)
def update_selected_date(click_data, _reset_clicks):
    if ctx.triggered_id == "reset-date-btn":
        return None
    if click_data and click_data.get("points"):
        return click_data["points"][0]["x"]
    return None


@app.callback(
    Output("allocation-chart", "figure"),
    Output("drilldown-text", "children"),
    Output("drilldown-info", "style"),
    Input("allocation-strategy", "value"),
    Input("allocation-view-toggle", "value"),
    Input("selected-date-store", "data"),
)
def update_allocation_chart(strategy, view_type, selected_date):
    weight_history = PIPELINE["results"][strategy]["weight_history"]
    dates = sorted(weight_history.keys())

    if selected_date:
        target = pd.to_datetime(selected_date)
        eligible = [d for d in dates if d <= target]
        use_date = max(eligible) if eligible else min(dates)
        banner_style = {"display": "flex"}
        banner_text = [
            html.I(className="bi bi-crosshair"),
            f" Drilled into {use_date.date()} (clicked on Performance chart)",
        ]
    else:
        use_date = max(dates)
        banner_style = {"display": "none"}
        banner_text = []

    weights = weight_history[use_date].sort_values(ascending=False)
    weights = weights[weights > 0.001]

    if view_type == "treemap":
        fig = go.Figure(go.Treemap(
            labels=weights.index, parents=[""] * len(weights), values=weights.values,
            marker=dict(colors=weights.values, colorscale="Blues"),
            textinfo="label+percent parent",
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    else:
        if len(weights) > 15:   # keep the donut readable for 25-50 holdings
            weights = pd.concat([weights.head(15),
                                 pd.Series({f"Other ({len(weights) - 15})": weights.iloc[15:].sum()})])
        fig = go.Figure(data=[go.Pie(
            labels=weights.index, values=weights.values, hole=0.45, sort=False,
            marker=dict(line=dict(color="white", width=2)),
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        )])
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    fig.update_layout(
        title=dict(text=f"{strategy} — {use_date.date()}", font=dict(size=14, family="Inter", color="#10151F")),
        template="plotly_white", height=480, font=PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white", transition=PLOTLY_TRANSITION,
        legend=dict(font=dict(size=11)),
    )
    return fig, banner_text, banner_style


@app.callback(
    Output("explorer-description", "children"),
    Output("explorer-compare-kpis", "children"),
    Output("explorer-holdings-chart", "figure"),
    Input("explorer-strategy", "value"),
    Input("explorer-compare", "value"),
)
def update_strategy_explorer(strategy, compare):
    info = STRATEGY_EXPLANATIONS[strategy]
    description = html.Div([
        html.Div(info["formula"], className="formula-chip"),
        html.P(info["description"], style={"lineHeight": "1.65", "color": "var(--text-secondary)",
                                            "fontSize": "14px"}),
    ], className="fade-in")

    weight_history = PIPELINE["results"][strategy]["weight_history"]
    latest_date = max(weight_history.keys())
    weights = weight_history[latest_date].sort_values(ascending=False)
    weights = weights[weights > 0.001].head(10)
    tickers = list(weights.index)

    kpis = html.Div()

    if compare and compare != "none" and compare != strategy:
        cmp_wh = PIPELINE["results"][compare]["weight_history"]
        cmp_latest = max(cmp_wh.keys())
        cmp_weights = cmp_wh[cmp_latest]
        cmp_vals = [cmp_weights.get(t, 0.0) for t in tickers]

        fig = go.Figure()
        fig.add_trace(go.Bar(y=tickers[::-1], x=weights.values[::-1], name=strategy, orientation="h",
                              marker_color=COLORS.get(strategy, "#1652F0"),
                              text=[f"{w:.1%}" for w in weights.values[::-1]], textposition="outside"))
        fig.add_trace(go.Bar(y=tickers[::-1], x=cmp_vals[::-1], name=compare, orientation="h",
                              marker_color=COLORS.get(compare, "#DD8452"),
                              text=[f"{w:.1%}" for w in cmp_vals[::-1]], textposition="outside"))
        fig.update_layout(barmode="group",
                           title=dict(text=f"{strategy} vs. {compare} — Top Holdings",
                                      font=dict(size=14, family="Inter", color="#10151F")))

        m1 = METRICS_DF.set_index("Strategy").loc[strategy]
        m2 = METRICS_DF.set_index("Strategy").loc[compare]

        def _diff_card(label, v1, v2, fmt):
            delta = v1 - v2
            arrow = "bi-arrow-up-short" if delta > 0 else ("bi-arrow-down-short" if delta < 0 else "bi-dash")
            cls = "kpi-positive" if delta > 0 else ("kpi-negative" if delta < 0 else "")
            return html.Div([
                html.Div(label, className="kpi-label"),
                html.Div(f"{fmt.format(v1)} vs {fmt.format(v2)}", className="kpi-value",
                          style={"fontSize": "16px"}),
                html.Div([html.I(className=f"bi {arrow}"), f" {fmt.format(abs(delta))} difference"],
                          className=f"kpi-sub {cls}"),
            ], className="app-card", style={"flex": "1"})

        kpis = html.Div([
            _diff_card("Sharpe Ratio", m1["Sharpe"], m2["Sharpe"], "{:.2f}"),
            _diff_card("CAGR", m1["CAGR"], m2["CAGR"], "{:.1%}"),
            _diff_card("Max Drawdown", m1["Max Drawdown"], m2["Max Drawdown"], "{:.1%}"),
        ], style={"display": "flex", "gap": "12px", "margin": "16px 0"}, className="fade-in")
    else:
        fig = go.Figure(data=[go.Bar(
            x=weights.values[::-1], y=tickers[::-1], orientation="h",
            marker_color=COLORS.get(strategy, "#1652F0"),
            text=[f"{w:.1%}" for w in weights.values[::-1]], textposition="outside",
        )])
        fig.update_layout(title=dict(text=f"{strategy} — Current Top Holdings (as of {latest_date.date()})",
                                      font=dict(size=14, family="Inter", color="#10151F")))

    fig.update_layout(template="plotly_white", height=440, font=PLOTLY_FONT,
                       margin=dict(l=70, r=40, t=50, b=40), xaxis_title="Portfolio Weight",
                       xaxis_tickformat=".0%", transition=PLOTLY_TRANSITION,
                       plot_bgcolor="white", paper_bgcolor="white")
    return description, kpis, fig


# -----------------------------------------------------------------------------
# 6. Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)