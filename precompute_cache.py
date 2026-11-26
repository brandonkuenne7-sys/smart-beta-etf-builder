# =============================================================================
# PRECOMPUTE CACHE — runs the full pipeline and saves everything the dashboard
# needs into pipeline_cache.json.
#
# Where it runs:
#   - Automatically, once a month, on GitHub Actions
#     (.github/workflows/refresh_data.yml), which then uploads the file to
#     PythonAnywhere and reloads the site.
#   - Or by hand on your own machine:  python precompute_cache.py
#   NOT on PythonAnywhere's free tier, which can't reach Yahoo Finance.
#
# Backtest window:
#   The start date comes from Config. The end date defaults to the first day
#   of the current month (yfinance treats `end` as exclusive), so the backtest
#   always runs through the most recent completed month. Override with
#   --end-date YYYY-MM-DD or the REFRESH_END_DATE environment variable.
#   The window is written into the cache so the dashboard can label it.
#
# JSON (not pickle) is used deliberately: pickle files embed the exact numpy/
# pandas internal object format they were created with, so a cache made on
# numpy 2.x can fail to load on a server running numpy 1.x. JSON is plain
# text and loads identically regardless of library versions on either end.
# =============================================================================

import argparse
import json
import os
from datetime import datetime, timezone

from smart_beta_etf_builder import (
    CFG, run_pipeline, FamaFrenchAnalytics,
)


def _default_end_date() -> str:
    today = datetime.now(timezone.utc).date()
    return today.replace(day=1).isoformat()


def _series_to_pairs(s):
    """Convert a pandas Series with a datetime index into plain [str, float]
    pairs so it can round-trip through JSON with no numpy/pandas dependency."""
    return [[str(idx), float(val)] for idx, val in s.items()]


def main():
    parser = argparse.ArgumentParser(description="Rebuild pipeline_cache.json")
    parser.add_argument("--end-date", default=os.environ.get("REFRESH_END_DATE") or _default_end_date(),
                        help="Backtest end date, exclusive (default: first day of the current month)")
    parser.add_argument("--output", default="pipeline_cache.json", help="Where to write the cache")
    args = parser.parse_args()

    # CFG is the shared Config instance, so this also updates the dates the
    # memo / education text reference.
    CFG.end_date = args.end_date
    print(f"Backtest window: {CFG.start_date} -> {CFG.end_date} (end exclusive)")

    print("Running full pipeline (requires internet access to Yahoo Finance) ...")
    pipeline_output = run_pipeline(CFG)

    print("Running Fama-French attribution (requires internet access) ...")
    try:
        ff_df = FamaFrenchAnalytics(CFG).run_all(pipeline_output["results"], model="FF5")
        ff_records = ff_df.reset_index().to_dict(orient="records")
    except Exception as e:
        print(f"[warn] Fama-French regression skipped: {e}")
        ff_records = []

    results_json = {}
    for name, res in pipeline_output["results"].items():
        results_json[name] = {
            "equity_curve": _series_to_pairs(res["equity_curve"]),
            "turnover_history": _series_to_pairs(res["turnover_history"]),
            "weight_history": {
                str(date): {str(t): float(w) for t, w in weights.items()}
                for date, weights in res["weight_history"].items()
            },
        }

    benchmark = pipeline_output["benchmark_curve"]
    cache = {
        "results": results_json,
        "benchmark_curve": _series_to_pairs(benchmark),
        "ff_attribution": ff_records,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "start_date": CFG.start_date,
        "end_date": CFG.end_date,
        "data_through": str(benchmark.index.max().date()),
    }

    # Write to a temp file first so a crash mid-write never leaves a
    # half-written cache where the dashboard (or the workflow) would pick it up.
    tmp_path = args.output + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_path, args.output)

    print(f"Saved {args.output} (data through {cache['data_through']}).")


if __name__ == "__main__":
    main()
