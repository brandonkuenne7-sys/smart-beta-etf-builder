# =============================================================================
# PRECOMPUTE CACHE — run this on your own machine or in Colab (full internet
# access), NOT on PythonAnywhere's free tier (which can't reach Yahoo Finance).
#
# It runs the full pipeline once and saves everything the dashboard needs into
# pipeline_cache.json. Upload that single file to PythonAnywhere alongside
# dash_app.py, smart_beta_etf_builder.py, and requirements.txt.
#
# JSON (not pickle) is used deliberately: pickle files embed the exact numpy/
# pandas internal object format they were created with, so a cache made on
# your Mac's numpy 2.x can fail to load on a server running numpy 1.x
# ("No module named 'numpy._core'"). JSON is just plain text — numbers and
# strings — so it loads identically regardless of which numpy/pandas version
# is installed on either end.
#
# Re-run this locally + re-upload the .json whenever you want fresher data.
# =============================================================================

import json
from datetime import datetime, timezone
from smart_beta_etf_builder import (
    CFG, run_pipeline, PerformanceAnalytics, FamaFrenchAnalytics,
)

print("Running full pipeline (requires internet access to Yahoo Finance) ...")
pipeline_output = run_pipeline(CFG)

print("Running Fama-French attribution (requires internet access) ...")
try:
    ff_df = FamaFrenchAnalytics(CFG).run_all(pipeline_output["results"], model="FF5")
    ff_records = ff_df.reset_index().to_dict(orient="records")
except Exception as e:
    print(f"[warn] Fama-French regression skipped: {e}")
    ff_records = []


def _series_to_pairs(s):
    """Convert a pandas Series with a datetime index into plain [str, float]
    pairs so it can round-trip through JSON with no numpy/pandas dependency."""
    return [[str(idx), float(val)] for idx, val in s.items()]


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

cache = {
    "results": results_json,
    "benchmark_curve": _series_to_pairs(pipeline_output["benchmark_curve"]),
    "ff_attribution": ff_records,
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
}

with open("pipeline_cache.json", "w") as f:
    json.dump(cache, f)

print("Saved pipeline_cache.json — upload this file to PythonAnywhere.")
