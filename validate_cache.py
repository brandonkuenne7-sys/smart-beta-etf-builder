"""
Sanity checks for a freshly generated pipeline_cache.json, run by the monthly
refresh workflow BEFORE anything is published. If any check fails, the
workflow stops, GitHub emails you, and the live site keeps last month's data.

Usage:
    python validate_cache.py NEW_CACHE.json [--previous OLD_CACHE.json]
"""

import argparse
import json
import math
import sys
from datetime import date, timedelta

EXPECTED_STRATEGIES = {
    "Equal Weight", "Momentum", "Min Volatility", "Value", "Quality",
    "Multi-Factor", "Minimum Variance (Optimized)",
}
MIN_TRADING_DAYS = 2000          # ~8 years; the backtest starts in 2015
MAX_STALENESS_DAYS = 10          # last price vs. the (exclusive) end date
SHARPE_RANGE = (-2.0, 4.0)       # anything outside this is almost certainly a data error
CAGR_RANGE = (-0.5, 1.0)


def _day(s: str) -> date:
    return date.fromisoformat(s[:10])


def _curve_stats(pairs):
    values = [v for _, v in pairs]
    rets = [b / a - 1 for a, b in zip(values, values[1:]) if a]
    n = len(rets)
    mean = sum(rets) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (n - 1))
    sharpe = (mean / sd) * math.sqrt(252) if sd else float("nan")
    cagr = values[-1] / values[0]
    cagr = cagr ** (252 / n) - 1 if cagr > 0 else float("nan")
    return sharpe, cagr


def validate(cache: dict, previous: dict | None) -> list[str]:
    errors = []

    for key in ("results", "benchmark_curve", "ff_attribution", "end_date"):
        if key not in cache:
            errors.append(f"missing top-level key '{key}'")
    if errors:
        return errors

    end = _day(cache["end_date"])

    names = set(cache["results"])
    if names != EXPECTED_STRATEGIES:
        errors.append(f"strategy set changed: missing {sorted(EXPECTED_STRATEGIES - names)}, "
                      f"unexpected {sorted(names - EXPECTED_STRATEGIES)}")

    curves = {name: res.get("equity_curve", []) for name, res in cache["results"].items()}
    curves["SPY (benchmark)"] = cache["benchmark_curve"]

    for name, pairs in curves.items():
        if len(pairs) < MIN_TRADING_DAYS:
            errors.append(f"{name}: only {len(pairs)} days of history")
            continue
        bad = [d for d, v in pairs if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0)]
        if bad:
            errors.append(f"{name}: {len(bad)} missing/invalid values (first at {bad[0]})")
            continue
        last = _day(pairs[-1][0])
        if (end - last).days > MAX_STALENESS_DAYS:
            errors.append(f"{name}: last data point {last} is stale vs end date {end}")
        sharpe, cagr = _curve_stats(pairs)
        if not SHARPE_RANGE[0] <= sharpe <= SHARPE_RANGE[1]:
            errors.append(f"{name}: Sharpe {sharpe:.2f} outside plausible range {SHARPE_RANGE}")
        if not CAGR_RANGE[0] <= cagr <= CAGR_RANGE[1]:
            errors.append(f"{name}: CAGR {cagr:.1%} outside plausible range {CAGR_RANGE}")

    for name, res in cache["results"].items():
        wh = res.get("weight_history") or {}
        if not wh:
            errors.append(f"{name}: no holdings history")
            continue
        latest = wh[max(wh)]
        total = sum(latest.values())
        if not 0.98 <= total <= 1.02 or any(w < -1e-9 for w in latest.values()):
            errors.append(f"{name}: latest weights sum to {total:.3f} or include shorts")

    ff = cache["ff_attribution"]
    if len(ff) != len(EXPECTED_STRATEGIES):
        errors.append(f"Fama-French attribution has {len(ff)} rows, expected {len(EXPECTED_STRATEGIES)}")

    if previous:
        prev_last = _day(previous["benchmark_curve"][-1][0])
        new_last = _day(cache["benchmark_curve"][-1][0])
        if new_last < prev_last:
            errors.append(f"new data ends {new_last}, earlier than the live data ({prev_last})")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("--previous")
    args = parser.parse_args()

    with open(args.cache) as f:
        cache = json.load(f)
    previous = None
    if args.previous:
        with open(args.previous) as f:
            previous = json.load(f)

    errors = validate(cache, previous)
    if errors:
        print("Cache validation FAILED — not publishing:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    last = cache["benchmark_curve"][-1][0][:10]
    print(f"Cache OK: {len(cache['results'])} strategies, data through {last}, "
          f"{len(cache['benchmark_curve'])} trading days.")


if __name__ == "__main__":
    main()
