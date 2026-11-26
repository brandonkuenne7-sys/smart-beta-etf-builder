"""
Show the raw SEC EDGAR figures behind a ticker's point-in-time Value/Quality
inputs. Useful for checking outliers.

Usage: SEC_CONTACT_EMAIL=you@example.com python scripts/inspect_edgar.py SNDK [MU ...] [--as-of 2026-10-30]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smart_beta_etf_builder as sb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    a = ap.parse_args()
    d = pd.Timestamp(a.as_of)
    empty = pd.DataFrame(index=[d], columns=a.tickers, dtype=float)
    ed = sb.EdgarFundamentals(sb.CFG, empty, empty.fillna(0.0))
    ed.load(a.tickers)
    for t in a.tickers:
        print(f"\n===== {t} as of {d.date()} =====")
        for item, df in ed._facts.get(t, {}).items():
            known = df[df["filed"] <= d].sort_values(["end", "priority"], ascending=[False, True]).head(4)
            print(f"-- {item}")
            print(known.to_string(index=False))


if __name__ == "__main__":
    main()
