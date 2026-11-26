"""
Refresh data/sp500_membership.csv.gz from github.com/fja05680/sp500
(historical S&P 500 constituents, one row per index change since 1996).

Keeps rows from 2012 onward (plus the list in force on 2012-01-01) so the
file stays small. If the download fails or looks wrong, the committed copy is
left untouched and the refresh carries on with it.

Usage: python scripts/update_membership.py
"""
import csv
import gzip
import io
import os
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
       "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "sp500_membership.csv.gz")
KEEP_FROM = "2012-01-01"


def main():
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            text = r.read().decode("utf-8")
    except Exception as e:
        print(f"[warn] membership download failed ({e}); keeping committed copy")
        return 0

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or set(rows[0]) != {"date", "tickers"} or len(rows[-1]["tickers"].split(",")) < 480:
        print("[warn] downloaded membership file looks wrong; keeping committed copy")
        return 0

    keep = [r for r in rows if r["date"] >= KEEP_FROM]
    prior = [r for r in rows if r["date"] < KEEP_FROM][-1:]
    with gzip.GzipFile(OUT, "wb", mtime=0) as gz, io.TextIOWrapper(gz, newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tickers"])
        for r in prior + keep:
            w.writerow([r["date"], r["tickers"]])
    print(f"Membership history updated: {len(prior + keep)} rows, latest change {keep[-1]['date']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
