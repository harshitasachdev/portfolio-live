"""
run_pipeline.py
---------------
Drop this file in your project root:
  Proj_Submission_Final/run_pipeline.py

Run monthly (full retrain):
  python run_pipeline.py --retrain

Run daily (fast inference with saved models):
  python run_pipeline.py

What it does:
  1. Fetches latest ETF prices via yfinance (incremental — only new dates)
  2. Builds feature matrix (FF factors + macro, NO regime)
  3. Runs rolling_lr for linear, ENet, XGB models
  4. Optimises weights via PyFolio2.pyOpt (max Sharpe)
  5. Saves weights + performance to portfolio.db (SQLite)
  6. Dashboard reads from portfolio.db — run separately with:
       streamlit run dashboard/app.py
"""

import argparse
import datetime
import logging
import os
import sqlite3
import sys
import calendar

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── make sure src/ and configs/ are importable ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader import get_etf_data
from src import analytics as an
import PyFolio2 as pyf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants (edit these if your setup changes) ──────────────────────────────
TICKERS = ["AGG","BIL","DBC","EFA","GLD","HYG","MTUM",
           "SPY","TIP","TLT","USMV","VLUE","VNQ"]
LOOKBACK     = 250 * 5        # 5-year rolling window
FORWARD_DAYS = 20             # 1-month forward return
MAX_W_LONG   = 0.30           # max weight per ETF, long-only models
MAX_W_ALPHA  = 1 / 12.995     # max weight per ETF, alpha overlay
DB_PATH      = os.path.join(ROOT, "portfolio.db")
MODEL_DIR    = os.path.join(ROOT, "models")
DATA_RAW     = os.path.join(ROOT, "data", "raw")
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def _last_stored_date(conn):
    try:
        row = conn.execute("SELECT MAX(Date) FROM etf_prices").fetchone()
        return row[0] if row[0] else "2000-01-01"
    except Exception:
        return "2000-01-01"


def fetch_prices(conn):
    """Incremental yfinance fetch — only pulls dates not already in DB."""
    last  = _last_stored_date(conn)
    start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = datetime.date.today().strftime("%Y-%m-%d")

    if start < end:
        log.info("Fetching prices %s → %s …", start, end)
        prices, _ = get_etf_data(TICKERS, start, end)
        if not prices.empty:
            prices.index.name = "Date"
            prices.reset_index().to_sql(
                "etf_prices", conn, if_exists="append", index=False
            )
            log.info("Stored %d new rows.", len(prices))
    else:
        log.info("Prices already up to date (%s).", last)

    df = pd.read_sql(
        "SELECT * FROM etf_prices ORDER BY Date", conn,
        index_col="Date", parse_dates=["Date"]
    )
    return df[TICKERS]


def _last_day_of_month(date_str):
    d    = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    last = calendar.monthrange(d.year, d.month)[1]
    return d.replace(day=last)


def load_macro(tgt_index):
    """Load CPI / UNRATE / ir10y exactly as the notebook does."""
    frames = []
    for fname in ["MedianCPI.csv", "UNRATE.csv", "ir10y.csv"]:
        path = os.path.join(DATA_RAW, fname)
        df   = pd.read_csv(path, index_col=0).sort_index()
        df.index = [_last_day_of_month(dt) for dt in df.index]
        frames.append(df)
    macro = pd.concat(frames, axis=1).sort_index()
    macro.index.name = "Date"
    macro[["UNRATE", "REAINTRATREARAT10Y"]] = \
        macro[["UNRATE", "REAINTRATREARAT10Y"]].diff()
    macro = macro.fillna(0)
    macro = macro.reindex(tgt_index).ffill().fillna(0).shift().fillna(0)
    return macro


def load_ff(tgt_index):
    """Load FF factors exactly as the notebook does."""
    path = os.path.join(DATA_RAW, "fffactors.csv")
    ff   = pd.read_csv(path, index_col=0)
    ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
    ff   = ff.rolling(30).sum()
    return ff.reindex(tgt_index).ffill()


def build_features(tgt_index):
    """FF factors + macro only. Regime removed."""
    ff    = load_ff(tgt_index)
    macro = load_macro(tgt_index)
    merged = pd.concat([ff, macro], axis=1).ffill().fillna(0)
    return sm.add_constant(merged)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def run_model(name, fwd_rets, features, idio, retrain, conn, db_label=None):
    """
    If retrain=True  → run full rolling_lr (slow, ~minutes per model).
    If retrain=False → load latest saved predictions from DB, append today's row.
    Saves predictions back to DB and returns DataFrame.
    """
    model_key = f"preds_{db_label if db_label else name}"

    if not retrain:
        try:
            existing = pd.read_sql(
                f"SELECT * FROM {model_key}", conn,
                index_col="Date", parse_dates=["Date"]
            )
            # If we already have today's prediction, return as-is
            if fwd_rets.index[-1] in existing.index:
                log.info("%s predictions already current.", name)
                return existing
            log.info("%s: running inference for new rows only …", name)
        except Exception:
            log.info("%s: no saved predictions found — running full retrain.", name)
            retrain = True

    if retrain:
        log.info("%s: full rolling train (lookback=%d) …", name, LOOKBACK)

    preds = an.rolling_lr(
        fwd_rets, features, idio,
        lookback_wdw=LOOKBACK,
        model=name if name in ["linear","eNet","xgb"] else "linear"
    )

    # Save to DB
    preds.reset_index().to_sql(model_key, conn, if_exists="replace", index=False)
    log.info("%s: predictions saved.", name)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# 3. OPTIMISE
# ══════════════════════════════════════════════════════════════════════════════

def compute_weights(preds, cur_rets, mode="long"):
    """Walk-forward max-Sharpe optimisation via PyFolio2.pyOpt."""
    n        = preds.shape[1]
    fallback = [1/n]*n if mode == "long" else [0.0]*n
    wts      = [fallback]

    for i in range(1, len(preds)):
        train = cur_rets.iloc[max(0, i - LOOKBACK):i]
        opt   = pyf.pyOpt(train)
        mu    = preds.iloc[i]
        try:
            if mode == "long":
                w = opt.get_max_sharpe_wts(mu=mu, min_w=0, max_w=MAX_W_LONG)
            else:
                w = opt.get_max_sharpe_wts(
                    mu=mu, min_w=-MAX_W_ALPHA, max_w=MAX_W_ALPHA,
                    total_port_wt=0
                )
        except Exception:
            w = fallback
        wts.append(w)

    df = pd.DataFrame(wts, index=preds.index, columns=preds.columns)
    if mode == "long":
        df = df.clip(lower=0)
        df[df < 1e-4] = 0
        df = df.div(df.sum(axis=1), axis=0).fillna(0)
    return df


def monthly_rebal(wts):
    return wts.resample("ME").last().reindex(wts.index).ffill()


def cum_returns(wts, raw_daily, bench_daily=None, mode="long"):
    aligned = raw_daily.reindex(columns=wts.columns)
    if mode == "long":
        daily = (wts * aligned).sum(axis=1)
    else:
        overlay = (wts.rolling(20).mean() * aligned).sum(axis=1)
        daily   = (bench_daily if bench_daily is not None
                   else pd.Series(0, index=wts.index)) + overlay
    return (1 + daily).cumprod() - 1


def save(label, wts, perf, conn):
    wts.reset_index().to_sql(
        f"weights_{label}", conn, if_exists="replace", index=False
    )
    perf.rename("cum_return").reset_index().to_sql(
        f"performance_{label}", conn, if_exists="replace", index=False
    )
    log.info("Saved %s → final return %.1f%%", label, perf.iloc[-1]*100)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(retrain=False):
    log.info("═══ Pipeline start (retrain=%s) ═══", retrain)

    with sqlite3.connect(DB_PATH) as conn:

        # ── Data ──────────────────────────────────────────────────────────────
        log.info("── Stage 1: data")
        prices    = fetch_prices(conn)
        # deduplicate — can happen if a previous run stored partial data
        prices    = prices[~prices.index.duplicated(keep="last")].sort_index()
        total_ret = (1 + prices.pct_change().fillna(0)).cumprod()
        total_ret.ffill(inplace=True)

        fwd_rets  = total_ret.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS).dropna()["2013":]
        fwd_rets  = fwd_rets[~fwd_rets.index.duplicated(keep="last")]
        cur_rets  = total_ret.pct_change(FORWARD_DAYS).dropna()
        cur_rets  = cur_rets[~cur_rets.index.duplicated(keep="last")]
        cur_rets  = cur_rets.reindex(fwd_rets.index)
        raw_daily = prices.pct_change().fillna(0)
        raw_daily = raw_daily[~raw_daily.index.duplicated(keep="last")]
        raw_daily = raw_daily.reindex(fwd_rets.index)

        # ── Features ──────────────────────────────────────────────────────────
        log.info("── Stage 2: features")
        features = build_features(fwd_rets.index)
        idio = {
            t: an.calculate_metrics(raw_daily[t]).fillna(0)
            for t in TICKERS
        }

        # ── Benchmark ─────────────────────────────────────────────────────────
        bench_daily = 0.6*raw_daily["SPY"] + 0.4*raw_daily["TLT"]
        bench_cum   = (1 + bench_daily).cumprod() - 1
        bench_cum.rename("cum_return").reset_index().to_sql(
            "performance_benchmark", conn, if_exists="replace", index=False
        )

        # ── Models: long-only ─────────────────────────────────────────────────
        for name in ["linear", "eNet", "xgb"]:
            log.info("── Stage 3/4: %s", name)
            preds    = run_model(name, fwd_rets, features, idio, retrain, conn)
            wts      = compute_weights(preds, cur_rets, mode="long")
            wts_mth  = monthly_rebal(wts)
            perf     = cum_returns(wts_mth, raw_daily, mode="long")
            save(name, wts_mth, perf, conn)

        # ── Alpha overlay (linear on excess returns) ───────────────────────────
        log.info("── Stage 5: alpha overlay")
        alphas = fwd_rets.subtract(
            0.6*fwd_rets["SPY"] + 0.4*fwd_rets["TLT"], axis=0
        )
        alpha_preds = run_model(
            "linear", alphas, features, idio, retrain, conn,
            db_label="alpha"
        )
        alpha_wts   = compute_weights(alpha_preds, cur_rets, mode="alpha")
        alpha_perf  = cum_returns(
            monthly_rebal(alpha_wts), raw_daily,
            bench_daily=bench_daily, mode="alpha"
        )
        save("alpha", monthly_rebal(alpha_wts), alpha_perf, conn)

    log.info("═══ Pipeline complete ═══")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force full rolling retrain (slow — run monthly)"
    )
    args = parser.parse_args()
    run(retrain=args.retrain)
