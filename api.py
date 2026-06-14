"""
api.py
------
FastAPI server — reads from portfolio.db and serves JSON to the Lovable frontend.

Endpoints:
  GET /performance          → all models + benchmark cumulative returns
  GET /weights/{model}      → latest weights for a model
  GET /weights/{model}/history → weight history over time
  GET /metrics              → Sharpe, drawdown, total return per model
  GET /health               → ping
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.db")

app = FastAPI(title="Portfolio API")

# Allow Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)


def load_perf(label: str) -> pd.Series | None:
    try:
        with get_conn() as conn:
            df = pd.read_sql(
                f"SELECT * FROM performance_{label} ORDER BY Date",
                conn, index_col="Date", parse_dates=["Date"]
            )
        return df["cum_return"]
    except Exception:
        return None


def load_wts(label: str) -> pd.DataFrame | None:
    try:
        with get_conn() as conn:
            df = pd.read_sql(
                f"SELECT * FROM weights_{label} ORDER BY Date",
                conn, index_col="Date", parse_dates=["Date"]
            )
        return df
    except Exception:
        return None


def sharpe(rets: pd.Series, ann: int = 252) -> float:
    return float((rets.mean() / rets.std()) * np.sqrt(ann)) if rets.std() > 0 else 0.0


def max_dd(cum: pd.Series) -> float:
    return float(((1 + cum) / (1 + cum).cummax() - 1).min())


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/performance")
def performance(start: str = "2018-01-01", end: str = None):
    """Returns cumulative returns for all models + benchmark as time series."""
    result = {}
    for label in ["linear", "eNet", "xgb", "alpha", "benchmark"]:
        p = load_perf(label)
        if p is None:
            continue
        if start:
            p = p[p.index >= start]
        if end:
            p = p[p.index <= end]
        result[label] = {
            "dates":   [d.strftime("%Y-%m-%d") for d in p.index],
            "values":  [round(v * 100, 4) for v in p.values],
        }
    return result


@app.get("/weights/{model}")
def latest_weights(model: str):
    """Returns the most recent portfolio weights for a model."""
    allowed = ["linear", "eNet", "xgb", "alpha"]
    if model not in allowed:
        raise HTTPException(status_code=404, detail=f"Model must be one of {allowed}")
    w = load_wts(model)
    if w is None:
        raise HTTPException(status_code=404, detail="No weights found")
    latest = w.iloc[-1]
    latest = latest[latest.abs() > 1e-4].sort_values(ascending=False)
    return {
        "model":   model,
        "date":    w.index[-1].strftime("%Y-%m-%d"),
        "weights": {t: round(float(v), 6) for t, v in latest.items()},
    }


@app.get("/weights/{model}/history")
def weight_history(model: str, start: str = "2018-01-01"):
    """Returns monthly weight history for stacked bar chart."""
    allowed = ["linear", "eNet", "xgb", "alpha"]
    if model not in allowed:
        raise HTTPException(status_code=404, detail=f"Model must be one of {allowed}")
    w = load_wts(model)
    if w is None:
        raise HTTPException(status_code=404, detail="No weights found")
    w = w[w.index >= start]
    w_mth = w.resample("ME").last().dropna(how="all")
    return {
        "dates":   [d.strftime("%Y-%m-%d") for d in w_mth.index],
        "tickers": w_mth.columns.tolist(),
        "weights": {
            t: [round(float(v), 6) for v in w_mth[t].values]
            for t in w_mth.columns
        },
    }


@app.get("/metrics")
def metrics(start: str = "2018-01-01"):
    """Returns key metrics for all models."""
    result = {}
    for label in ["linear", "eNet", "xgb", "alpha", "benchmark"]:
        p = load_perf(label)
        if p is None:
            continue
        p = p[p.index >= start]
        r = p.pct_change().fillna(0)
        result[label] = {
            "total_return":    round(float(p.iloc[-1]) * 100, 2),
            "sharpe":          round(sharpe(r), 3),
            "max_drawdown":    round(max_dd(p) * 100, 2),
            "ann_volatility":  round(float(r.std() * np.sqrt(252)) * 100, 2),
            "last_updated":    p.index[-1].strftime("%Y-%m-%d"),
        }
    return result
