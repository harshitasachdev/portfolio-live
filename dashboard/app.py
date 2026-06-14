"""
dashboard/app.py
----------------
Drop this file into:
  Proj_Submission_Final/dashboard/app.py

Run with:
  streamlit run dashboard/app.py
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portfolio.db")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
)

TICKERS = ["AGG","BIL","DBC","EFA","GLD","HYG","MTUM",
           "SPY","TIP","TLT","USMV","VLUE","VNQ"]
COLOURS = {
    "linear":    "#3B5BDB",
    "eNet":      "#0CA678",
    "xgb":       "#F76707",
    "alpha":     "#AE3EC9",
    "benchmark": "#868E96",
}

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load(table):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(
                f"SELECT * FROM {table}", conn,
                index_col="Date", parse_dates=["Date"]
            )
    except Exception:
        return None


def perf(label):
    df = load(f"performance_{label}")
    return df["cum_return"] if df is not None else None


def wts(label):
    return load(f"weights_{label}")


def daily_rets(cum):
    return cum.pct_change().fillna(0)


def sharpe(rets, ann=252):
    return (rets.mean() / rets.std()) * np.sqrt(ann) if rets.std() > 0 else 0.0


def max_dd(cum):
    return ((1 + cum) / (1 + cum).cummax() - 1).min()


def capm(port_r, bench_r):
    common = port_r.index.intersection(bench_r.index)
    X = sm.add_constant(bench_r.loc[common])
    res = sm.OLS(port_r.loc[common], X).fit()
    return res.params.iloc[0], res.params.iloc[1]


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Controls")
model = st.sidebar.selectbox("Model", ["alpha", "linear", "eNet", "xgb"], index=0)
dates = st.sidebar.date_input(
    "Date range",
    value=[pd.Timestamp("2018-01-01").date(), pd.Timestamp.today().date()]
)
roll_win = st.sidebar.slider("Rolling window (days)", 30, 252, 63)

# ── load ──────────────────────────────────────────────────────────────────────

p_model = perf(model)
p_bench = perf("benchmark")

if p_model is None:
    st.warning(
        "No data found in portfolio.db.  \n"
        "Run `python run_pipeline.py --retrain` first, then refresh."
    )
    st.stop()

start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
p_model = p_model.loc[start:end]
p_bench = p_bench.loc[start:end] if p_bench is not None else None

r_model = daily_rets(p_model)
r_bench = daily_rets(p_bench) if p_bench is not None else None

# ── header ────────────────────────────────────────────────────────────────────

st.title("📈 Portfolio dashboard")
st.caption(
    f"Model: **{model.upper()}** · "
    f"Last data point: {p_model.index[-1].strftime('%d %b %Y')}"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total return",    f"{p_model.iloc[-1]*100:.1f}%")
c2.metric("Ann. volatility", f"{r_model.std()*np.sqrt(252)*100:.1f}%")
c3.metric("Sharpe ratio",    f"{sharpe(r_model):.2f}")
c4.metric("Max drawdown",    f"{max_dd(p_model)*100:.1f}%")
if r_bench is not None:
    alpha_v, beta_v = capm(r_model, r_bench)
    c5.metric("Jensen α (ann.)", f"{alpha_v*252*100:.2f}%")

st.divider()

# ── cumulative returns ────────────────────────────────────────────────────────

st.subheader("Cumulative returns")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=p_model.index, y=p_model*100,
    name=model.upper(),
    line=dict(color=COLOURS[model], width=2)
))
if p_bench is not None:
    fig.add_trace(go.Scatter(
        x=p_bench.index, y=p_bench*100,
        name="60/40 benchmark",
        line=dict(color=COLOURS["benchmark"], width=1.5, dash="dash")
    ))
fig.update_layout(
    yaxis_title="Cumulative return (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02),
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ── drawdown ──────────────────────────────────────────────────────────────────

st.subheader("Drawdown")
dd_series = ((1 + p_model) / (1 + p_model).cummax() - 1) * 100
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dd_series.index, y=dd_series,
    fill="tozeroy", fillcolor="rgba(224,49,49,0.12)",
    line=dict(color="#c92a2a", width=1),
    name="Drawdown"
))
fig_dd.update_layout(
    yaxis_title="Drawdown (%)",
    hovermode="x unified",
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_dd, use_container_width=True)

# ── rolling Sharpe + vol ──────────────────────────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.subheader(f"Rolling Sharpe ({roll_win}d)")
    rs = (r_model.rolling(roll_win).mean() /
          r_model.rolling(roll_win).std()) * np.sqrt(252)
    fig_sh = px.line(rs, labels={"value": "Sharpe", "Date": ""})
    fig_sh.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sh.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_sh, use_container_width=True)

with col_r:
    st.subheader(f"Rolling volatility ({roll_win}d, ann.)")
    rv = r_model.rolling(roll_win).std() * np.sqrt(252) * 100
    fig_vol = px.line(rv, labels={"value": "Volatility (%)", "Date": ""})
    fig_vol.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_vol, use_container_width=True)

# ── CAPM scatter ──────────────────────────────────────────────────────────────

if r_bench is not None:
    st.subheader("CAPM — portfolio vs 60/40")
    rb = r_bench.reindex(r_model.index).fillna(0)
    a, b = capm(r_model, rb)
    x_line = np.linspace(rb.min(), rb.max(), 100)
    y_line  = a + b * x_line

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=rb, y=r_model, mode="markers",
        marker=dict(size=3, color=COLOURS[model], opacity=0.4),
        name="Daily returns"
    ))
    fig_c.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#c92a2a", width=2),
        name=f"α={a*252*100:.2f}% ann   β={b:.2f}"
    ))
    fig_c.update_layout(
        xaxis_title="Benchmark return",
        yaxis_title="Portfolio return",
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_c, use_container_width=True)

# ── weights ───────────────────────────────────────────────────────────────────

w = wts(model)
if w is not None:
    st.subheader("Portfolio weights over time")
    w_filtered = w.loc[start:end]
    w_mth = w_filtered.resample("ME").last().dropna(how="all")

    colours_list = px.colors.qualitative.D3
    fig_w = go.Figure()
    for i, t in enumerate(w_mth.columns):
        fig_w.add_trace(go.Bar(
            x=w_mth.index, y=w_mth[t],
            name=t,
            marker_color=colours_list[i % len(colours_list)]
        ))
    fig_w.update_layout(
        barmode="stack",
        yaxis_title="Weight",
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Latest weights")
    latest = w.iloc[-1].sort_values(ascending=False)
    latest = latest[latest.abs() > 1e-4]
    st.dataframe(
        latest.rename("Weight").to_frame().style.format("{:.2%}"),
        use_container_width=True
    )

# ── model comparison ──────────────────────────────────────────────────────────

st.divider()
st.subheader("All models vs benchmark")
fig_all = go.Figure()
for m in ["linear", "eNet", "xgb", "alpha"]:
    p = perf(m)
    if p is not None:
        p = p.loc[start:end]
        fig_all.add_trace(go.Scatter(
            x=p.index, y=p*100,
            name=m.upper(),
            line=dict(color=COLOURS[m], width=1.8)
        ))
if p_bench is not None:
    fig_all.add_trace(go.Scatter(
        x=p_bench.index, y=p_bench*100,
        name="60/40",
        line=dict(color=COLOURS["benchmark"], dash="dash", width=1.5)
    ))
fig_all.update_layout(
    yaxis_title="Cumulative return (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02),
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_all, use_container_width=True)

st.caption("Data: yfinance · Models retrain monthly via `python run_pipeline.py --retrain`")
