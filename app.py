import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
from nsepython import nse_optionchain_scrapper
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

# ---------------------------
# Auto Refresh (30 sec)
# ---------------------------
st_autorefresh(interval=30000, key="refresh")

# ---------------------------
# Header + Manual Refresh
# ---------------------------
col_title, col_btn = st.columns([6, 1])

with col_title:
    st.title("📊 NIFTY Live Sentiment Dashboard")

with col_btn:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

st.caption("⏱ Auto-refreshes every 30 seconds")
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ---------------------------
# Fetch Data (cached)
# ---------------------------
@st.cache_data(ttl=300)
def load_data():
    return yf.download("^NSEI", period="5d", interval="5m")

data = load_data()

if data.empty:
    st.error("Failed to fetch market data.")
    st.stop()

# ---------------------------
# Indicators
# ---------------------------
data['EMA20'] = data['Close'].ewm(span=20).mean()
data['EMA50'] = data['Close'].ewm(span=50).mean()

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data['RSI'] = rsi(data)

data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
data['VWAP'] = (data['TP'] * data['Volume']).cumsum() / data['Volume'].cumsum()

# ATR (Volatility)
data['H-L'] = data['High'] - data['Low']
data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
data['ATR'] = data['TR'].rolling(14).mean()

# ---------------------------
# PCR (cached)
# ---------------------------
@st.cache_data(ttl=300)
def get_pcr():
    try:
        oc = nse_optionchain_scrapper("NIFTY")
        total_ce_oi = 0
        total_pe_oi = 0

        for item in oc['records']['data']:
            if 'CE' in item and 'PE' in item:
                total_ce_oi += item['CE']['openInterest']
                total_pe_oi += item['PE']['openInterest']

        return total_pe_oi / total_ce_oi if total_ce_oi != 0 else None
    except:
        return None

pcr = get_pcr()

# ---------------------------
# Latest Data
# ---------------------------
latest = data.iloc[-1]

# ---------------------------
# Market Regime
# ---------------------------
recent_high = data['High'].rolling(20).max().iloc[-1]
recent_low = data['Low'].rolling(20).min().iloc[-1]

range_pct = (recent_high - recent_low) / latest['Close']

if range_pct < 0.005:
    regime = "SIDEWAYS"
else:
    regime = "TRENDING"

# ---------------------------
# Breakout Logic
# ---------------------------
prev_high = data['High'].iloc[-2]
prev_low = data['Low'].iloc[-2]

bullish_breakout = latest['Close'] > prev_high
bearish_breakdown = latest['Close'] < prev_low

# ---------------------------
# Signal Logic
# ---------------------------
score = 0
reasons = []

# EMA
if latest['EMA20'] > latest['EMA50']:
    score += 2
    reasons.append("Trend bullish (EMA20 > EMA50)")
else:
    score -= 2
    reasons.append("Trend bearish")

# RSI
if latest['RSI'] > 55:
    score += 1
    reasons.append("Momentum strong")
elif latest['RSI'] < 45:
    score -= 1
    reasons.append("Momentum weak")

# VWAP
if latest['Close'] > latest['VWAP']:
    score += 2
    reasons.append("Above VWAP")
else:
    score -= 2
    reasons.append("Below VWAP")

# PCR
if pcr is not None:
    if pcr > 1:
        score += 1
        reasons.append("PCR bullish")
    elif pcr < 0.8:
        score -= 1
        reasons.append("PCR bearish")

# ---------------------------
# Filters
# ---------------------------
filters_passed = True

if regime == "SIDEWAYS":
    filters_passed = False
    reasons.append("Sideways market")

if latest['ATR'] < latest['Close'] * 0.002:
    filters_passed = False
    reasons.append("Low volatility")

if score > 0 and not bullish_breakout:
    filters_passed = False
    reasons.append("No bullish breakout")

if score < 0 and not bearish_breakdown:
    filters_passed = False
    reasons.append("No bearish breakdown")

# ---------------------------
# Final Signal
# ---------------------------
if not filters_passed:
    signal = "⚪ NO TRADE"
elif score >= 4:
    signal = "🟢 STRONG BULLISH"
elif score >= 2:
    signal = "🟢 BULLISH"
elif score <= -4:
    signal = "🔴 STRONG BEARISH"
elif score <= -2:
    signal = "🔴 BEARISH"
else:
    signal = "⚪ NO TRADE"

confidence = abs(score) / 6

# ---------------------------
# UI
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📢 Signal")
    st.markdown(f"## {signal}")
    st.metric("Confidence", f"{confidence*100:.0f}%")

    st.metric("Close", f"{latest['Close']:.2f}")
    st.metric("RSI", f"{latest['RSI']:.2f}")
    st.metric("PCR", f"{pcr:.2f}" if pcr else "N/A")

with col2:
    st.subheader("📊 Market Context")
    st.write(f"Regime: {regime}")
    st.write(f"ATR: {latest['ATR']:.2f}")
    st.write(f"EMA20: {latest['EMA20']:.2f}")
    st.write(f"EMA50: {latest['EMA50']:.2f}")
    st.write(f"VWAP: {latest['VWAP']:.2f}")

# ---------------------------
# Reasons
# ---------------------------
st.subheader("🧠 Why this signal?")
for r in reasons:
    st.write(f"• {r}")

# ---------------------------
# Chart
# ---------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
))

fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], name="EMA20"))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], name="EMA50"))
fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name="VWAP"))

st.plotly_chart(fig, use_container_width=True)
