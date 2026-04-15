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
# Auto Refresh
# ---------------------------
auto_refresh = st.toggle("🔄 Auto Refresh", value=True)
if auto_refresh:
    st_autorefresh(interval=30000, key="refresh")

# ---------------------------
# Header
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
# Fetch Data
# ---------------------------
@st.cache_data(ttl=300)
def load_data():
    return yf.download("^NSEI", period="5d", interval="5m")

data = load_data()

if data.empty:
    st.error("Failed to fetch market data.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna(subset=['Close'])

if len(data) < 20:
    st.warning("Not enough data yet...")
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
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)

cum_vol = data['Volume'].cumsum()

data['VWAP'] = np.where(
    cum_vol != 0,
    (data['TP'] * data['Volume']).cumsum() / cum_vol,
    np.nan
)

# ATR
data['H-L'] = data['High'] - data['Low']
data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
data['ATR'] = data['TR'].rolling(14).mean()

# ---------------------------
# PCR
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
prev = data.iloc[-2]

# 5-min change
price_now = latest['Close']
price_5m_ago = prev['Close']

change_5m = price_now - price_5m_ago
change_5m_pct = (change_5m / price_5m_ago) * 100 if price_5m_ago != 0 else 0

# ---------------------------
# Candle Analysis
# ---------------------------
o, c, h, l = latest['Open'], latest['Close'], latest['High'], latest['Low']
prev_o, prev_c = prev['Open'], prev['Close']

body = abs(c - o)
range_candle = h - l
upper_wick = h - max(o, c)
lower_wick = min(o, c) - l

# Candle Type
if body < range_candle * 0.2:
    candle_type = "Doji (Indecision)"
elif c > o and body > range_candle * 0.6:
    candle_type = "Strong Bullish"
elif c < o and body > range_candle * 0.6:
    candle_type = "Strong Bearish"
elif lower_wick > body * 2 and upper_wick < body:
    candle_type = "Hammer"
elif upper_wick > body * 2 and lower_wick < body:
    candle_type = "Shooting Star"
else:
    candle_type = "Normal"

# Patterns
pattern = None
if (prev_c < prev_o) and (c > o) and (c > prev_o) and (o < prev_c):
    pattern = "Bullish Engulfing"
elif (prev_c > prev_o) and (c < o) and (o > prev_c) and (c < prev_o):
    pattern = "Bearish Engulfing"

# Context
trend = "UPTREND" if latest['EMA20'] > latest['EMA50'] else "DOWNTREND"
context_insight = []

if candle_type == "Strong Bullish" and trend == "UPTREND":
    context_insight.append("Continuation: buyers strong")

if candle_type == "Strong Bearish" and trend == "DOWNTREND":
    context_insight.append("Continuation: sellers strong")

if "Hammer" in candle_type and trend == "DOWNTREND":
    context_insight.append("Possible reversal up")

if "Shooting Star" in candle_type and trend == "UPTREND":
    context_insight.append("Possible reversal down")

if c > latest['VWAP'] and "Bullish" in candle_type:
    context_insight.append("VWAP confirms bullish bias")

if c < latest['VWAP'] and "Bearish" in candle_type:
    context_insight.append("VWAP confirms bearish bias")

if pattern:
    context_insight.append(pattern)

if not context_insight:
    context_insight.append("No strong signal")

# ---------------------------
# Candle Numeric Breakdown
# ---------------------------
body_pct = (body / range_candle) * 100 if range_candle else 0
upper_wick_pct = (upper_wick / range_candle) * 100 if range_candle else 0
lower_wick_pct = (lower_wick / range_candle) * 100 if range_candle else 0

price_change = c - o
price_change_pct = (price_change / o) * 100 if o else 0
volatility_pct = (range_candle / c) * 100 if c else 0

if body_pct > 70:
    strength = "🔥 Very Strong"
elif body_pct > 50:
    strength = "💪 Strong"
elif body_pct > 30:
    strength = "⚖️ Moderate"
else:
    strength = "😐 Weak"

# ---------------------------
# Market Logic
# ---------------------------
recent_high = data['High'].rolling(20).max().iloc[-1]
recent_low = data['Low'].rolling(20).min().iloc[-1]

range_pct = (recent_high - recent_low) / latest['Close']
regime = "SIDEWAYS" if range_pct < 0.005 else "TRENDING"

prev_high = data['High'].iloc[-2]
prev_low = data['Low'].iloc[-2]

bullish_breakout = latest['Close'] > prev_high
bearish_breakdown = latest['Close'] < prev_low

score = 0
reasons = []

if latest['EMA20'] > latest['EMA50']:
    score += 2
    reasons.append("Trend bullish")
else:
    score -= 2
    reasons.append("Trend bearish")

if latest['RSI'] > 55:
    score += 1
    reasons.append("Momentum strong")
elif latest['RSI'] < 45:
    score -= 1
    reasons.append("Momentum weak")

if latest['Close'] > latest['VWAP']:
    score += 2
    reasons.append("Above VWAP")
else:
    score -= 2
    reasons.append("Below VWAP")

if pcr is not None:
    if pcr > 1:
        score += 1
        reasons.append("PCR bullish")
    elif pcr < 0.8:
        score -= 1
        reasons.append("PCR bearish")

filters_passed = True

if regime == "SIDEWAYS":
    filters_passed = False
    reasons.append("Sideways")

if latest['ATR'] < latest['Close'] * 0.002:
    filters_passed = False
    reasons.append("Low volatility")

if score > 0 and not bullish_breakout:
    filters_passed = False

if score < 0 and not bearish_breakdown:
    filters_passed = False

# Final Signal
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
    st.metric("Close", f"{price_now:.2f}")
    st.metric("5m Change", f"{change_5m:.2f} ({change_5m_pct:.2f}%)")
    st.metric("RSI", f"{latest['RSI']:.2f}")
    st.metric("PCR", f"{pcr:.2f}" if pcr else "N/A")

with col2:
    st.subheader("📊 Market Context")
    st.write(f"Regime: {regime}")
    st.write(f"ATR: {latest['ATR']:.2f}")
    st.write(f"EMA20: {latest['EMA20']:.2f}")
    st.write(f"EMA50: {latest['EMA50']:.2f}")
    st.write(f"VWAP: {latest['VWAP']:.2f}")

# Candle Analysis UI
st.subheader("🧠 Candle Analysis")
st.write(f"Type: {candle_type}")
if pattern:
    st.write(f"Pattern: {pattern}")
for i in context_insight:
    st.write(f"• {i}")

# Candle Quant UI
st.subheader("📊 Candle Breakdown")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Body %", f"{body_pct:.1f}%")
    st.metric("Price Change", f"{price_change:.2f}")
    st.metric("Change %", f"{price_change_pct:.2f}%")

with colB:
    st.metric("Upper Wick %", f"{upper_wick_pct:.1f}%")
    st.metric("Lower Wick %", f"{lower_wick_pct:.1f}%")
    st.metric("Volatility %", f"{volatility_pct:.2f}%")

with colC:
    st.metric("Strength", strength)
    st.metric("High", f"{h:.2f}")
    st.metric("Low", f"{l:.2f}")

# Reasons
st.subheader("🧠 Why this signal?")
for r in reasons:
    st.write(f"• {r}")

# Chart
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
))

fig.add_trace(go.Scatter(x=data.index, y=data['EMA20']))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50']))
fig.add_trace(go.Scatter(x=data.index, y=data['VWAP']))

st.plotly_chart(fig, use_container_width=True)

































# import streamlit as st
# from streamlit_autorefresh import st_autorefresh
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from nsepython import nse_optionchain_scrapper
# import plotly.graph_objects as go
# from datetime import datetime

# st.set_page_config(layout="wide")

# # ---------------------------
# # Auto Refresh (30 sec)
# # ---------------------------
# auto_refresh = st.toggle("🔄 Auto Refresh", value=True)
# if auto_refresh:
#     st_autorefresh(interval=30000, key="refresh")

# # ---------------------------
# # Header + Manual Refresh
# # ---------------------------
# col_title, col_btn = st.columns([6, 1])

# with col_title:
#     st.title("📊 NIFTY Live Sentiment Dashboard")

# with col_btn:
#     if st.button("🔄 Refresh"):
#         st.cache_data.clear()
#         st.rerun()

# st.caption("⏱ Auto-refreshes every 30 seconds")
# st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# # ---------------------------
# # Fetch Data (cached)
# # ---------------------------
# @st.cache_data(ttl=300)
# def load_data():
#     return yf.download("^NSEI", period="5d", interval="5m")

# data = load_data()

# if data.empty:
#     st.error("Failed to fetch market data.")
#     st.stop()

# if isinstance(data.columns, pd.MultiIndex):
#     data.columns = data.columns.get_level_values(0)

# data = data.dropna(subset=['Close'])

# if len(data) < 20:
#     st.warning("Not enough data yet...")
#     st.stop()

# # ---------------------------
# # Indicators
# # ---------------------------
# data['EMA20'] = data['Close'].ewm(span=20).mean()
# data['EMA50'] = data['Close'].ewm(span=50).mean()

# def rsi(df, period=14):
#     delta = df['Close'].diff()
#     gain = delta.clip(lower=0)
#     loss = -delta.clip(upper=0)
#     avg_gain = gain.rolling(period).mean()
#     avg_loss = loss.rolling(period).mean()
#     rs = avg_gain / avg_loss
#     return 100 - (100 / (1 + rs))

# data['RSI'] = rsi(data)

# data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
# data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
# data['Volume'] = data['Volume'].fillna(0)

# cum_vol = data['Volume'].cumsum()

# data['VWAP'] = np.where(
#     cum_vol != 0,
#     (data['TP'] * data['Volume']).cumsum() / cum_vol,
#     np.nan
# )

# # ATR
# data['H-L'] = data['High'] - data['Low']
# data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
# data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
# data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
# data['ATR'] = data['TR'].rolling(14).mean()

# # ---------------------------
# # PCR
# # ---------------------------
# @st.cache_data(ttl=300)
# def get_pcr():
#     try:
#         oc = nse_optionchain_scrapper("NIFTY")
#         total_ce_oi = 0
#         total_pe_oi = 0

#         for item in oc['records']['data']:
#             if 'CE' in item and 'PE' in item:
#                 total_ce_oi += item['CE']['openInterest']
#                 total_pe_oi += item['PE']['openInterest']

#         return total_pe_oi / total_ce_oi if total_ce_oi != 0 else None
#     except:
#         return None

# pcr = get_pcr()

# # ---------------------------
# # Latest Data
# # ---------------------------
# latest = data.iloc[-1]

# prev_5m = data.iloc[-2]

# price_now = latest['Close']
# price_5m_ago = prev_5m['Close']

# change_5m = price_now - price_5m_ago
# change_5m_pct = (change_5m / price_5m_ago) * 100 if price_5m_ago != 0 else 0

# # ===========================
# # ✅ ADVANCED CANDLE ANALYSIS (ADDED)
# # ===========================

# prev = data.iloc[-2]

# o = latest['Open']
# c = latest['Close']
# h = latest['High']
# l = latest['Low']

# prev_o = prev['Open']
# prev_c = prev['Close']

# body = abs(c - o)
# range_candle = h - l
# upper_wick = h - max(o, c)
# lower_wick = min(o, c) - l

# if body < range_candle * 0.2:
#     candle_type = "Doji (Indecision)"
# elif c > o and body > range_candle * 0.6:
#     candle_type = "Strong Bullish"
# elif c < o and body > range_candle * 0.6:
#     candle_type = "Strong Bearish"
# elif lower_wick > body * 2 and upper_wick < body:
#     candle_type = "Hammer (Bullish Reversal)"
# elif upper_wick > body * 2 and lower_wick < body:
#     candle_type = "Shooting Star (Bearish Reversal)"
# else:
#     candle_type = "Normal Candle"

# pattern = None

# if (prev_c < prev_o) and (c > o) and (c > prev_o) and (o < prev_c):
#     pattern = "Bullish Engulfing"
# elif (prev_c > prev_o) and (c < o) and (o > prev_c) and (c < prev_o):
#     pattern = "Bearish Engulfing"

# context_insight = []

# trend = "UPTREND" if latest['EMA20'] > latest['EMA50'] else "DOWNTREND"

# if candle_type == "Strong Bullish" and trend == "UPTREND":
#     context_insight.append("Strong continuation: buyers in control in uptrend")

# if candle_type == "Strong Bearish" and trend == "DOWNTREND":
#     context_insight.append("Strong continuation: sellers dominating")

# if "Hammer" in candle_type and trend == "DOWNTREND":
#     context_insight.append("Potential reversal: buyers stepping in after fall")

# if "Shooting Star" in candle_type and trend == "UPTREND":
#     context_insight.append("Potential reversal: sellers rejecting higher prices")

# if c > latest['VWAP'] and "Bullish" in candle_type:
#     context_insight.append("Bullish bias supported by VWAP")

# if c < latest['VWAP'] and "Bearish" in candle_type:
#     context_insight.append("Bearish bias supported by VWAP")

# if pattern == "Bullish Engulfing":
#     context_insight.append("Strong reversal signal: bullish engulfing")

# if pattern == "Bearish Engulfing":
#     context_insight.append("Strong reversal signal: bearish engulfing")

# if not context_insight:
#     context_insight.append("No strong contextual signal")

# # ===========================

# # ---------------------------
# # Candle Numeric Breakdown
# # ---------------------------

# body_pct = (body / range_candle) * 100 if range_candle != 0 else 0
# upper_wick_pct = (upper_wick / range_candle) * 100 if range_candle != 0 else 0
# lower_wick_pct = (lower_wick / range_candle) * 100 if range_candle != 0 else 0

# price_change = c - o
# price_change_pct = (price_change / o) * 100 if o != 0 else 0

# volatility_pct = (range_candle / c) * 100 if c != 0 else 0

# # Strength classification
# if body_pct > 70:
#     strength = "🔥 Very Strong"
# elif body_pct > 50:
#     strength = "💪 Strong"
# elif body_pct > 30:
#     strength = "⚖️ Moderate"
# else:
#     strength = "😐 Weak / Indecisive"



# # ---------------------------
# # Candle Numeric Dashboard
# # ---------------------------

# st.subheader("📊 Candle Breakdown (Quant View)")

# colA, colB, colC = st.columns(3)

# with colA:
#     st.metric("Body %", f"{body_pct:.1f}%")
#     st.metric("Price Change", f"{price_change:.2f}")
#     st.metric("Change %", f"{price_change_pct:.2f}%")

# with colB:
#     st.metric("Upper Wick %", f"{upper_wick_pct:.1f}%")
#     st.metric("Lower Wick %", f"{lower_wick_pct:.1f}%")
#     st.metric("Volatility %", f"{volatility_pct:.2f}%")

# with colC:
#     st.metric("Candle Strength", strength)
#     st.metric("High", f"{h:.2f}")
#     st.metric("Low", f"{l:.2f}")
#     st.metric("5-min Ago", f"{price_5m_ago:.2f}")
#     st.metric("Now", f"{price_now:.2f}")
#     st.metric("5-min Change", f"{change_5m:.2f} ({change_5m_pct:.2f}%)")



# # Market Regime
# recent_high = data['High'].rolling(20).max().iloc[-1]
# recent_low = data['Low'].rolling(20).min().iloc[-1]

# range_pct = (recent_high - recent_low) / latest['Close']

# regime = "SIDEWAYS" if range_pct < 0.005 else "TRENDING"

# # Breakout
# prev_high = data['High'].iloc[-2]
# prev_low = data['Low'].iloc[-2]

# bullish_breakout = latest['Close'] > prev_high
# bearish_breakdown = latest['Close'] < prev_low

# # Signal Logic
# score = 0
# reasons = []

# if latest['EMA20'] > latest['EMA50']:
#     score += 2
#     reasons.append("Trend bullish (EMA20 > EMA50)")
# else:
#     score -= 2
#     reasons.append("Trend bearish")

# if latest['RSI'] > 55:
#     score += 1
#     reasons.append("Momentum strong")
# elif latest['RSI'] < 45:
#     score -= 1
#     reasons.append("Momentum weak")

# if latest['Close'] > latest['VWAP']:
#     score += 2
#     reasons.append("Above VWAP")
# else:
#     score -= 2
#     reasons.append("Below VWAP")

# if pcr is not None:
#     if pcr > 1:
#         score += 1
#         reasons.append("PCR bullish")
#     elif pcr < 0.8:
#         score -= 1
#         reasons.append("PCR bearish")

# filters_passed = True

# if regime == "SIDEWAYS":
#     filters_passed = False
#     reasons.append("Sideways market")

# if latest['ATR'] < latest['Close'] * 0.002:
#     filters_passed = False
#     reasons.append("Low volatility")

# if score > 0 and not bullish_breakout:
#     filters_passed = False
#     reasons.append("No bullish breakout")

# if score < 0 and not bearish_breakdown:
#     filters_passed = False
#     reasons.append("No bearish breakdown")

# if not filters_passed:
#     signal = "⚪ NO TRADE"
# elif score >= 4:
#     signal = "🟢 STRONG BULLISH"
# elif score >= 2:
#     signal = "🟢 BULLISH"
# elif score <= -4:
#     signal = "🔴 STRONG BEARISH"
# elif score <= -2:
#     signal = "🔴 BEARISH"
# else:
#     signal = "⚪ NO TRADE"

# confidence = abs(score) / 6

# # UI
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("📢 Signal")
#     st.markdown(f"## {signal}")
#     st.metric("Confidence", f"{confidence*100:.0f}%")
#     st.metric("Close", f"{latest['Close']:.2f}")
#     st.metric("RSI", f"{latest['RSI']:.2f}")
#     st.metric("PCR", f"{pcr:.2f}" if pcr else "N/A")

# with col2:
#     st.subheader("📊 Market Context")
#     st.write(f"Regime: {regime}")
#     st.write(f"ATR: {latest['ATR']:.2f}")
#     st.write(f"EMA20: {latest['EMA20']:.2f}")
#     st.write(f"EMA50: {latest['EMA50']:.2f}")
#     st.write(f"VWAP: {latest['VWAP']:.2f}")

# # Reasons
# st.subheader("🧠 Why this signal?")
# for r in reasons:
#     st.write(f"• {r}")

# # ✅ Candle UI
# st.subheader("🧠 Advanced Candle Analysis")
# st.write(f"**Candle Type:** {candle_type}")
# if abs(change_5m_pct) < 0.1:
#     filters_passed = False
#     reasons.append("No momentum (5m change too small)")
    
# if pattern:
#     st.write(f"**Pattern:** {pattern}")
# st.write("**Context Insight:**")
# for i in context_insight:
#     st.write(f"• {i}")

# # Chart
# fig = go.Figure()

# fig.add_trace(go.Candlestick(
#     x=data.index,
#     open=data['Open'],
#     high=data['High'],
#     low=data['Low'],
#     close=data['Close']
# ))

# fig.add_trace(go.Scatter(x=data.index, y=data['EMA20']))
# fig.add_trace(go.Scatter(x=data.index, y=data['EMA50']))
# fig.add_trace(go.Scatter(x=data.index, y=data['VWAP']))

# st.plotly_chart(fig, use_container_width=True)

















# # import streamlit as st
# # from streamlit_autorefresh import st_autorefresh
# # import yfinance as yf
# # import pandas as pd
# # import numpy as np
# # from nsepython import nse_optionchain_scrapper
# # import plotly.graph_objects as go
# # from datetime import datetime

# # st.set_page_config(layout="wide")

# # # ---------------------------
# # # Auto Refresh (30 sec)
# # # ---------------------------
# # st_autorefresh(interval=30000, key="refresh")

# # # ---------------------------
# # # Header + Manual Refresh
# # # ---------------------------
# # col_title, col_btn = st.columns([6, 1])

# # with col_title:
# #     st.title("📊 NIFTY Live Sentiment Dashboard")

# # with col_btn:
# #     if st.button("🔄 Refresh"):
# #         st.cache_data.clear()
# #         st.rerun()

# # st.caption("⏱ Auto-refreshes every 30 seconds")
# # st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# # # ---------------------------
# # # Fetch Data (cached)
# # # ---------------------------
# # @st.cache_data(ttl=300)
# # def load_data():
# #     return yf.download("^NSEI", period="5d", interval="5m")

# # data = load_data()

# # if data.empty:
# #     st.error("Failed to fetch market data.")
# #     st.stop()

# # # ✅ FIX: Flatten columns (yfinance MultiIndex issue)
# # if isinstance(data.columns, pd.MultiIndex):
# #     data.columns = data.columns.get_level_values(0)

# # # ✅ FIX: Clean missing Close values
# # data = data.dropna(subset=['Close'])

# # # ✅ FIX 2: Prevent crash if not enough data
# # if len(data) < 20:
# #     st.warning("Not enough data yet...")
# #     st.stop()

# # # ---------------------------
# # # Indicators
# # # ---------------------------
# # data['EMA20'] = data['Close'].ewm(span=20).mean()
# # data['EMA50'] = data['Close'].ewm(span=50).mean()

# # def rsi(df, period=14):
# #     delta = df['Close'].diff()
# #     gain = delta.clip(lower=0)
# #     loss = -delta.clip(upper=0)
# #     avg_gain = gain.rolling(period).mean()
# #     avg_loss = loss.rolling(period).mean()
# #     rs = avg_gain / avg_loss
# #     return 100 - (100 / (1 + rs))

# # data['RSI'] = rsi(data)

# # # ✅ FIX 3: SAFE VWAP
# # data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3

# # data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
# # data['Volume'] = data['Volume'].fillna(0)

# # cum_vol = data['Volume'].cumsum()

# # data['VWAP'] = np.where(
# #     cum_vol != 0,
# #     (data['TP'] * data['Volume']).cumsum() / cum_vol,
# #     np.nan
# # )

# # # ATR (Volatility)
# # data['H-L'] = data['High'] - data['Low']
# # data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
# # data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
# # data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
# # data['ATR'] = data['TR'].rolling(14).mean()

# # # ---------------------------
# # # PCR (cached)
# # # ---------------------------
# # @st.cache_data(ttl=300)
# # def get_pcr():
# #     try:
# #         oc = nse_optionchain_scrapper("NIFTY")
# #         total_ce_oi = 0
# #         total_pe_oi = 0

# #         for item in oc['records']['data']:
# #             if 'CE' in item and 'PE' in item:
# #                 total_ce_oi += item['CE']['openInterest']
# #                 total_pe_oi += item['PE']['openInterest']

# #         return total_pe_oi / total_ce_oi if total_ce_oi != 0 else None
# #     except:
# #         return None

# # pcr = get_pcr()

# # # ---------------------------
# # # Latest Data
# # # ---------------------------
# # latest = data.iloc[-1]

# # # ---------------------------
# # # Market Regime
# # # ---------------------------
# # recent_high = data['High'].rolling(20).max().iloc[-1]
# # recent_low = data['Low'].rolling(20).min().iloc[-1]

# # range_pct = (recent_high - recent_low) / latest['Close']

# # if range_pct < 0.005:
# #     regime = "SIDEWAYS"
# # else:
# #     regime = "TRENDING"

# # # ---------------------------
# # # Breakout Logic
# # # ---------------------------
# # prev_high = data['High'].iloc[-2]
# # prev_low = data['Low'].iloc[-2]

# # bullish_breakout = latest['Close'] > prev_high
# # bearish_breakdown = latest['Close'] < prev_low

# # # ---------------------------
# # # Signal Logic
# # # ---------------------------
# # score = 0
# # reasons = []

# # if latest['EMA20'] > latest['EMA50']:
# #     score += 2
# #     reasons.append("Trend bullish (EMA20 > EMA50)")
# # else:
# #     score -= 2
# #     reasons.append("Trend bearish")

# # if latest['RSI'] > 55:
# #     score += 1
# #     reasons.append("Momentum strong")
# # elif latest['RSI'] < 45:
# #     score -= 1
# #     reasons.append("Momentum weak")

# # if latest['Close'] > latest['VWAP']:
# #     score += 2
# #     reasons.append("Above VWAP")
# # else:
# #     score -= 2
# #     reasons.append("Below VWAP")

# # if pcr is not None:
# #     if pcr > 1:
# #         score += 1
# #         reasons.append("PCR bullish")
# #     elif pcr < 0.8:
# #         score -= 1
# #         reasons.append("PCR bearish")

# # # ---------------------------
# # # Filters
# # # ---------------------------
# # filters_passed = True

# # if regime == "SIDEWAYS":
# #     filters_passed = False
# #     reasons.append("Sideways market")

# # if latest['ATR'] < latest['Close'] * 0.002:
# #     filters_passed = False
# #     reasons.append("Low volatility")

# # if score > 0 and not bullish_breakout:
# #     filters_passed = False
# #     reasons.append("No bullish breakout")

# # if score < 0 and not bearish_breakdown:
# #     filters_passed = False
# #     reasons.append("No bearish breakdown")

# # # ---------------------------
# # # Final Signal
# # # ---------------------------
# # if not filters_passed:
# #     signal = "⚪ NO TRADE"
# # elif score >= 4:
# #     signal = "🟢 STRONG BULLISH"
# # elif score >= 2:
# #     signal = "🟢 BULLISH"
# # elif score <= -4:
# #     signal = "🔴 STRONG BEARISH"
# # elif score <= -2:
# #     signal = "🔴 BEARISH"
# # else:
# #     signal = "⚪ NO TRADE"

# # confidence = abs(score) / 6

# # # ---------------------------
# # # UI
# # # ---------------------------
# # col1, col2 = st.columns(2)

# # with col1:
# #     st.subheader("📢 Signal")
# #     st.markdown(f"## {signal}")
# #     st.metric("Confidence", f"{confidence*100:.0f}%")

# #     st.metric("Close", f"{latest['Close']:.2f}")
# #     st.metric("RSI", f"{latest['RSI']:.2f}")
# #     st.metric("PCR", f"{pcr:.2f}" if pcr else "N/A")

# # with col2:
# #     st.subheader("📊 Market Context")
# #     st.write(f"Regime: {regime}")
# #     st.write(f"ATR: {latest['ATR']:.2f}")
# #     st.write(f"EMA20: {latest['EMA20']:.2f}")
# #     st.write(f"EMA50: {latest['EMA50']:.2f}")
# #     st.write(f"VWAP: {latest['VWAP']:.2f}")

# # # ---------------------------
# # # Reasons
# # # ---------------------------
# # st.subheader("🧠 Why this signal?")
# # for r in reasons:
# #     st.write(f"• {r}")

# # # ---------------------------
# # # Chart
# # # ---------------------------
# # fig = go.Figure()

# # fig.add_trace(go.Candlestick(
# #     x=data.index,
# #     open=data['Open'],
# #     high=data['High'],
# #     low=data['Low'],
# #     close=data['Close'],
# #     name="Price"
# # ))

# # fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], name="EMA20"))
# # fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], name="EMA50"))
# # fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name="VWAP"))

# # st.plotly_chart(fig, use_container_width=True)


















# # # import streamlit as st
# # # from streamlit_autorefresh import st_autorefresh
# # # import yfinance as yf
# # # import pandas as pd
# # # import numpy as np
# # # from nsepython import nse_optionchain_scrapper
# # # import plotly.graph_objects as go
# # # from datetime import datetime

# # # st.set_page_config(layout="wide")

# # # # ---------------------------
# # # # Auto Refresh (30 sec)
# # # # ---------------------------
# # # st_autorefresh(interval=30000, key="refresh")

# # # # ---------------------------
# # # # Header + Manual Refresh
# # # # ---------------------------
# # # col_title, col_btn = st.columns([6, 1])

# # # with col_title:
# # #     st.title("📊 NIFTY Live Sentiment Dashboard")

# # # with col_btn:
# # #     if st.button("🔄 Refresh"):
# # #         st.cache_data.clear()
# # #         st.rerun()

# # # st.caption("⏱ Auto-refreshes every 30 seconds")
# # # st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# # # # ---------------------------
# # # # Fetch Data (cached)
# # # # ---------------------------
# # # @st.cache_data(ttl=300)
# # # def load_data():
# # #     return yf.download("^NSEI", period="5d", interval="5m")

# # # data = load_data()

# # # if data.empty:
# # #     st.error("Failed to fetch market data.")
# # #     st.stop()

# # # # ---------------------------
# # # # Indicators
# # # # ---------------------------
# # # data['EMA20'] = data['Close'].ewm(span=20).mean()
# # # data['EMA50'] = data['Close'].ewm(span=50).mean()

# # # def rsi(df, period=14):
# # #     delta = df['Close'].diff()
# # #     gain = delta.clip(lower=0)
# # #     loss = -delta.clip(upper=0)
# # #     avg_gain = gain.rolling(period).mean()
# # #     avg_loss = loss.rolling(period).mean()
# # #     rs = avg_gain / avg_loss
# # #     return 100 - (100 / (1 + rs))

# # # data['RSI'] = rsi(data)

# # # data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
# # # data['VWAP'] = (data['TP'] * data['Volume']).cumsum() / data['Volume'].cumsum()

# # # # ATR (Volatility)
# # # data['H-L'] = data['High'] - data['Low']
# # # data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
# # # data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
# # # data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
# # # data['ATR'] = data['TR'].rolling(14).mean()

# # # # ---------------------------
# # # # PCR (cached)
# # # # ---------------------------
# # # @st.cache_data(ttl=300)
# # # def get_pcr():
# # #     try:
# # #         oc = nse_optionchain_scrapper("NIFTY")
# # #         total_ce_oi = 0
# # #         total_pe_oi = 0

# # #         for item in oc['records']['data']:
# # #             if 'CE' in item and 'PE' in item:
# # #                 total_ce_oi += item['CE']['openInterest']
# # #                 total_pe_oi += item['PE']['openInterest']

# # #         return total_pe_oi / total_ce_oi if total_ce_oi != 0 else None
# # #     except:
# # #         return None

# # # pcr = get_pcr()

# # # # ---------------------------
# # # # Latest Data
# # # # ---------------------------
# # # latest = data.iloc[-1]

# # # # ---------------------------
# # # # Market Regime
# # # # ---------------------------
# # # recent_high = data['High'].rolling(20).max().iloc[-1]
# # # recent_low = data['Low'].rolling(20).min().iloc[-1]

# # # range_pct = (recent_high - recent_low) / latest['Close']

# # # if range_pct < 0.005:
# # #     regime = "SIDEWAYS"
# # # else:
# # #     regime = "TRENDING"

# # # # ---------------------------
# # # # Breakout Logic
# # # # ---------------------------
# # # prev_high = data['High'].iloc[-2]
# # # prev_low = data['Low'].iloc[-2]

# # # bullish_breakout = latest['Close'] > prev_high
# # # bearish_breakdown = latest['Close'] < prev_low

# # # # ---------------------------
# # # # Signal Logic
# # # # ---------------------------
# # # score = 0
# # # reasons = []

# # # # EMA
# # # if latest['EMA20'] > latest['EMA50']:
# # #     score += 2
# # #     reasons.append("Trend bullish (EMA20 > EMA50)")
# # # else:
# # #     score -= 2
# # #     reasons.append("Trend bearish")

# # # # RSI
# # # if latest['RSI'] > 55:
# # #     score += 1
# # #     reasons.append("Momentum strong")
# # # elif latest['RSI'] < 45:
# # #     score -= 1
# # #     reasons.append("Momentum weak")

# # # # VWAP
# # # if latest['Close'] > latest['VWAP']:
# # #     score += 2
# # #     reasons.append("Above VWAP")
# # # else:
# # #     score -= 2
# # #     reasons.append("Below VWAP")

# # # # PCR
# # # if pcr is not None:
# # #     if pcr > 1:
# # #         score += 1
# # #         reasons.append("PCR bullish")
# # #     elif pcr < 0.8:
# # #         score -= 1
# # #         reasons.append("PCR bearish")

# # # # ---------------------------
# # # # Filters
# # # # ---------------------------
# # # filters_passed = True

# # # if regime == "SIDEWAYS":
# # #     filters_passed = False
# # #     reasons.append("Sideways market")

# # # if latest['ATR'] < latest['Close'] * 0.002:
# # #     filters_passed = False
# # #     reasons.append("Low volatility")

# # # if score > 0 and not bullish_breakout:
# # #     filters_passed = False
# # #     reasons.append("No bullish breakout")

# # # if score < 0 and not bearish_breakdown:
# # #     filters_passed = False
# # #     reasons.append("No bearish breakdown")

# # # # ---------------------------
# # # # Final Signal
# # # # ---------------------------
# # # if not filters_passed:
# # #     signal = "⚪ NO TRADE"
# # # elif score >= 4:
# # #     signal = "🟢 STRONG BULLISH"
# # # elif score >= 2:
# # #     signal = "🟢 BULLISH"
# # # elif score <= -4:
# # #     signal = "🔴 STRONG BEARISH"
# # # elif score <= -2:
# # #     signal = "🔴 BEARISH"
# # # else:
# # #     signal = "⚪ NO TRADE"

# # # confidence = abs(score) / 6

# # # # ---------------------------
# # # # UI
# # # # ---------------------------
# # # col1, col2 = st.columns(2)

# # # with col1:
# # #     st.subheader("📢 Signal")
# # #     st.markdown(f"## {signal}")
# # #     st.metric("Confidence", f"{confidence*100:.0f}%")

# # #     st.metric("Close", f"{latest['Close']:.2f}")
# # #     st.metric("RSI", f"{latest['RSI']:.2f}")
# # #     st.metric("PCR", f"{pcr:.2f}" if pcr else "N/A")

# # # with col2:
# # #     st.subheader("📊 Market Context")
# # #     st.write(f"Regime: {regime}")
# # #     st.write(f"ATR: {latest['ATR']:.2f}")
# # #     st.write(f"EMA20: {latest['EMA20']:.2f}")
# # #     st.write(f"EMA50: {latest['EMA50']:.2f}")
# # #     st.write(f"VWAP: {latest['VWAP']:.2f}")

# # # # ---------------------------
# # # # Reasons
# # # # ---------------------------
# # # st.subheader("🧠 Why this signal?")
# # # for r in reasons:
# # #     st.write(f"• {r}")

# # # # ---------------------------
# # # # Chart
# # # # ---------------------------
# # # fig = go.Figure()

# # # fig.add_trace(go.Candlestick(
# # #     x=data.index,
# # #     open=data['Open'],
# # #     high=data['High'],
# # #     low=data['Low'],
# # #     close=data['Close'],
# # #     name="Price"
# # # ))

# # # fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], name="EMA20"))
# # # fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], name="EMA50"))
# # # fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name="VWAP"))

# # # st.plotly_chart(fig, use_container_width=True)
