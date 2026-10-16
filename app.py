import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Gold & Silver Live Tracker", layout="wide")

st.title("🪙 Personal Bullion (Gold & Silver) Live Dashboard")
st.caption(f"Last Updated: {datetime.now().strftime('%d %b %Y, %I:%M:%S %p')} | Multi-Timeframe Technical & Macro Tracker")

# Refresh Button
if st.button("🔄 Refresh Live Data"):
    st.cache_data.clear()

# --- HELPER FUNCTIONS FOR TECHNICAL INDICATORS ---
def calculate_indicators(df):
    if df.empty or len(df) < 50:
        return df
    
    df = df.copy()
    
    # 1. Moving Averages (20, 50, 200)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. ADX (14)
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm_gt = plus_dm > minus_dm
    minus_dm_gt = minus_dm > plus_dm
    plus_dm[~plus_dm_gt] = 0
    minus_dm[~minus_dm_gt] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # 5. Price Structure (Higher Highs / Higher Lows check over last 10 bars)
    recent_highs = df['High'].tail(10)
    recent_lows = df['Low'].tail(10)
    df['Structure'] = "Range / Sideways"
    if recent_highs.iloc[-1] > recent_highs.iloc[0] and recent_lows.iloc[-1] > recent_lows.iloc[0]:
        df.loc[df.index[-1], 'Structure'] = "Higher High - Higher Low (Uptrend)"
    elif recent_highs.iloc[-1] < recent_highs.iloc[0] and recent_lows.iloc[-1] < recent_lows.iloc[0]:
        df.loc[df.index[-1], 'Structure'] = "Lower High - Lower Low (Downtrend)"
        
    return df

@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return calculate_indicators(data)

# --- SECTION 1: FUNDAMENTAL & MACRO DRIVERS + GOLD-SILVER RATIO ---
st.header("1. ग्लोबल मैक्रो, MCX फैक्टर्स और Gold-Silver Ratio")

macro_tickers = {
    "Gold (COMEX $)": "GC=F",
    "Silver (COMEX $)": "SI=F",
    "Dollar Index (DXY)": "DX-Y.NYB",
    "USD/INR (रुपया-डॉलर)": "INR=X",
    "US 10Y Yield (%)": "^TNX",
    "TIPS ETF (Real Yield Proxy)": "TIP"
}

macro_data = {}
for name, sym in macro_tickers.items():
    df_m = yf.download(sym, period="1mo", interval="1d", progress=False)
    if isinstance(df_m.columns, pd.MultiIndex):
        df_m.columns = df_m.columns.get_level_values(0)
    if not df_m.empty and len(df_m) >= 2:
        curr = float(df_m['Close'].iloc[-1])
        prev = float(df_m['Close'].iloc[-2])
        chg = ((curr - prev) / prev) * 100
        macro_data[name] = (curr, chg, df_m['Close'])

cols = st.columns(4)

# Gold-Silver Ratio Calculation
if "Gold (COMEX $)" in macro_data and "Silver (COMEX $)" in macro_data:
    g_price, g_chg, g_series = macro_data["Gold (COMEX $)"]
    s_price, s_chg, s_series = macro_data["Silver (COMEX $)"]
    
    gs_ratio_series = (g_series / s_series).dropna()
    curr_gsr = gs_ratio_series.iloc[-1]
    prev_gsr = gs_ratio_series.iloc[-2]
    gsr_sma10 = gs_ratio_series.tail(10).mean()
    gsr_chg = curr_gsr - prev_gsr
    
    with cols[0]:
        st.metric("⚖️ Gold-Silver Ratio", f"{curr_gsr:.2f}", f"{gsr_chg:+.2f}")
        if curr_gsr > gsr_sma10:
            st.info("💡 **संकेत:** अनुपात बढ़ रहा है (10-D Avg से ऊपर)। **गोल्ड पेटल (Gold)** चांदी से ज़्यादा मज़बूत है।")
        else:
            st.success("💡 **संकेत:** अनुपात गिर रहा है (10-D Avg से नीचे)। **चांदी (Silver)** गोल्ड से बेहतर चल रही है।")

# DXY
if "Dollar Index (DXY)" in macro_data:
    val, chg, _ = macro_data["Dollar Index (DXY)"]
    with cols[1]:
        st.metric("💵 Dollar Index (DXY)", f"{val:.2f}", f"{chg:+.2f}%", delta_color="inverse")
        if chg < 0:
            st.success("✅ डॉलर कमज़ोर है (गोल्ड/सिल्वर के लिए सपोर्टिव)")
        else:
            st.warning("⚠️ डॉलर मज़बूत हो रहा है (भाव पर दबाव संभव)")

# USD/INR (MCX Specific)
if "USD/INR (रुपया-डॉलर)" in macro_data:
    val, chg, _ = macro_data["USD/INR (रुपया-डॉलर)"]
    with cols[2]:
        st.metric("🇮🇳 रुपया-डॉलर (USD/INR)", f"₹{val:.2f}", f"{chg:+.2f}%")
        if chg > 0:
            st.info("📈 रुपया कमज़ोर: अंतरराष्ट्रीय भाव स्थिर रहने पर भी **MCX को ऊपर जाने का सहारा मिलेगा**।")
        else:
            st.warning("📉 रुपया मज़बूत: MCX में तेज़ी थोड़ी सीमित रह सकती है।")

# US Yields
if "US 10Y Yield (%)" in macro_data:
    val, chg, _ = macro_data["US 10Y Yield (%)"]
    with cols[3]:
        st.metric("🏦 US 10Y Yield", f"{val:.2f}%", f"{chg:+.2f}%", delta_color="inverse")
        if chg < 0:
            st.success("✅ बॉन्ड यील्ड गिर रही है (गोल्ड के लिए सकारात्मक)")
        else:
            st.error("⚠️ बॉन्ड यील्ड बढ़ रही है (गोल्ड पर दबाव)")

st.divider()

# --- SECTION 2: MULTI-TIMEFRAME TECHNICAL ENGINE ---
st.header("2. Multi-Timeframe Technical Scanner (चार्ट और नियम)")
st.markdown("**नियम:** Weekly से दिशा तय करें ➡️ Daily से ट्रेड की तैयारी करें ➡️ 1-Hour / 15-Min से एंट्री लें।")

asset_choice = st.radio(
    "एसेट चुनें (Select Asset to Analyze):",
    ["Gold (International COMEX - GC=F)", "Silver (International COMEX - SI=F)", "Gold BeES (Indian NSE - GOLDBEES.NS)", "Silver BeES (Indian NSE - SILVERBEES.NS)"],
    horizontal=True
)

ticker_map = {
    "Gold (International COMEX - GC=F)": "GC=F",
    "Silver (International COMEX - SI=F)": "SI=F",
    "Gold BeES (Indian NSE - GOLDBEES.NS)": "GOLDBEES.NS",
    "Silver BeES (Indian NSE - SILVERBEES.NS)": "SILVERBEES.NS"
}
selected_ticker = ticker_map[asset_choice]

timeframes = {
    "1. Weekly (दिशा तय करें)": ("5y", "1wk"),
    "2. Daily (ट्रेड की तैयारी)": ("2y", "1d"),
    "3. 1-Hour (एंट्री ज़ोन)": ("60d", "1h"),
    "4. 15-Min (सटीक एंट्री)": ("15d", "15m")
}

tf_results = []
tf_dfs = {}

for tf_name, (period, interval) in timeframes.items():
    df_tf = fetch_data(selected_ticker, period, interval)
    tf_dfs[tf_name] = df_tf
    if not df_tf.empty:
        last = df_tf.iloc[-1]
        prev = df_tf.iloc[-2] if len(df_tf) > 1 else last
        
        price = last['Close']
        sma20 = last['SMA20']
        sma50 = last['SMA50']
        sma200 = last['SMA200']
        adx = last['ADX']
        rsi = last['RSI']
        macd = last['MACD']
        macd_sig = last['MACD_Signal']
        macd_hist = last['MACD_Hist']
        prev_hist = prev['MACD_Hist']
        
        # Rule 1: Moving Average Condition
        if pd.notna(sma200):
            if price > sma20 and price > sma50 and price > sma200 and sma20 > sma50:
                ma_status = "🟢 मज़बूत तेज़ी (Price > 20 > 50 > 200)"
            elif price < sma20 and price < sma50 and price < sma200 and sma20 < sma50:
                ma_status = "🔴 मज़बूत मंदी (Price < 20 < 50 < 200)"
            else:
                ma_status = "🟡 मिला-जुला / Pullback"
        else:
            ma_status = "🟢 तेज़ी (20>50)" if (price > sma20 and sma20 > sma50) else "🔴 मंदी / Range"

        # Rule 2: ADX (14) Condition
        if adx > 25:
            adx_status = f"🔥 मज़बूत Trend ({adx:.1f} > 25)"
        elif adx < 20:
            adx_status = f"⚠️ Range/दायरे में ({adx:.1f} < 20) - Trend ट्रेड से बचें"
        else:
            adx_status = f"⏳ Trend बन रहा है ({adx:.1f})"

        # Rule 3: MACD Condition
        if macd > macd_sig and macd_hist > prev_hist:
            macd_status = "🟢 तेज़ी की पुष्टि (Above Signal & Hist ⬆️)"
        elif macd > macd_sig:
            macd_status = "🟢 Bullish (Hist धीमा)"
        elif macd < macd_sig and macd_hist < prev_hist:
            macd_status = "🔴 मंदी की पुष्टि (Below Signal & Hist ⬇️)"
        else:
            macd_status = "🔴 Bearish (Hist सुधर रहा)"

        # Rule 4: RSI Condition
        if rsi > 70:
            rsi_status = f"🟢 {rsi:.1f} (>50 तेज़ी | >70 मज़बूत मोमेंटम, अकेले बेचें नहीं)"
        elif rsi > 50:
            rsi_status = f"🟢 {rsi:.1f} (50 के ऊपर - तेज़ी का संकेत)"
        elif rsi < 30:
            rsi_status = f"🔴 {rsi:.1f} (<30 मंदी, अकेले खरीदें नहीं)"
        else:
            rsi_status = f"🔴 {rsi:.1f} (50 के नीचे - कमज़ोर)"

        tf_results.append({
            "Timeframe": tf_name,
            "LTP (भाव)": f"{price:.2f}",
            "Price Structure": last['Structure'],
            "Moving Averages (20, 50, 200)": ma_status,
            "ADX (14) नियम": adx_status,
            "MACD स्थिति": macd_status,
            "RSI (14) स्थिति": rsi_status
        })

st.table(pd.DataFrame(tf_results))

# --- SECTION 3: INTERACTIVE CHART ---
st.subheader("📊 Interactive Chart (Moving Averages के साथ)")
chart_tf = st.selectbox("चार्ट का Timeframe चुनें:", list(timeframes.keys()), index=1)
df_plot = tf_dfs[chart_tf].tail(120)

if not df_plot.empty:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA20'], line=dict(color='blue', width=1.5), name='20 SMA'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='orange', width=1.5), name='50 SMA'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='red', width=2), name='200 SMA'))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- SECTION 4: MCX vs COMEX PARITY & FUNDAMENTAL CHECKLIST ---
col_mcx, col_fund = st.columns(2)

with col_mcx:
    st.subheader("4. 🇮🇳 MCX vs COMEX Parity Calculator")
    st.caption("अंतरराष्ट्रीय स्पॉट और घरेलू MCX भाव के फर्क (Premium/Discount) को समझने के लिए:")
    
    default_comex = macro_data["Gold (COMEX $)"][0] if "Gold (COMEX $)" in macro_data else 2600.0
    default_usdinr = macro_data["USD/INR (रुपया-डॉलर)"][0] if "USD/INR (रुपया-डॉलर)" in macro_data else 84.0
    
    c_gold = st.number_input("COMEX Gold Price ($ / Troy Ounce)", value=float(round(default_comex, 2)))
    u_inr = st.number_input("Live USD/INR Rate (₹)", value=float(round(default_usdinr, 2)))
    imp_duty = st.number_input("आयात शुल्क + GST / Local Taxes (%)", value=9.0, step=0.5)
    
    # 1 Troy Ounce = 31.1034768 grams
    base_inr_per_gram = (c_gold * u_inr) / 31.1035
    landed_per_gram = base_inr_per_gram * (1 + (imp_duty / 100))
    mcx_10g_fair = landed_per_gram * 10
    
    st.metric("अनुमानित MCX Gold (10 ग्राम) Fair Value", f"₹{mcx_10g_fair:,.0f}")
    st.metric("अनुमानित Gold Petal (1 ग्राम) Fair Value", f"₹{landed_per_gram:,.0f}")
    st.caption("*नोट: अगर असली MCX भाव इस Fair Value से ऊपर चल रहा है, तो घरेलू मांग या रुपये की वजह से प्रीमियम है।*")

with col_fund:
    st.subheader("📋 ट्रेड लेने से पहले की Fundamental Checklist")
    st.write("ट्रेड लेने से पहले इन खबरों की स्थिति पर टिक लगाएं:")
    c1 = st.checkbox("फेड की नीति: क्या ब्याज दर कटौती (Rate Cut) की उम्मीद है?")
    c2 = st.checkbox("केंद्रीय बैंकों की खरीद: क्या WGC के मासिक आंकड़ों में खरीद मज़बूत है?")
    c3 = st.checkbox("भू-राजनीतिक तनाव (Geopolitics): क्या सेफ-हेवन मांग बनी हुई है?")
    c4 = st.checkbox("चांदी विशेष: क्या Solar / Electronics (औद्योगिक मांग) के आंकड़े सकारात्मक हैं?")
    
    score = sum([c1, c2, c3, c4])
    if score >= 3:
        st.success(f"Fundamental Score: {score}/4 — फंडामेंटल रुझान मज़बूत तेज़ी के पक्ष में है।")
    elif score >= 1:
        st.info(f"Fundamental Score: {score}/4 — फंडामेंटल सामान्य हैं, पूरी तरह Technicals पर निर्भर रहें।")
    else:
        st.warning("कोई भी फंडामेंटल ट्रिगर चेक नहीं किया गया है।")
