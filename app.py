import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 股市操盤手 V5.0 Ultimate", layout="wide")

# CSS 美化 (讓分頁標籤變大一點)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 共用函式 ---

@st.cache_data(ttl=3600)
def get_stock_info(code, search_code):
    try:
        stock_name = twstock.codes[code].name
    except:
        stock_name = code

    try:
        stock = yf.Ticker(search_code)
        info = stock.info

        dividend_rate = info.get('dividendRate', 0)
        current_price = info.get('currentPrice') or info.get('previousClose')

        if dividend_rate and current_price and current_price > 0:
            calculated_yield = dividend_rate / current_price
        else:
            raw_yield = info.get('dividendYield', 0)
            calculated_yield = raw_yield if raw_yield and raw_yield < 0.2 else 0

        fundamentals = {
            "name": stock_name,
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "yield_pct": calculated_yield,
            "eps": info.get('trailingEps', 'N/A'),
            "market_cap": info.get('marketCap', 0),
            "beta": info.get('beta', 'N/A')
        }
    except:
        fundamentals = None
    return fundamentals

@st.cache_data(ttl=3600)
def get_data(ticker_symbol, start):
    try:
        df = yf.download(ticker_symbol, start=start, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 基礎指標
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_60'] = ta.sma(df['Close'], length=60)
        df['RSI'] = ta.rsi(df['Close'], length=14)

        # 為了 AI 預測，多增加一些特徵值
        df['Momentum'] = df['Close'] - df['Close'].shift(5) # 動能
        df['Volatility'] = df['Close'].rolling(5).std() # 波動率

        df = df.dropna() # 移除空值
        return df
    except:
        return None

def get_news(stock_name):
    try:
        query = urllib.parse.quote(stock_name)
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        feed = feedparser.parse(rss_url)
        return feed.entries[:5]
    except:
        return []

# --- AI 預測模型 ---
def train_and_predict(df):
    # 準備資料：用「過去的指標」預測「明天的收盤價」
    df_ml = df.copy()
    df_ml['Target'] = df_ml['Close'].shift(-1) # 目標是明天的價格

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'Momentum', 'Volatility']

    data = df_ml.dropna() # 移除最後一行(因為沒有明天)

    X = data[features]
    y = data['Target']

    # 切分訓練集與測試集
    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 建立模型 (隨機森林)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 準確度評估 (R平方)
    score = model.score(X_test, y_test)

    # 預測明天 (用最後一天的數據來預測未知)
    last_row = df_ml.iloc[[-1]][features]
    predicted_price = model.predict(last_row)[0]

    return predicted_price, score

# --- 3. 介面佈局 (使用 Tabs) ---
st.sidebar.header("🔍 設定與搜尋")
ticker_input = st.sidebar.text_input("請輸入台股代號", value="2330")
time_range = st.sidebar.radio("時間區間", ["6個月", "1年", "3年"], index=1)

# 處理日期
end_date = datetime.now()
if time_range == "6個月": start_date = end_date - timedelta(days=180)
elif time_range == "1年": start_date = end_date - timedelta(days=365)
else: start_date = end_date - timedelta(days=1095)

# 處理代號
if not ticker_input.endswith(".TW") and not ticker_input.endswith(".TWO"):
    ticker_search = ticker_input + ".TW"
else:
    ticker_search = ticker_input
    ticker_input = ticker_input.split('.')[0]

# 建立三大分頁
tab1, tab2, tab3 = st.tabs(["📊 個股綜合分析", "🧠 AI 股價預測", "🎯 智慧選股雷達"])

# --- Tab 1: 既有的個股分析功能 ---
with tab1:
    info_data = get_stock_info(ticker_input, ticker_search)
    df = get_data(ticker_search, start_date)

    if df is not None and info_data is not None:
        st.subheader(f"{info_data['name']} ({ticker_input}) 即時儀表板")

        # 基本面卡片
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("本益比", f"{info_data['pe_ratio']:.1f}" if isinstance(info_data['pe_ratio'], (int, float)) else "N/A")
        yield_val = info_data['yield_pct']
        c2.metric("殖利率", f"{yield_val*100:.2f}%" if isinstance(yield_val, (int, float)) else "N/A")
        c3.metric("EPS", f"{info_data['eps']:.2f}" if isinstance(info_data['eps'], (int, float)) else "N/A")
        c4.metric("Beta", f"{info_data['beta']:.2f}" if isinstance(info_data['beta'], (int, float)) else "N/A")

        # 圖表
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], line=dict(color='orange', width=1), name='5日'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='20日'), row=1, col=1)
        colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='量'), row=2, col=1)
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 新聞
        with st.expander(f"📰 查看 {info_data['name']} 最新新聞"):
            news_list = get_news(info_data['name'])
            for news in news_list:
                st.write(f"- [{news.title}]({news.link})")
    else:
        st.error("查無資料")

# --- Tab 2: AI 預測功能 (True AI) ---
with tab2:
    if df is not None:
        st.subheader(f"🤖 AI 預測實驗室：{info_data['name']}")
        st.info("說明：此功能使用「隨機森林 (Random Forest)」機器學習模型，根據過去的價量、波動、RSI 等特徵，預測「下一個交易日」的收盤價。")

        if st.button("🚀 開始訓練模型並預測"):
            with st.spinner("AI 正在學習這檔股票的歷史慣性..."):
                predicted_price, accuracy = train_and_predict(df)

                # 顯示結果
                last_price = df['Close'].iloc[-1]
                change = predicted_price - last_price
                change_pct = (change / last_price) * 100

                col_ai1, col_ai2 = st.columns(2)

                with col_ai1:
                    st.metric("AI 預測下個交易日價格", f"{predicted_price:.2f}", f"{change_pct:.2f}%")
                    if change > 0:
                        st.success(f"AI 判斷趨勢：看漲 📈 (目標價 {predicted_price:.2f})")
                    else:
                        st.error(f"AI 判斷趨勢：看跌 📉 (目標價 {predicted_price:.2f})")

                with col_ai2:
                    st.metric("模型信心度 (R² Score)", f"{accuracy*100:.1f}%")
                    if accuracy > 0.8:
                        st.caption("✅ 模型在測試數據上表現優異，參考價值高。")
                    else:
                        st.caption("⚠️ 此股波動無規律，模型預測能力較低，請謹慎參考。")

# --- Tab 3: 智慧選股雷達 (Screener) ---
with tab3:
    st.subheader("🎯 智慧選股雷達 (掃描熱門股)")
    st.write("此功能將掃描「台灣 50 成分股」與熱門標的，找出符合 **「黃金交叉 (短線轉強)」** 或 **「RSI 超賣 (跌深反彈)」** 的潛力股。")

    # 為了避免等待太久，我們只掃描精選名單
    target_stocks = ['2330', '2317', '2454', '2308', '2603', '2609', '2615', '2881', '2882', '2412', '1605', '2303', '3008', '3037', '3034']

    if st.button("📡 啟動全市場掃描"):
        results = []
        progress_bar = st.progress(0)

        status_text = st.empty()

        for i, code in enumerate(target_stocks):
            status_text.text(f"正在掃描：{code}...")
            progress_bar.progress((i + 1) / len(target_stocks))

            # 抓資料
            stock_code = code + ".TW"
            try:
                # 只抓最近 30 天夠算指標就好，比較快
                d = yf.download(stock_code, period="1mo", progress=False)
                if d.empty: continue
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)

                # 算指標
                sma5 = ta.sma(d['Close'], length=5).iloc[-1]
                sma20 = ta.sma(d['Close'], length=20).iloc[-1]
                prev_sma5 = ta.sma(d['Close'], length=5).iloc[-2]
                prev_sma20 = ta.sma(d['Close'], length=20).iloc[-2]
                rsi = ta.rsi(d['Close'], length=14).iloc[-1]
                close = d['Close'].iloc[-1]

                signal = ""
                # 判斷黃金交叉
                if prev_sma5 <= prev_sma20 and sma5 > sma20:
                    signal = "🔥 黃金交叉 (買進訊號)"
                # 判斷 RSI 超賣
                elif rsi < 30:
                    signal = "💎 RSI 超賣 (跌深反彈)"

                if signal:
                    # 嘗試抓中文名
                    try: name = twstock.codes[code].name
                    except: name = code

                    results.append({
                        "代號": code,
                        "名稱": name,
                        "現價": f"{close:.2f}",
                        "RSI": f"{rsi:.2f}",
                        "訊號": signal
                    })

            except Exception as e:
                continue

        status_text.text("掃描完成！")

        if results:
            st.success(f"找到 {len(results)} 檔潛力股！")
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("目前掃描名單中沒有發現符合策略的股票。")
