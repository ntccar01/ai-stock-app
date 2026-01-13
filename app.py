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
import requests
from FinMind.data import DataLoader

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 股市操盤手 V7.0 FinMind版", layout="wide")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心函式 (改用 FinMind) ---

@st.cache_data(ttl=3600)
def get_stock_info(code, search_code):
    # 這裡我們保留 Yahoo 抓基本面，因為 FinMind 主要強在價量資料
    # 如果 Yahoo 基本面也擋，我們至少還有 FinMind 的股價可以看 K 線
    data = {
        "name": code,
        "pe": "N/A", "yield": 0, "eps": "N/A", "beta": "N/A",
        "financial_data": {"years": [], "revenues": [], "earnings": []}
    }
    try:
        if code.isdigit():
            try: data["name"] = twstock.codes[code].name
            except: pass

        stock = yf.Ticker(search_code)
        info = stock.info
        
        # 嘗試抓基本面 (失敗也沒關係，不影響 K 線)
        try:
            data["pe"] = info.get('trailingPE', 'N/A')
            data["eps"] = info.get('trailingEps', 'N/A')
            data["beta"] = info.get('beta', 'N/A')
            
            # 殖利率
            div = info.get('dividendRate', 0)
            pri = info.get('currentPrice') or info.get('previousClose')
            if div and pri: data["yield"] = div / pri
        except: pass

        return data
    except:
        return data

@st.cache_data(ttl=3600)
def get_data(ticker_symbol, start_date):
    # --- 核心修改：改用 FinMind 抓股價 ---
    try:
        # FinMind 需要的是 "2330" 這種純數字，不需要 ".TW"
        stock_id = ticker_symbol.split('.')[0]
        start_str = start_date.strftime('%Y-%m-%d')
        
        dl = DataLoader()
        # 下載台股日成交資訊
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_str)
        
        if df.empty:
            # 如果 FinMind 失敗，最後嘗試一次 Yahoo (當作備用)
            print("FinMind empty, trying Yahoo backup...")
            return get_data_yahoo_backup(ticker_symbol, start_date)

        # FinMind 的欄位是小寫，我們要改成大寫以符合後面的程式邏輯
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 
            'close': 'Close', 'Trading_Volume': 'Volume'
        })
        
        # 設定日期為索引
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # 確保數據類型是數字
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # --- 計算技術指標 ---
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_60'] = ta.sma(df['Close'], length=60)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        return df.dropna()
        
    except Exception as e:
        print(f"FinMind Error: {e}")
        return None

def get_data_yahoo_backup(ticker_symbol, start):
    # 這是原本的 Yahoo 下載邏輯，當作備用方案
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        df = yf.download(ticker_symbol, start=start, progress=False, session=session)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty or 'Close' not in df.columns: return None
        
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_60'] = ta.sma(df['Close'], length=60)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        return df.dropna()
    except:
        return None

@st.cache_data(ttl=3600)
def get_market_data(start):
    try:
        # 大盤我們還是試試看 Yahoo，因為 FinMind 抓大盤要另外的代碼
        df = yf.download("^TWII", start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df['Close']
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

def train_and_predict(df):
    try:
        df_ml = df.copy()
        df_ml['Target'] = df_ml['Close'].shift(-1)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'Momentum']
        data = df_ml.dropna()
        X = data[features]
        y = data['Target']
        if len(X) < 10: return 0, 0
        split = int(len(X) * 0.9)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X.iloc[:split], y.iloc[:split])
        score = model.score(X.iloc[split:], y.iloc[split:])
        pred = model.predict(df_ml.iloc[[-1]][features])[0]
        return pred, score
    except:
        return 0, 0

# --- 3. 介面邏輯 ---
st.sidebar.header("🔍 設定與搜尋")
ticker_input = st.sidebar.text_input("輸入代號", value="2330")
time_range = st.sidebar.radio("區間", ["6個月", "1年", "3年"], index=1)

end_date = datetime.now()
if time_range == "6個月": start_date = end_date - timedelta(days=180)
elif time_range == "1年": start_date = end_date - timedelta(days=365)
else: start_date = end_date - timedelta(days=1095)

# 代號處理 (確保有 .TW 給 Yahoo 用，純數字給 FinMind 用)
if not ticker_input.endswith(".TW") and not ticker_input.endswith(".TWO"):
    ticker_search = ticker_input + ".TW"
else:
    ticker_search = ticker_input
    ticker_input = ticker_input.split('.')[0] 

# --- 執行抓取 ---
# 1. 抓股價 (優先用 FinMind)
df = get_data(ticker_search, start_date)
# 2. 抓基本面 (用 Yahoo，失敗就算了)
info = get_stock_info(ticker_input, ticker_search)
# 3. 抓大盤
market_df = get_market_data(start_date)

tab1, tab2, tab3 = st.tabs(["📊 綜合分析與績效", "🧠 AI 預測模型", "🎯 智慧選股掃描"])

# --- TAB 1 ---
with tab1:
    if df is not None:
        st.subheader(f"📈 {info['name']} ({ticker_input}) 深度分析")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("本益比", f"{info['pe']:.1f}" if info['pe'] != 'N/A' else "-")
        c2.metric("殖利率", f"{info['yield']*100:.2f}%" if info['yield'] else "-")
        c3.metric("EPS", f"{info['eps']:.2f}" if info['eps'] != 'N/A' else "-")
        c4.metric("收盤價", f"{df['Close'].iloc[-1]:.1f}")
        
        # 績效圖
        st.markdown("### 🆚 績效對決")
        try:
            stock_norm = (df['Close'] / df['Close'].iloc[0] - 1) * 100
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=df.index, y=stock_norm, mode='lines', name=info['name'], line=dict(color='red')))
            
            if market_df is not None:
                market_aligned = market_df.reindex(df.index, method='ffill')
                market_norm = (market_aligned / market_aligned.iloc[0] - 1) * 100
                fig_compare.add_trace(go.Scatter(x=df.index, y=market_norm, mode='lines', name='大盤', line=dict(color='gray', dash='dash')))
            
            fig_compare.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_compare, use_container_width=True)
        except:
            st.write("績效圖繪製失敗 (資料長度不符)")

        st.markdown("---")
        
        col_chart, col_news = st.columns([2, 1])
        with col_chart:
            st.markdown("### 🕯️ K線圖")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='月線'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量'), row=2, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_news:
             with st.expander(f"📰 {info['name']} 最新新聞", expanded=True):
                news_list = get_news(info['name'])
                for news in news_list:
                    st.write(f"- [{news.title}]({news.link})")
    else:
        st.error(f"❌ 無法讀取 {ticker_input} 的股價資料。")
        st.info("如果是剛剛上市的股票，FinMind 可能還沒有資料。請嘗試成熟的股票如 2330, 2317。")

# --- TAB 2 ---
with tab2:
    if df is not None:
        st.subheader(f"🤖 AI 預測實驗室")
        if st.button("🚀 執行 AI 運算"):
            with st.spinner("AI 運算中..."):
                pred, acc = train_and_predict(df)
                if pred > 0:
                    last = df['Close'].iloc[-1]
                    chg = (pred - last) / last * 100
                    c1, c2 = st.columns(2)
                    c1.metric("AI 預測價格", f"{pred:.2f}", f"{chg:.2f}%")
                    c2.metric("模型信心度", f"{acc*100:.1f}%")
    else:
        st.warning("無資料")

# --- TAB 3 ---
with tab3:
    st.subheader("🎯 智慧選股雷達 (FinMind版)")
    target_stocks = ['2330', '2317', '2454', '2603', '2881']
    if st.button("📡 快速掃描"):
        results = []
        bar = st.progress(0)
        
        # 使用 FinMind 批量掃描
        dl = DataLoader()
        start_scan = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        for i, code in enumerate(target_stocks):
            bar.progress((i+1)/len(target_stocks))
            try:
                # 這裡改用 FinMind 抓取
                d = dl.taiwan_stock_daily(stock_id=code, start_date=start_scan)
                if d.empty: continue
                
                # 簡單整理數據
                close = d['close'].iloc[-1]
                rsi = ta.rsi(pd.Series(d['close']), 14).iloc[-1]
                
                results.append({
                    "代號": code,
                    "現價": close,
                    "RSI": round(rsi, 2)
                })
            except: continue
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("掃描無結果")
