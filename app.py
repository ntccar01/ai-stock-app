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
st.set_page_config(page_title="AI 股市操盤手 V7.1 上櫃支援版", layout="wide")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心函式 ---

@st.cache_data(ttl=3600)
def get_stock_info(code, search_code):
    data = {
        "name": code,
        "pe": "N/A", "yield": 0, "eps": "N/A", "beta": "N/A",
        "financial_data": {"years": [], "revenues": [], "earnings": []}
    }
    try:
        # 嘗試抓取中文名稱 (twstock 支援上市上櫃)
        if code in twstock.codes:
            data["name"] = twstock.codes[code].name
        
        # 抓取 Yahoo 基本面
        stock = yf.Ticker(search_code)
        info = stock.info
        
        try:
            data["pe"] = info.get('trailingPE', 'N/A')
            data["eps"] = info.get('trailingEps', 'N/A')
            data["beta"] = info.get('beta', 'N/A')
            div = info.get('dividendRate', 0)
            pri = info.get('currentPrice') or info.get('previousClose')
            if div and pri: data["yield"] = div / pri
        except: pass

        return data
    except:
        return data

@st.cache_data(ttl=3600)
def get_data(ticker_symbol, start_date):
    try:
        # FinMind 只需要純數字代號 (例如 8069)
        stock_id = ticker_symbol.split('.')[0]
        start_str = start_date.strftime('%Y-%m-%d')
        
        dl = DataLoader()
        # FinMind 會自動去資料庫找，不分上市上櫃
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_str)
        
        if df.empty:
            print("FinMind empty, trying Yahoo backup...")
            return get_data_yahoo_backup(ticker_symbol, start_date)

        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 
            'close': 'Close', 'Trading_Volume': 'Volume'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

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

# --- 3. 介面邏輯 (已升級：支援上市上櫃判斷) ---
st.sidebar.header("🔍 設定與搜尋")
ticker_input = st.sidebar.text_input("輸入代號", value="2330")
time_range = st.sidebar.radio("區間", ["6個月", "1年", "3年"], index=1)

end_date = datetime.now()
if time_range == "6個月": start_date = end_date - timedelta(days=180)
elif time_range == "1年": start_date = end_date - timedelta(days=365)
else: start_date = end_date - timedelta(days=1095)

# --- 智慧代號判斷 ---
ticker_input = ticker_input.strip() # 去除前後空白
if ticker_input.isdigit():
    # 如果使用者只輸入數字 (如 8069)
    suffix = ".TW" # 預設上市
    
    # 檢查是否為上櫃股 (使用 twstock 清單)
    if ticker_input in twstock.codes:
        if twstock.codes[ticker_input].type == "上櫃":
            suffix = ".TWO"
            
    ticker_search = ticker_input + suffix
    ticker_display = ticker_input
else:
    # 如果使用者自己輸入了 .TW 或 .TWO
    ticker_search = ticker_input
    ticker_display = ticker_input.split('.')[0]

# --- 執行抓取 ---
df = get_data(ticker_search, start_date)
info = get_stock_info(ticker_display, ticker_search)
market_df = get_market_data(start_date)

tab1, tab2, tab3 = st.tabs(["📊 綜合分析與績效", "🧠 AI 預測模型", "🎯 智慧選股掃描"])

# --- TAB 1 ---
with tab1:
    if df is not None:
        st.subheader(f"📈 {info['name']} ({ticker_display}) 深度分析")
        
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
        st.error(f"❌ 無法讀取 {ticker_display} 的資料。")

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
    st.subheader("🎯 智慧選股雷達 (含上櫃)")
    # 這裡加入一些上櫃熱門股範例：8069(元太), 3293(鈊象)
    target_stocks = ['2330', '2317', '2454', '8069', '3293']
    if st.button("📡 快速掃描"):
        results = []
        bar = st.progress(0)
        
        dl = DataLoader()
        start_scan = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        for i, code in enumerate(target_stocks):
            bar.progress((i+1)/len(target_stocks))
            try:
                d = dl.taiwan_stock_daily(stock_id=code, start_date=start_scan)
                if d.empty: continue
                
                close = d['close'].iloc[-1]
                rsi = ta.rsi(pd.Series(d['close']), 14).iloc[-1]
                
                # 判斷上市上櫃顯示名稱
                name = code
                if code in twstock.codes:
                    name = twstock.codes[code].name

                results.append({
                    "名稱": name,
                    "代號": code,
                    "現價": close,
                    "RSI": round(rsi, 2)
                })
            except: continue
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("掃描無結果")
