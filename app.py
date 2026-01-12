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
import io

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 股市操盤手 V6.1 Fix", layout="wide")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心函式 ---

@st.cache_data(ttl=3600)
def get_stock_info(code, search_code):
    # 預設值，避免當機
    data = {
        "name": code,
        "pe": "N/A",
        "yield": 0,
        "eps": "N/A",
        "beta": "N/A",
        "financial_data": {"years": [], "revenues": [], "earnings": []}
    }
    
    try:
        # 嘗試抓取中文名稱
        try:
            data["name"] = twstock.codes[code].name
        except:
            data["name"] = code

        stock = yf.Ticker(search_code)
        info = stock.info
        
        # 抓取財報
        try:
            financials = stock.financials
            if not financials.empty:
                years = [str(d)[:4] for d in financials.columns[:3]]
                revenues = financials.loc['Total Revenue'][:3].values / 100000000
                earnings = financials.loc['Net Income'][:3].values / 100000000
                data["financial_data"] = {
                    "years": years[::-1],
                    "revenues": revenues[::-1] if len(revenues)>0 else [],
                    "earnings": earnings[::-1] if len(earnings)>0 else []
                }
        except:
            pass # 財報抓不到就算了，不要當機

        # 抓取基本面
        dividend_rate = info.get('dividendRate', 0) 
        current_price = info.get('currentPrice') or info.get('previousClose')
        
        if dividend_rate and current_price:
            data["yield"] = dividend_rate / current_price
        else:
            data["yield"] = info.get('dividendYield', 0)

        data["pe"] = info.get('trailingPE', 'N/A')
        data["eps"] = info.get('trailingEps', 'N/A')
        data["beta"] = info.get('beta', 'N/A')

        return data

    except Exception as e:
        print(f"Error fetching info: {e}")
        return None

@st.cache_data(ttl=3600)
def get_data(ticker_symbol, start):
    try:
        df = yf.download(ticker_symbol, start=start, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_60'] = ta.sma(df['Close'], length=60)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        df['Volatility'] = df['Close'].rolling(5).std()
        
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
        
        if len(X) < 10: return 0, 0 # 資料太少不預測

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

if not ticker_input.endswith(".TW") and not ticker_input.endswith(".TWO"):
    ticker_search = ticker_input + ".TW"
else:
    ticker_search = ticker_input
    ticker_input = ticker_input.split('.')[0]

# 抓取資料
info = get_stock_info(ticker_input, ticker_search)
df = get_data(ticker_search, start_date)
market_df = get_market_data(start_date)

tab1, tab2, tab3 = st.tabs(["📊 綜合分析與績效", "🧠 AI 預測模型", "🎯 智慧選股掃描"])

# --- TAB 1 ---
with tab1:
    if df is not None and info is not None:
        st.subheader(f"📈 {info['name']} ({ticker_input}) 深度分析")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("本益比", f"{info['pe']:.1f}" if info['pe'] != 'N/A' else "N/A")
        c2.metric("殖利率", f"{info['yield']*100:.2f}%" if info['yield'] else "N/A")
        c3.metric("EPS", f"{info['eps']:.2f}" if info['eps'] != 'N/A' else "N/A")
        c4.metric("Beta", f"{info['beta']:.2f}" if info['beta'] != 'N/A' else "N/A")
        
        # 績效比較圖
        st.markdown("### 🆚 績效對決：個股 vs 大盤")
        stock_norm = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        if market_df is not None:
            market_aligned = market_df.reindex(df.index, method='ffill')
            market_norm = (market_aligned / market_aligned.iloc[0] - 1) * 100
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=df.index, y=stock_norm, mode='lines', name=info['name'], line=dict(color='red')))
            fig_compare.add_trace(go.Scatter(x=df.index, y=market_norm, mode='lines', name='大盤', line=dict(color='gray', dash='dash')))
            fig_compare.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")
        
        # K線與財報
        col_chart, col_fund = st.columns([2, 1])
        with col_chart:
            st.markdown("### 🕯️ K線圖")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='月線'), row=1, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_fund:
            st.markdown("### 💰 財報趨勢")
            fin = info['financial_data']
            if len(fin['years']) > 0:
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=fin['years'], y=fin['revenues'], name='營收'))
                fig_fin.add_trace(go.Bar(x=fin['years'], y=fin['earnings'], name='淨利'))
                fig_fin.update_layout(barmode='group', height=500)
                st.plotly_chart(fig_fin, use_container_width=True)
            else:
                st.info("無財報數據")

        # 新聞
        with st.expander(f"📰 {info['name']} 最新新聞"):
            news_list = get_news(info['name'])
            for news in news_list:
                st.write(f"- [{news.title}]({news.link})")

    else:
        st.error("⚠️ 無法讀取資料，請確認股票代號是否正確 (例如 2330 或 2330.TW)。")

# --- TAB 2: AI 修正版 ---
with tab2:
    # 這裡加入更嚴格的檢查
    if df is not None and info is not None:
        st.subheader(f"🤖 AI 預測實驗室：{info['name']}")
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
                    st.error("資料不足，無法預測")
    else:
        st.warning("請先在側邊欄輸入正確的股票代號。")

# --- TAB 3 ---
with tab3:
    st.subheader("🎯 智慧選股雷達")
    target_stocks = ['2330', '2317', '2454', '2308', '2603', '2609', '2881', '2882', '2412', '1605']
    if st.button("📡 掃描熱門股"):
        results = []
        bar = st.progress(0)
        for i, code in enumerate(target_stocks):
            bar.progress((i+1)/len(target_stocks))
            try:
                d = yf.download(code+".TW", period="1mo", progress=False)
                if d.empty: continue
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                
                sma5 = ta.sma(d['Close'], 5).iloc[-1]
                sma20 = ta.sma(d['Close'], 20).iloc[-1]
                prev_sma5 = ta.sma(d['Close'], 5).iloc[-2]
                prev_sma20 = ta.sma(d['Close'], 20).iloc[-2]
                rsi = ta.rsi(d['Close'], 14).iloc[-1]
                
                sig = ""
                if prev_sma5 <= prev_sma20 and sma5 > sma20: sig = "🔥 黃金交叉"
                elif rsi < 25: sig = "💎 RSI超賣"
                
                if sig:
                    results.append({"代號": code, "現價": f"{d['Close'].iloc[-1]:.2f}", "訊號": sig})
            except: continue
        
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("無符合條件股票")
