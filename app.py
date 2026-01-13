import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock
import urllib.parse
from datetime import datetime, timedelta
import requests
from FinMind.data import DataLoader
import xgboost as xgb
import time

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 股市操盤手 (輕量極速版)", layout="wide")

# CSS 優化介面
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    /* 讓進度條更明顯 */
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# 左側狀態列
status_text = st.sidebar.empty()

# --- 2. 核心函式 ---

@st.cache_data(ttl=3600)
def get_stock_info(code, search_code):
    data = {"name": code, "pe": "N/A", "yield": 0, "eps": "N/A"}
    try:
        if code in twstock.codes:
            data["name"] = twstock.codes[code].name
        
        stock = yf.Ticker(search_code)
        info = stock.info
        
        data["pe"] = info.get('trailingPE', 'N/A')
        data["eps"] = info.get('trailingEps', 'N/A')
        div = info.get('dividendRate', 0)
        pri = info.get('currentPrice') or info.get('previousClose')
        if div and pri: data["yield"] = div / pri
        return data
    except:
        return data

@st.cache_data(ttl=3600)
def get_data(ticker_symbol, start_date):
    try:
        stock_id = ticker_symbol.split('.')[0]
        start_str = start_date.strftime('%Y-%m-%d')
        
        dl = DataLoader()
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_str)
        
        if df.empty:
            return get_data_yahoo_backup(ticker_symbol, start_date)

        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 
            'close': 'Close', 'Trading_Volume': 'Volume'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # 技術指標
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        return df.dropna()
    except:
        return None

def get_data_yahoo_backup(ticker_symbol, start):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        df = yf.download(ticker_symbol, start=start, progress=False, session=session)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
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

# --- 優化後的 AI 函式 (輕量化 + 進度顯示) ---
def train_and_predict_light(df, status_placeholder):
    try:
        status_placeholder.write("🔄 步驟 1/4: 整理數據中...")
        df_ml = df.copy()
        df_ml['Close_Lag1'] = df_ml['Close'].shift(1)
        df_ml['Volume_Lag1'] = df_ml['Volume'].shift(1)
        df_ml['RSI_Lag1'] = df_ml['RSI'].shift(1)
        df_ml['Bias_20'] = (df_ml['Close'] - df_ml['SMA_20']) / df_ml['SMA_20']
        df_ml['Target'] = df_ml['Close'].shift(-1)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'Close_Lag1', 'Bias_20']
        
        data = df_ml.dropna()
        X = data[features]
        y = data['Target']

        if len(X) < 30: return 0, 0

        split = int(len(X) * 0.9)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        status_placeholder.write("🧠 步驟 2/4: 建構神經網路 (XGBoost)...")
        
        # --- 關鍵修改：降低參數負擔，適應雲端環境 ---
        model = xgb.XGBRegressor(
            n_estimators=200,      # 從 1000 降到 200 (大幅加速)
            learning_rate=0.05,    # 稍微提高學習率以彌補樹的減少
            max_depth=4,           # 降低深度防止記憶體溢出
            early_stopping_rounds=20,
            objective='reg:squarederror',
            n_jobs=1,              # 強制單核心運行，避免多執行緒卡死
            random_state=42
        )

        status_placeholder.write("🚀 步驟 3/4: 開始訓練模型...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        status_placeholder.write("✨ 步驟 4/4: 進行最終預測...")
        score = model.score(X_test, y_test)
        last_row = df_ml.iloc[[-1]][features]
        pred = model.predict(last_row)[0]
        
        return pred, score

    except Exception as e:
        print(f"AI Error: {e}")
        return 0, 0

# --- 3. 介面邏輯 ---
st.sidebar.header("🔍 設定與搜尋")
ticker_input = st.sidebar.text_input("輸入代號", value="2330")
time_range = st.sidebar.radio("區間", ["6個月", "1年", "3年"], index=1)

end_date = datetime.now()
if time_range == "6個月": start_date = end_date - timedelta(days=180)
elif time_range == "1年": start_date = end_date - timedelta(days=365)
else: start_date = end_date - timedelta(days=1095)

# 智慧代號處理
ticker_input = ticker_input.strip()
if ticker_input.isdigit():
    suffix = ".TW"
    if ticker_input in twstock.codes:
        if twstock.codes[ticker_input].type == "上櫃":
            suffix = ".TWO"
    ticker_search = ticker_input + suffix
    ticker_display = ticker_input
else:
    ticker_search = ticker_input
    ticker_display = ticker_input.split('.')[0]

# 抓取資料
status_text.text("⏳ 資料下載中...")
df = get_data(ticker_search, start_date)
status_text.text("⏳ 讀取基本面...")
info = get_stock_info(ticker_display, ticker_search)
status_text.text("⏳ 讀取大盤...")
market_df = get_market_data(start_date)
status_text.empty()

tab1, tab2, tab3 = st.tabs(["📊 綜合分析", "🧠 AI 預測 (XGBoost)", "🎯 智慧選股"])

# --- TAB 1: 圖表 ---
with tab1:
    if df is not None:
        st.subheader(f"📈 {info['name']} ({ticker_display})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("本益比", f"{info['pe']:.1f}" if info['pe'] != 'N/A' else "-")
        c2.metric("殖利率", f"{info['yield']*100:.2f}%" if info['yield'] else "-")
        c3.metric("EPS", f"{info['eps']:.2f}" if info['eps'] != 'N/A' else "-")
        c4.metric("收盤價", f"{df['Close'].iloc[-1]:.1f}")
        
        st.markdown("### 🕯️ K線圖")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='月線'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量'), row=2, col=1)
        fig.update_layout(height=450, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"❌ 無法讀取資料，請檢查代號。")

# --- TAB 2: AI 預測 ---
with tab2:
    if df is not None:
        st.subheader(f"🤖 AI 預測實驗室")
        st.markdown("說明：使用 XGBoost 演算法 (輕量版) 進行即時運算。")
        
        if st.button("🚀 執行 AI 運算", type="primary"):
            # 建立一個容器來顯示進度，而不是單純的 spinner
            progress_box = st.container()
            with progress_box:
                msg_slot = st.empty() # 佔位符
                
                # 呼叫函數，並傳入佔位符以便即時更新文字
                pred, acc = train_and_predict_light(df, msg_slot)
                
                if pred > 0:
                    msg_slot.success("✅ 運算完成！") # 最終替換文字
                    
                    last = df['Close'].iloc[-1]
                    chg = (pred - last) / last * 100
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("AI 預測價格", f"{pred:.2f}", f"{chg:.2f}%")
                    c2.metric("模型信心度 (R2)", f"{acc*100:.1f}%")
                    
                    if acc < 0:
                        st.caption("⚠️ 註：信心度為負值代表近期股價波動不規則，僅供參考。")
                else:
                    msg_slot.error("資料不足或運算失敗。")
    else:
        st.warning("請先輸入有效代號")

# --- TAB 3: 選股 ---
with tab3:
    st.subheader("🎯 快速選股")
    target_stocks = ['2330', '2317', '2454', '8069', '3293']
    if st.button("📡 開始掃描"):
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        start_scan = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})

        for i, code in enumerate(target_stocks):
            bar.progress((i+1)/len(target_stocks))
            status.text(f"正在分析 {code}...")
            try:
                suffix = ".TW"
                if code in twstock.codes and twstock.codes[code].type == "上櫃": suffix = ".TWO"
                
                d = yf.download(code + suffix, start=start_scan, progress=False, session=session)
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                
                if not d.empty and len(d) > 14:
                    close = d['Close'].iloc[-1]
                    rsi = ta.rsi(d['Close'], 14).iloc[-1]
                    name = twstock.codes[code].name if code in twstock.codes else code
                    results.append({"代號": code, "名稱": name, "現價": f"{close:.1f}", "RSI": round(rsi, 2)})
            except: continue
        
        status.empty()
        bar.empty()
        if results: st.dataframe(pd.DataFrame(results), use_container_width=True)
        else: st.warning("無結果")
