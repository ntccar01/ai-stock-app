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
import requests
from FinMind.data import DataLoader
import xgboost as xgb
import time

# --- 1. 網頁設定 ---
st.set_page_config(page_title="AI 股市操盤手 (除錯模式)", layout="wide")
st.title("🛠️ 系統除錯模式 (Debug Mode)")

# --- 2. 核心函式 (已加入除錯訊息) ---

def get_stock_info(code, search_code):
    data = {"name": code, "pe": "N/A", "yield": 0, "eps": "N/A"}
    try:
        if code in twstock.codes:
            data["name"] = twstock.codes[code].name
        stock = yf.Ticker(search_code)
        info = stock.info
        data["pe"] = info.get('trailingPE', 'N/A')
        data["eps"] = info.get('trailingEps', 'N/A')
        return data
    except:
        return data

def get_data(ticker_symbol, start_date):
    status = st.empty() # 建立一個空位顯示狀態
    status.write(f"⏳ 正在嘗試從 FinMind 下載 {ticker_symbol}...")
    
    try:
        stock_id = ticker_symbol.split('.')[0]
        start_str = start_date.strftime('%Y-%m-%d')
        
        dl = DataLoader()
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_str)
        
        if df.empty:
            status.write("⚠️ FinMind 無資料，轉用 Yahoo Finance...")
            return get_data_yahoo_backup(ticker_symbol, start_date)

        status.write("✅ FinMind 下載成功！處理數據中...")
        df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        
        # 技術指標
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        status.empty() # 清除狀態訊息
        return df.dropna()
        
    except Exception as e:
        status.error(f"❌ FinMind 下載失敗: {e}")
        return get_data_yahoo_backup(ticker_symbol, start_date)

def get_data_yahoo_backup(ticker_symbol, start):
    st.write("⏳ 正使用 Yahoo 下載 (備援)...")
    try:
        df = yf.download(ticker_symbol, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: 
            st.error("❌ Yahoo 也抓不到資料")
            return None
        
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        return df.dropna()
    except Exception as e:
        st.error(f"❌ Yahoo 下載失敗: {e}")
        return None

# --- 除錯版 AI 函式 (顯示詳細步驟) ---
def train_and_predict_debug(df):
    debug_log = st.expander("🕵️ AI 運算詳細日誌", expanded=True)
    with debug_log:
        st.write("1. 開始特徵工程...")
        try:
            df_ml = df.copy()
            df_ml['Close_Lag1'] = df_ml['Close'].shift(1)
            df_ml['Volume_Lag1'] = df_ml['Volume'].shift(1)
            df_ml['Bias_20'] = (df_ml['Close'] - df_ml['SMA_20']) / df_ml['SMA_20']
            df_ml['Target'] = df_ml['Close'].shift(-1)
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'Close_Lag1', 'Bias_20']
            data = df_ml.dropna()
            
            if len(data) < 30:
                st.error(f"❌ 有效資料不足 (只有 {len(data)} 筆)，無法訓練")
                return 0, 0
            
            X = data[features]
            y = data['Target']
            split = int(len(X) * 0.9)
            
            st.write(f"2. 資料準備完成 (訓練集: {split} 筆)")
            st.write("3. 正在初始化 XGBoost 模型...")
            
            # 簡化參數以加快測試速度
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3, objective='reg:squarederror')
            
            st.write("4. 開始訓練 (Model Fitting)...")
            model.fit(X.iloc[:split], y.iloc[:split])
            
            st.write("5. 訓練完成！正在預測...")
            last_row = df_ml.iloc[[-1]][features]
            pred = model.predict(last_row)[0]
            score = model.score(X.iloc[split:], y.iloc[split:])
            
            st.success(f"✅ 運算成功！預測值: {pred}")
            return pred, score
            
        except Exception as e:
            st.error(f"❌ AI 發生錯誤: {str(e)}")
            # 這裡會把詳細錯誤印出來，讓我們知道是不是缺少套件
            st.exception(e) 
            return 0, 0

# --- 介面邏輯 ---
st.sidebar.header("🔍 設定")
ticker = st.sidebar.text_input("代號", "2330.TW")
start_date = datetime.now() - timedelta(days=365)

if st.button("🚀 開始測試"):
    df = get_data(ticker, start_date)
    
    if df is not None:
        st.subheader("📊 數據預覽")
        st.dataframe(df.tail())
        
        st.subheader("🤖 AI 測試區")
        pred, acc = train_and_predict_debug(df)
        
        if pred > 0:
            st.metric("預測結果", f"{pred:.2f}", f"準確度: {acc:.2f}")
    else:
        st.error("無法取得數據，請檢查代號或網路。")
