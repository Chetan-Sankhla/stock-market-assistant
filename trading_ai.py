import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import uuid
from pytz import timezone
from datetime import datetime, timedelta
from pattern_detectors import detect_patterns
from candlestick_pattern_detection import detect_candle_patterns
from trade_finder import find_trades, find_trades_1, find_trades_2
from trade_finder_helpers import get_trade_mode

# Function to download stock data
def download_data(ticker, period="180d", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        data.index = data.index.tz_convert('Asia/Kolkata')

    data["Formatted Date"] = data.index.strftime('%d-%m-%Y %H:%M:%S')

    return data

# Function to calculate indicators (Pivot, ATR, EMA)
def calculate_indicators(data):
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['EMA_20'] = ta.ema(data['Close'], length=20)
    data['EMA_50'] = ta.ema(data['Close'], length=50)
    data['Volume_MA'] = ta.sma(data['Volume'], length=20)
    data['Swing_High'] = data["High"].rolling(window=10).max().shift(1)
    data['Swing_Low'] = data["Low"].rolling(window=10).min().shift(1)

    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    # RSI Calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Average volume for volume spike detection
    data['AvgVolume20'] = data['Volume'].rolling(window=20).mean()

    # ADX Calculation
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
    if adx is not None:
        data['ADX'] = adx['ADX_14']  # or adx.iloc[:, 0] if columns are uncertain

    # MACD and Signal line
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']


    return data

def generate_signals(data, rsi_thresh=30):
    data['Signal'] = 0

    # Ensure all required data is available
    aligned = data[['Close', 'SMA50', 'RSI', 'Volume', 'AvgVolume20']].notnull().all(axis=1)

    # Apply conditions only where data is fully available
    valid_data = data[aligned]

    buy_conditions = (
        (valid_data['Close'] > valid_data['SMA50']) &
        (valid_data['RSI'] < rsi_thresh) &
        (valid_data['Volume'] > 1.5 * valid_data['AvgVolume20'])
    )

    data.loc[valid_data[buy_conditions].index, 'Signal'] = 1
    return data

def train_ml_model(snapshots, trades):
    if len(snapshots) != len(trades):
        st.error("Mismatch in number of snapshots and trades.")
        return None

    for i, trade in enumerate(trades.itertuples()):
        snapshots[i]["label"] = 1 if trade.pnl > 0 else 0

    df_snapshots = pd.DataFrame(snapshots)
    X = df_snapshots.drop("label", axis=1)
    y = df_snapshots["label"]

    if len(X) > 1:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "rf_trade_model.pkl")
    joblib.dump(X_train.columns.tolist(), "rf_features.pkl")
    return model

# Updated Prediction Function
def predict_trade_success(snapshot):
    model = joblib.load("rf_trade_model.pkl")
    features = joblib.load("rf_features.pkl")
    features = [f for f in features if f in snapshot]  # keep only keys present in snapshot
    entry_df = pd.DataFrame([snapshot])[features]

    prediction = model.predict(entry_df)[0]
    proba_array = model.predict_proba(entry_df)[0]
    class_labels = model.classes_
    class_index = list(class_labels).index(prediction)
    proba = proba_array[class_index]
    return prediction, proba

# Function to show net P&L graph
def plot_net_pnl(trades):
    # Ensure pnl is numeric
    trades = trades.copy()
    trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce')
    trades = trades.dropna(subset=['pnl'])

    pnl = trades['pnl'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pnl, label="Net PnL", color='green')
    ax.set_title("Net PnL Over Time")
    ax.set_xlabel("Trades")
    ax.set_ylabel("Cumulative PnL")
    ax.legend()
    
    st.pyplot(fig)

def scan_stocks(data, ticker):
    potential_buys = []
    try:
        data = calculate_indicators(data)
        data = generate_signals(data)

        if data.empty or 'Close' not in data.columns:
            st.write(f"❌ No data or missing 'Close' for {ticker}")
            return potential_buys

        last_row = data.iloc[-1]
        if last_row['Signal'] == 1:
            st.write(f"✅ Buy signal for {ticker}")

            trade_levels = calculate_trade_levels(last_row)
            if trade_levels:
                st.write(f"📈 Entry Price: ₹{trade_levels['entry']:.2f}")
                st.write(f"🛑 Stop Loss: ₹{trade_levels['stop_loss']:.2f}")
                st.write(f"🎯 Target: ₹{trade_levels['target']:.2f}")
                st.write(f"💰 Risk/Reward Ratio: {trade_levels['rr_ratio']:.2f}")

                potential_buys.append({
                    'ticker': ticker,
                    **trade_levels
                })
            else:
                st.warning(f"⚠️ Not enough data to calculate trade levels for {ticker}")
        else:
            st.write(f"📉 No Buy Signal for {ticker}")
    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {type(e).__name__} - {e}")

    return potential_buys

@st.cache_data
def load_nse_tickers():
    df = pd.read_csv("Yahoo-Finance-Ticker-Symbols.csv")  # Or use a remote URL
    return df

# Download and process data (cached version if desired)
@st.cache_data(ttl=600*5)  # Cache for 30 minutes
def download_data_cached(ticker, period, interval):
    return download_data(ticker, period, interval)

def calculate_trade_levels(row):
    atr = row['ATR']
    swing_low = row['Swing_Low']
    swing_high = row['Swing_High']
    close = row['Close']

    if pd.isna([atr, swing_low, swing_high, close]).any():
        return None  # Incomplete data

    entry_price = close
    stop_loss = swing_low - 0.5 * atr
    risk = entry_price - stop_loss
    target = entry_price + 2 * risk
    rr_ratio = (target - entry_price) / (entry_price - stop_loss)

    return {
        "entry": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "rr_ratio": rr_ratio
    }

def show_candlestick_chart(data, patterns=None, candlestick_patterns=True, trades=None, active_trades=None):
    fig = go.Figure()
    draw_trade_annotations(fig, trades)
    # Plot candlestick data
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Candlesticks"
    ))

    # Plot EMAs or other indicators
    if "EMA_20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["EMA_20"],
            mode='lines', name='EMA 20', line=dict(color='blue', width=1)))

    if "EMA_50" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["EMA_50"],
            mode='lines', name='EMA 50', line=dict(color='red', width=1)))

    # Highlight pattern components (like price action patterns) if provided
    if patterns:
        for pattern_name, matches in patterns.items():
            #print(f"1: {pattern_name}, {matches}")
            for pattern_data in matches:
                components = pattern_data.get("components", {})
                #print(f"2: {components}")
                times = []
                prices = []
                for label, point in components.items():
                    #print(f"3: {label}, {point}")
                    if isinstance(point, dict):
                        #print(f"4")
                        t = point.get('time') or point.get('date')
                        #print(f"5: {t}")
                        if t is not None:
                            times.append(t)
                            #print(f"6: {t}")
                            #fig.add_vline(x=t, line=dict(color="purple", width=2, dash="dot"))
                            price = point.get('price')
                            #print(f"7: {price}")
                            if price is not None:
                                prices.append(price)
                                #print(f"8: {price}")
                                #fig.add_annotation(
                                #    x=t, y=price,
                                #    text=f"{label.replace('_', ' ').title()}",
                                #    showarrow=True, arrowhead=2, ax=0, ay=-40
                                #)
                    elif isinstance(point, tuple) and "times" in point and "prices" in point:
                        times = point.get("times", ())
                        prices = point.get("prices", ())
                        #print(f"9: {times}, {prices}")
                        for t, p in zip(times, prices):
                            times.append(t)
                            prices.append(p)
                            #fig.add_vline(x=t, line=dict(color="orange", width=1, dash="dash"))
                            #fig.add_annotation(
                            #    x=t, y=p,
                            #    text=f"{pattern_name}: Neckline",
                            #    showarrow=True, arrowhead=1, ax=0, ay=30
                            #)
                if len(times) >= 2:
                    fig.add_trace(go.Scatter(
                        x=times,
                        y= prices,
                        mode='lines+markers+text',
                        name=pattern_name,
                        line=dict(width=3, color='black', dash='longdashdot'),
                        marker=dict(size=6),
                        textposition="top center"
                    ))
    #print(f"7")
    # Final layout
    fig.update_layout(
        title="📊 Candlestick Chart with Indicators and Patterns",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show candlestick patterns in a table below
    if "patterns" in data.columns:
        pattern_rows = [
            {"Date": idx.strftime("%Y-%m-%d %H:%M:%S"), "Patterns": ", ".join(patterns)}
            for idx, patterns in data["patterns"].items()
            if patterns  # only non-empty
        ]
        if pattern_rows:
            st.markdown("### 📋 Detected Candlestick Patterns")
            pattern_df = pd.DataFrame(pattern_rows)
            st.dataframe(pattern_df, height=400, use_container_width=True)
            

def show_patterns_data(patterns):
    if patterns:
        st.subheader("🔍 Detected Chart Patterns")

        for pattern_name, occurrences in patterns.items():
            if occurrences:
                st.write(f"✅ {pattern_name}: {len(occurrences)} detected")

                for occ in occurrences:
                    components = occ.get("components", {})
                    head_date = occ.get("date", "")
                    head_price = occ.get("price", 0)

                    # Tabular format for H&S or Inverse H&S
                    if all(k in components for k in ["left_shoulder", "mid_point_1", "head", "mid_point_2", "right_shoulder"]):
                        row = {
                            "Left Shoulder": f"₹{components['left_shoulder']['price']:.2f} / {components['left_shoulder']['date']}",
                            "Mid Point 1": f"₹{components['mid_point_1']['price']:.2f} / {components['mid_point_1']['date']}",
                            "Head": f"₹{components['head']['price']:.2f} / {components['head']['date']}",
                            "Mid Point 2": f"₹{components['mid_point_2']['price']:.2f} / {components['mid_point_2']['date']}",
                            "Right Shoulder": f"₹{components['right_shoulder']['price']:.2f} / {components['right_shoulder']['date']}"
                        }
                        df_row = pd.DataFrame([row])
                        st.table(df_row)

                    else:
                        # For other patterns, display key-value component details
                        #formatted_head_date = head_date.strftime("%Y-%m-%d %H:%M:%S")
                        #print(f"{head_date}")
                        st.markdown(f"**Detected at:** {head_date} | Price: ₹{head_price:.2f}")
                        flat_data = {
                            k.replace("_", " ").title(): f"₹{v['price']:.2f} / {v['time']}"
                            for k, v in components.items()
                            if isinstance(v, dict) and "price" in v and "time" in v
                        }
                        df_flat = pd.DataFrame([flat_data])
                        st.table(df_flat)
            else:
                st.write(f"❌ No {pattern_name} detected.")

def show_auto_refresh():
    REFRESH_INTERVAL = 5  # in minutes

    # Initialize session state for refresh tracking
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now() - timedelta(minutes=REFRESH_INTERVAL)

    # Timer display for user
    time_left = REFRESH_INTERVAL * 60 - (datetime.now() - st.session_state.last_refresh_time).seconds
    minutes, seconds = divmod(time_left, 60)
    st.info(f"⏳ Next auto-refresh in {minutes}m {seconds}s")

    # Auto-refresh after 5 minutes
    if datetime.now() - st.session_state.last_refresh_time >= timedelta(minutes=REFRESH_INTERVAL):
        st.session_state.last_refresh_time = datetime.now()
        st.rerun()

def show_pattern_detection_stats(data):
    # 🔍 Pattern Detection
    st.subheader("🔍 Pattern Detection")
    patterns = detect_patterns(data)

    if patterns:
        for p in patterns:
            st.success(f"Pattern Detected: {p}")
    else:
        st.info("No known chart patterns detected in the selected timeframe.")
    
    return patterns

def show_potential_buys(data, ticker):
    potential_buys = scan_stocks(data, ticker)
    st.subheader("🔥 Stocks with Buy Signals:")
    for trade in potential_buys:
        st.markdown(f"""
        ### {trade['ticker']}
        - 📈 Entry Price: ₹{trade['entry']:.2f}
        - 🛑 Stop Loss: ₹{trade['stop_loss']:.2f}
        - 🎯 Target: ₹{trade['target']:.2f}
        - 💰 Risk/Reward Ratio: {trade['rr_ratio']:.2f}
        """)

def show_trades_and_prediction(data, period, interval):
# Detect trades and prepare snapshots
    trades, snapshots, active_trades = find_trades(data, period, interval)

    # Display Active Trade
    if active_trades is not None:
        st.subheader("📈 Active Trade Signal Method 0")
        st.write(f"**Type**: {active_trades['type'].capitalize()}")
        st.write(f"**Entry Date**: {active_trades['entry_date'].strftime('%d-%m-%Y %H:%M:%S')}")
        st.write(f"**Entry Price**: {active_trades['entry_price']:.2f}")
        st.write(f"**Stoploss**: {active_trades['stop_loss']:.2f}")
        st.write(f"**Target**: {active_trades['target']:.2f}")

        # Prepare snapshot for prediction
        index = data.index.get_loc(active_trades['entry_date'])
        row = data.iloc[index]
        rsi_val = ta.rsi(data['Close'], length=14)
        rsi_value = rsi_val.iloc[index] if not pd.isna(rsi_val.iloc[index]) else None
        temp_snapshot = {
            'ema_diff': row['EMA_20'] - row['EMA_50'] if pd.notna(row['EMA_20']) and pd.notna(row['EMA_50']) else None,
            'atr': row['ATR'] if pd.notna(row['ATR']) else None,
            'rsi': rsi_value,
            'price': row['Close'],
            'mode': get_trade_mode(interval),
            'position': 1
        }

        # Predict
        snapshot_for_prediction = {k: v for k, v in temp_snapshot.items() if k != 'position'}
        try:
            prediction, proba = predict_trade_success(snapshot_for_prediction)
            st.write("🤖 Model Prediction")
            if isinstance(proba, (float, int)):
                if prediction == 1:
                    st.success(f"✅ Likely Profitable Trade ({proba * 100:.2f}% confidence)")
                else:
                    st.warning(f"⚠️ Likely Unprofitable Trade ({proba * 100:.2f}% confidence)")

                color = "green" if prediction == 1 else "red"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Trade Success Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 50], "color": "lightcoral"},
                            {"range": [50, 100], "color": "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
            else:
                st.warning(f"Model returned non-probability output: {proba}")
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
    else:
        st.info("ℹ️ No Active Trade signal found")

    # Show past trades
    st.write(f"Total Trades: {len(trades)}")
    if len(trades) > 0:
        win_rate = (len(trades[trades['pnl'] > 0]) / len(trades)) * 100
        st.write(f"Win Rate: {win_rate:.2f}%")
        st.subheader("📋 Trade List")
        st.dataframe(trades)
        plot_net_pnl(trades)

        # Train model
        if len(snapshots) > 0:
            model = train_ml_model(snapshots, trades)
            if model:
                st.success("✅ Model trained successfully on trade outcomes.")

    return trades, active_trades, snapshots

def show_trades_and_prediction_1(data, period, interval):
    # Detect trades and prepare snapshots
    trades, snapshots, active_trades = find_trades_1(data, period, interval)

    # Display Active Trade
    if active_trades is not None and len(active_trades) > 0:
        st.subheader("📈 Active Trade Signals Method 1")

    for i, trade in enumerate(active_trades if isinstance(active_trades, list) else active_trades.to_dict("records")):
        st.markdown(f"### 🔹 Trade #{i+1} - {trade['type'].capitalize()}")

        st.write(f"**Entry Date**: {trade['entry_date'].strftime('%d-%m-%Y %H:%M:%S')}")
        st.write(f"**Entry Price**: {trade['entry_price']:.2f}")
        st.write(f"**Stoploss**: {trade['stop_loss']:.2f}")
        st.write(f"**Target**: {trade['target']:.2f}")

        # Prepare snapshot for prediction
        try:
            index = data.index.get_loc(trade['entry_date'])
            row = data.iloc[index]
        except KeyError:
            st.warning("⚠️ Entry date not in data range. Skipping prediction.")
            continue

        rsi_val = ta.rsi(data['Close'], length=14)
        rsi_value = rsi_val.iloc[index] if not pd.isna(rsi_val.iloc[index]) else None
        temp_snapshot = {
            'ema_diff': row['EMA_20'] - row['EMA_50'] if pd.notna(row['EMA_20']) and pd.notna(row['EMA_50']) else None,
            'atr': row['ATR'] if pd.notna(row['ATR']) else None,
            'rsi': rsi_value,
            'price': row['Close'],
            'mode': get_trade_mode(interval),
            'position': 1
        }

        # Predict
        snapshot_for_prediction = {k: v for k, v in temp_snapshot.items() if k != 'position'}
        try:
            prediction, proba = predict_trade_success(snapshot_for_prediction)
            st.write("🤖 Model Prediction")

            if isinstance(proba, (float, int)):
                if prediction == 1:
                    st.success(f"✅ Likely Profitable Trade ({proba * 100:.2f}% confidence)")
                else:
                    st.warning(f"⚠️ Likely Unprofitable Trade ({proba * 100:.2f}% confidence)")

                color = "green" if prediction == 1 else "red"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Trade Success Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 50], "color": "lightcoral"},
                            {"range": [50, 100], "color": "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
            else:
                st.warning(f"Model returned non-probability output: {proba}")
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
    else:
        st.info("ℹ️ No Active Trade signals found.")


    # Show past trades
    st.write(f"Total Trades: {len(trades)}")
    if len(trades) > 0:
        win_rate = (len(trades[trades['pnl'] > 0]) / len(trades)) * 100
        st.write(f"Win Rate: {win_rate:.2f}%")
        st.subheader("📋 Trade List")
        st.dataframe(trades)
        plot_net_pnl(trades)

        # Train model
        if len(snapshots) > 0:
            model = train_ml_model(snapshots, trades)
            if model:
                st.success("✅ Model trained successfully on trade outcomes.")

    return trades, active_trades, snapshots

def show_trades_and_prediction_2(data, period, interval):
    # Detect trades and prepare snapshots
    trades, snapshots, active_trades = find_trades_2(data, period, interval)

    # Display Active Trade
    if active_trades is not None and len(active_trades) > 0:
        st.subheader("📈 Active Trade Signals Method 2")

    for i, trade in enumerate(active_trades if isinstance(active_trades, list) else active_trades.to_dict("records")):
        st.markdown(f"### 🔹 Trade #{i+1} - {trade['type'].capitalize()}")

        st.write(f"**Entry Date**: {trade['entry_date'].strftime('%d-%m-%Y %H:%M:%S')}")
        st.write(f"**Entry Price**: {trade['entry_price']:.2f}")
        st.write(f"**Stoploss**: {trade['stop_loss']:.2f}")
        st.write(f"**Target**: {trade['target']:.2f}")

        # Prepare snapshot for prediction
        try:
            index = data.index.get_loc(trade['entry_date'])
            row = data.iloc[index]
        except KeyError:
            st.warning("⚠️ Entry date not in data range. Skipping prediction.")
            continue

        rsi_val = ta.rsi(data['Close'], length=14)
        rsi_value = rsi_val.iloc[index] if not pd.isna(rsi_val.iloc[index]) else None
        temp_snapshot = {
            'ema_diff': row['EMA_20'] - row['EMA_50'] if pd.notna(row['EMA_20']) and pd.notna(row['EMA_50']) else None,
            'atr': row['ATR'] if pd.notna(row['ATR']) else None,
            'rsi': rsi_value,
            'price': row['Close'],
            'mode': get_trade_mode(interval),
            'position': 1
        }

        # Predict
        snapshot_for_prediction = {k: v for k, v in temp_snapshot.items() if k != 'position'}
        try:
            prediction, proba = predict_trade_success(snapshot_for_prediction)
            st.write("🤖 Model Prediction")

            if isinstance(proba, (float, int)):
                if prediction == 1:
                    st.success(f"✅ Likely Profitable Trade ({proba * 100:.2f}% confidence)")
                else:
                    st.warning(f"⚠️ Likely Unprofitable Trade ({proba * 100:.2f}% confidence)")

                color = "green" if prediction == 1 else "red"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Trade Success Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 50], "color": "lightcoral"},
                            {"range": [50, 100], "color": "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
            else:
                st.warning(f"Model returned non-probability output: {proba}")
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
    else:
        st.info("ℹ️ No Active Trade signals found.")


    # Show past trades
    st.write(f"Total Trades: {len(trades)}")
    if len(trades) > 0:
        win_rate = (len(trades[trades['pnl'] > 0]) / len(trades)) * 100
        st.write(f"Win Rate: {win_rate:.2f}%")
        st.subheader("📋 Trade List")
        st.dataframe(trades)
        plot_net_pnl(trades)

        # Train model
        if len(snapshots) > 0:
            model = train_ml_model(snapshots, trades)
            if model:
                st.success("✅ Model trained successfully on trade outcomes.")

    return trades, active_trades, snapshots

def show_predictions(snapshots):
    # Show predictions for snapshots
        for i, snapshot in enumerate(snapshots):
            try:
                pred, proba = predict_trade_success(snapshot)
                trade_type = "Profitable" if pred == 1 else "Unprofitable"
                color = "green" if pred == 1 else "red"

                st.markdown(f"**Prediction for trade at price {snapshot['price']:.2f}: {trade_type}**")
                st.write(snapshot)

                if isinstance(proba, (float, int)):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        title={"text": "Trade Success Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 50], "color": "lightcoral"},
                                {"range": [50, 100], "color": "lightgreen"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, key=str(uuid.uuid4()))
                else:
                    st.warning(f"Categorical model output: {proba}")

            except Exception as e:
                st.error(f"Snapshot prediction failed: {e}")

def draw_trade_annotations(fig, trades):
    #print(trades.head())  # Check if it's a DataFrame
    #print(type(trades))
    if isinstance(trades, pd.DataFrame):
        trades_iter = trades.iterrows()
    elif isinstance(trades, list):
        trades_iter = enumerate(trades)
    else:
        st.error("❌ Unsupported trades format")
        return

    for _, trade in trades_iter:
        entry_time = trade['entry_date']
        exit_time = trade['exit_date']
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        target = trade['target']

        # 🟩 Draw profit zone
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=entry_time, x1=exit_time,
            y0=entry_price, y1=target,
            fillcolor="rgba(0, 200, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )

        # 🟥 Draw risk zone
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=entry_time, x1=exit_time,
            y0=stop_loss, y1=entry_price,
            fillcolor="rgba(200, 0, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )

        # 🔵 Entry marker
        fig.add_trace(go.Scatter(
            x=[entry_time], y=[entry_price],
            mode='markers+text',
            name='Entry',
            marker=dict(color='blue', size=8, symbol='triangle-up'),
            text=["Entry"], textposition="top center"
        ))

        # 🟠 Exit marker
        fig.add_trace(go.Scatter(
            x=[exit_time], y=[trade['exit_price']],
            mode='markers+text',
            name='Exit',
            marker=dict(color='orange', size=8, symbol='x'),
            text=["Exit"], textposition="top center"
        ))

        # 🔴 Stop Loss marker
        fig.add_trace(go.Scatter(
            x=[entry_time], y=[stop_loss],
            mode='markers+text',
            name='Stop Loss',
            marker=dict(color='red', size=6, symbol='triangle-down'),
            text=["SL"], textposition="bottom center"
        ))

def main():
    show_auto_refresh()

    # Sidebar section
    st.sidebar.header("📊 Configuration")

    df = load_nse_tickers()
    nse_df = df[df["Exchange"] == "NSI"].sort_values(by="Name")
    options = (nse_df['Name'] + " - " + nse_df['Ticker']).tolist()

    default_index = next((i for i, option in enumerate(options) if "NIFTY 50 - ^NSEI" in option), 0)

    selected_stock = st.sidebar.selectbox("Select Stock", options=options, index=default_index)
    ticker = selected_stock.split(" - ")[1]

    period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max"]
    interval_options = ["15m", "30m", "1h", "1d", "1m", "2m", "5m", "90m", "4h", "5d", "1wk", "1mo", "3mo"]

    period = st.sidebar.radio("Select Period", period_options, index=period_options.index("1d"))
    interval = st.sidebar.radio("Select Interval", interval_options, index=interval_options.index("15m"))

    # Toggle options
    st.sidebar.markdown("### 🧩 Display Options")
    show_trades = st.sidebar.checkbox("Show Trades", value=True)
    show_patterns = st.sidebar.checkbox("Show Chart Patterns", value=False)
    show_candles = st.sidebar.checkbox("Show Candlestick Patterns", value=False)
    show_buys = st.sidebar.checkbox("Show Potential Buys", value=False)
    show_predictions_results = st.sidebar.checkbox("Show ML Prediction Results", value=False)

    # Main section
    data = download_data_cached(ticker, period, interval)
    data = calculate_indicators(data)

    trades, active_trades, snapshots = show_trades_and_prediction(data, period, interval) if show_trades else ([], [],[])
    patterns = show_pattern_detection_stats(data) if show_patterns else None
    candlestick_patterns = detect_candle_patterns(data) if show_candles else None

    show_candlestick_chart(data, patterns, candlestick_patterns, trades, active_trades)
    #show_candlestick_chart(data, patterns, candlestick_patterns, trades_1, active_trades_1)

    trades_1, active_trades_1, snapshots_1 = show_trades_and_prediction_1(data, period, interval) if show_trades else ([], [],[])
    trades_2, active_trades_2, snapshots_2 = show_trades_and_prediction_2(data, period, interval) if show_trades else ([], [],[])

    if patterns is not None:
        show_patterns_data(patterns)

    if show_buys:
        show_potential_buys(data, ticker)
    
    if show_predictions_results:
        show_predictions(snapshots)



if __name__ == "__main__":
    st.title("FNO Trading Dashboard with Machine Learning")
    main()
