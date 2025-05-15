import pandas as pd

def is_index_data(data):
    if 'Volume' not in data.columns:
        return True
    # If volume is present but mostly NaN or zero, treat it as index
    volume_data = data['Volume']
    return volume_data.isna().all() or (volume_data.sum() == 0)

def classify_trade_context(current, atr):
    # Simple rule: if ATR is large relative to price or timeframe is higher, consider long-term
    if atr > 100:
        return "long_term"
    else:
        return "short_term"

def volume_confirmation(data, index, lookback=20):
    avg_volume = data['Volume'].iloc[index - lookback:index].mean()
    return data['Volume'].iloc[index] > 1.2 * avg_volume

def is_strong_trend(data, index):
    try:
        adx = data['ADX'].iloc[index]
        macd = data['MACD'].iloc[index]
        signal = data['MACD_signal'].iloc[index]
        if pd.isna(adx) or pd.isna(macd) or pd.isna(signal):
            return False
        return adx > 20 and macd > signal
    except:
        return False

def pattern_confidence(components):
    if not components:
        return 0
    return min(1.0, 0.2 * len(components))  # Each component adds 20% confidence

def is_good_time(timestamp):
    hour = timestamp.hour
    minute = timestamp.minute
    return (hour == 9 and minute >= 30) or (10 <= hour <= 14)

def enforce_risk_caps(sl, target, max_sl=100, max_target=200):
    sl = min(sl, max_sl)
    target = min(target, max_target)
    return sl, target

def detect_rsi_divergence(data, index):
    if index < 3:
        return False
    price_up = data['Close'].iloc[index] > data['Close'].iloc[index - 2]
    rsi_down = data['RSI'].iloc[index] < data['RSI'].iloc[index - 2]
    return price_up and rsi_down

def allow_entry_by_ml(snapshot, model):
    prob = model.predict_proba([list(snapshot.values())])[0][1]
    return prob >= 0.6

def calculate_position_size(capital, risk_per_trade, entry, sl):
    risk = abs(entry - sl)
    if risk == 0:
        return 0
    return int((capital * risk_per_trade) / risk)


def get_trade_mode(interval):
    if interval in ['1m', '2m', '5m', '15m', '30m', '1h']:
        return 'short_term'
    else:
        return 'long_term'

def calculate_sl_target(entry_price, swing, atr, mode):
    if mode == 'short_term':
        sl = swing - 0.5 * atr
        risk = entry_price - sl
        risk = min(risk, 100)  # Cap risk for short-term
        target = entry_price + 2 * risk
    else:  # long_term
        sl = swing - atr
        risk = entry_price - sl
        target = entry_price + 2.5 * risk  # Use higher R:R for longer horizon
    return sl, target

    # Function to detect trends (uptrend or downtrend)
def is_uptrend(data, index):
    closes = data['Close'].iloc[index-3:index]
    return closes.is_monotonic_increasing

def is_downtrend(data, index):
    closes = data['Close'].iloc[index-3:index]
    return closes.is_monotonic_decreasing

def has_valid_indicators(row):
    return not any(pd.isna(row[ind]) for ind in ['Pivot', 'EMA_20', 'ATR', 'Swing_Low', 'Swing_High'])