import pandas as pd
import pandas_ta as ta
from trade_finder_helpers import is_index_data,classify_trade_context,volume_confirmation,is_strong_trend,pattern_confidence,is_good_time,enforce_risk_caps,detect_rsi_divergence,allow_entry_by_ml,calculate_position_size,get_trade_mode,calculate_sl_target,is_uptrend,is_downtrend,has_valid_indicators

# Function to find trades based on the indicators
def find_trades(data, period, interval, trained_model=None, capital=100000):
    trades = []
    snapshots = []  # This will store the snapshots for training ML model
    active_trade = None

    for i in range(5, len(data) - 1):
        current = data.iloc[i]
        next_candle = data.iloc[i + 1]
        # Skip if any key indicators are missing
        if not has_valid_indicators(current):
            continue
        
        atr = current['ATR']
        candle_size = abs(current['Close'] - current['Open'])

        if candle_size < data.iloc[i]['ATR'] * 0.5:
            continue

        is_index = is_index_data(data)

        context = classify_trade_context(current, atr)
        if not is_index and not volume_confirmation(data, i):
            continue
        if not is_strong_trend(data, i):
            continue
        if not is_good_time(next_candle.name):
            continue
        if detect_rsi_divergence(data, i):
            continue

        if active_trade is None:
            if (current['Close'] > current['Pivot'] and
                next_candle['High'] >= current['High'] and
                current['Close'] > current['EMA_20'] and
                is_uptrend(data, i)):

                entry_price = current['High']
                stop_loss = current['Swing_Low'] - 0.5 * atr
                sl, target = enforce_risk_caps(entry_price - stop_loss, 2 * (entry_price - stop_loss))
                stop_loss = entry_price - sl
                target = entry_price + 2 * sl

                active_trade = {
                    'type': 'buy',
                    'entry_date': next_candle.name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target
                }


            elif (current['Close'] < current['Pivot'] and
                  next_candle['Low'] <= current['Low'] and
                  current['Close'] < current['EMA_20'] and
                  is_downtrend(data, i)):

                entry_price = current['Low']
                stop_loss = current['Swing_High'] + 0.5 * atr
                sl, target = enforce_risk_caps(stop_loss - entry_price, 2 * (stop_loss - entry_price))
                stop_loss = entry_price + sl
                target = entry_price - 2 * sl

                active_trade = {
                    'type': 'sell',
                    'entry_date': next_candle.name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target
                }

                # Optional: ML gating
                if trained_model:
                    snapshot = {
                    'ema_diff': current['EMA_20'] - current['EMA_50'],
                    'atr': atr,
                    'rsi': ta.rsi(data['Close'], length=14).iloc[i],
                    'price': current['Close']
                }
                    if not allow_entry_by_ml(snapshot, trained_model):
                        active_trade = None
                        continue


        elif active_trade:
            high = next_candle['High']
            low = next_candle['Low']
            exit_date = next_candle.name

            if active_trade['type'] == 'buy':
                if low <= active_trade['stop_loss']:
                    pnl = active_trade['stop_loss'] - active_trade['entry_price']
                    exit_price = active_trade['stop_loss']
                elif high >= active_trade['target']:
                    pnl = active_trade['target'] - active_trade['entry_price']
                    exit_price = active_trade['target']
                else:
                    continue

            elif active_trade['type'] == 'sell':
                if high >= active_trade['stop_loss']:
                    pnl = active_trade['entry_price'] - active_trade['stop_loss']
                    exit_price = active_trade['stop_loss']
                elif low <= active_trade['target']:
                    pnl = active_trade['entry_price'] - active_trade['target']
                    exit_price = active_trade['target']
                else:
                    continue

            trades.append({
                'type': active_trade['type'],
                'entry_date': active_trade['entry_date'],
                'exit_date': exit_date,
                'entry_price': active_trade['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'stop_loss': active_trade['stop_loss'],
                'target': active_trade['target']
            })
            success = 1 if pnl > 0 else 0
            snapshots.append({
                'ema_diff': current['EMA_20'] - current['EMA_50'],
                'atr': atr,
                'rsi': ta.rsi(data['Close'], length=14).iloc[i],
                'price': current['Close'],
                'position': success
            })

            active_trade = None

    return pd.DataFrame(trades), snapshots, active_trade

def find_trades_1(data, period, interval, trained_model=None, capital=100000):
    trades = []
    snapshots = []
    active_trades = []  # Allow multiple concurrent trades

    for i in range(5, len(data) - 1):
        current = data.iloc[i]
        next_candle = data.iloc[i + 1]
        if not has_valid_indicators(current):
            continue

        atr = current['ATR']
        candle_size = abs(current['Close'] - current['Open'])
        if candle_size < 0.5 * atr:
            continue

        is_index = is_index_data(data)
        context = classify_trade_context(current, atr)

        if not is_index and not volume_confirmation(data, i):
            continue
        if not is_strong_trend(data, i):
            continue
        if not is_good_time(next_candle.name):
            continue
        if detect_rsi_divergence(data, i):
            continue

        # Check entry conditions
        enter_long = (current['Close'] > current['Pivot'] and
                      next_candle['High'] >= current['High'] and
                      current['Close'] > current['EMA_20'] and
                      is_uptrend(data, i))

        enter_short = (current['Close'] < current['Pivot'] and
                       next_candle['Low'] <= current['Low'] and
                       current['Close'] < current['EMA_20'] and
                       is_downtrend(data, i))

        if enter_long or enter_short:
            entry_price = current['High'] if enter_long else current['Low']
            swing = current['Swing_Low'] if enter_long else current['Swing_High']
            stop_loss = swing - 0.5 * atr if enter_long else swing + 0.5 * atr
            sl_range = abs(entry_price - stop_loss)
            trend_strength = max(1.5, current['ADX'] / 20)
            target = entry_price + sl_range * trend_strength if enter_long else entry_price - sl_range * trend_strength

            trade = {
                'type': 'buy' if enter_long else 'sell',
                'entry_date': next_candle.name,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'trailing_sl': stop_loss,
                'target': target,
                'max_price': entry_price,
                'min_price': entry_price
            }

            if trained_model:
                snapshot = {
                    'ema_diff': current['EMA_20'] - current['EMA_50'],
                    'atr': atr,
                    'rsi': ta.rsi(data['Close'], length=14).iloc[i],
                    'price': current['Close']
                }
                if not allow_entry_by_ml(snapshot, trained_model):
                    continue

            active_trades.append(trade)

        updated_active_trades = []
        for trade in active_trades:
            high = next_candle['High']
            low = next_candle['Low']
            exit_price = None
            pnl = 0

            # Update max/min for trailing logic
            if trade['type'] == 'buy':
                trade['max_price'] = max(trade['max_price'], high)
                # Trailing SL
                trail_sl = trade['max_price'] - 1.5 * atr
                trade['trailing_sl'] = max(trade['trailing_sl'], trail_sl)

                if low <= trade['trailing_sl']:
                    exit_price = trade['trailing_sl']
                    pnl = exit_price - trade['entry_price']
                elif high >= trade['target']:
                    exit_price = trade['target']
                    pnl = exit_price - trade['entry_price']

            elif trade['type'] == 'sell':
                trade['min_price'] = min(trade['min_price'], low)
                trail_sl = trade['min_price'] + 1.5 * atr
                trade['trailing_sl'] = min(trade['trailing_sl'], trail_sl)

                if high >= trade['trailing_sl']:
                    exit_price = trade['trailing_sl']
                    pnl = trade['entry_price'] - exit_price
                elif low <= trade['target']:
                    exit_price = trade['target']
                    pnl = trade['entry_price'] - exit_price

            if exit_price:
                trades.append({
                    'type': trade['type'],
                    'entry_date': trade['entry_date'],
                    'exit_date': next_candle.name,
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'stop_loss': trade['stop_loss'],
                    'trailing_sl': trade['trailing_sl'],
                    'target': trade['target']
                })
                success = 1 if pnl > 0 else 0
                snapshots.append({
                    'ema_diff': current['EMA_20'] - current['EMA_50'],
                    'atr': atr,
                    'rsi': ta.rsi(data['Close'], length=14).iloc[i],
                    'price': current['Close'],
                    'position': success
                })
            else:
                updated_active_trades.append(trade)

        active_trades = updated_active_trades

    return pd.DataFrame(trades), snapshots, active_trades

def find_trades_2(data, period, interval, trained_model=None, capital=100000):
    trades = []
    snapshots = []
    active_trades = []  # allow multiple active trades

    for i in range(5, len(data) - 1):
        current = data.iloc[i]
        next_candle = data.iloc[i + 1]

        if not has_valid_indicators(current):
            continue

        atr = current['ATR']
        candle_size = abs(current['Close'] - current['Open'])
        if candle_size < atr * 0.5:
            continue

        is_index = is_index_data(data)
        if not is_index and not volume_confirmation(data, i):
            continue
        if not is_strong_trend(data, i):
            continue
        if not is_good_time(next_candle.name):
            continue
        if detect_rsi_divergence(data, i):
            continue

        # Check if we can add a new trade
        context = classify_trade_context(current, atr)
        rsi = ta.rsi(data['Close'], length=14).iloc[i]

        signal_buy = (
            current['Close'] > current['Pivot'] and
            next_candle['High'] >= current['High'] and
            current['Close'] > current['EMA_20'] and
            is_uptrend(data, i)
        )
        signal_sell = (
            current['Close'] < current['Pivot'] and
            next_candle['Low'] <= current['Low'] and
            current['Close'] < current['EMA_20'] and
            is_downtrend(data, i)
        )

        def calc_adaptive_target(entry, sl, trend_strength):
            scale = 1 + (trend_strength - 20) / 40  # simple adaptive scale
            return sl * 2 * scale

        if signal_buy:
            entry_price = current['High']
            stop_loss = current['Swing_Low'] - 0.5 * atr
            sl = entry_price - stop_loss
            sl, _ = enforce_risk_caps(sl, sl * 2)
            stop_loss = entry_price - sl
            trend_strength = data['ADX'].iloc[i]
            target = entry_price + calc_adaptive_target(entry_price, sl, trend_strength)

            if trained_model:
                snapshot = {
                    'ema_diff': current['EMA_20'] - current['EMA_50'],
                    'atr': atr,
                    'rsi': rsi,
                    'price': current['Close']
                }
                if not allow_entry_by_ml(snapshot, trained_model):
                    continue

            active_trades.append({
                'type': 'buy',
                'entry_date': next_candle.name,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'trailing_sl': stop_loss,
                'target': target
            })

        elif signal_sell:
            entry_price = current['Low']
            stop_loss = current['Swing_High'] + 0.5 * atr
            sl = stop_loss - entry_price
            sl, _ = enforce_risk_caps(sl, sl * 2)
            stop_loss = entry_price + sl
            trend_strength = data['ADX'].iloc[i]
            target = entry_price - calc_adaptive_target(entry_price, sl, trend_strength)

            if trained_model:
                snapshot = {
                    'ema_diff': current['EMA_20'] - current['EMA_50'],
                    'atr': atr,
                    'rsi': rsi,
                    'price': current['Close']
                }
                if not allow_entry_by_ml(snapshot, trained_model):
                    continue

            active_trades.append({
                'type': 'sell',
                'entry_date': next_candle.name,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'trailing_sl':stop_loss,
                'target': target
            })

        # Update and exit trades
        updated_trades = []
        for trade in active_trades:
            high = next_candle['High']
            low = next_candle['Low']
            exit_date = next_candle.name
            exit_price = None
            pnl = None

            if trade['type'] == 'buy':
                # Trailing SL logic
                move = high - trade['entry_price']
                if move > 10:
                    new_sl = trade['entry_price'] + (move - 5)
                    trade['trailing_sl'] = max(trade['trailing_sl'], new_sl)

                if low <= trade['trailing_sl']:
                    exit_price = trade['trailing_sl']
                    pnl = exit_price - trade['entry_price']
                elif high >= trade['target']:
                    exit_price = trade['target']
                    pnl = exit_price - trade['entry_price']

            elif trade['type'] == 'sell':
                move = trade['entry_price'] - low
                if move > 10:
                    new_sl = trade['entry_price'] - (move - 5)
                    trade['trailing_sl'] = min(trade['trailing_sl'], new_sl)

                if high >= trade['trailing_sl']:
                    exit_price = trade['trailing_sl']
                    pnl = trade['entry_price'] - exit_price
                elif low <= trade['target']:
                    exit_price = trade['target']
                    pnl = trade['entry_price'] - exit_price

            if exit_price is not None:
                trades.append({
                    'type': trade['type'],
                    'entry_date': trade['entry_date'],
                    'exit_date': exit_date,
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'stop_loss': trade['stop_loss'],
                    'trailing_sl': trade['trailing_sl'],
                    'target': trade['target']
                })
                snapshots.append({
                    'ema_diff': current['EMA_20'] - current['EMA_50'],
                    'atr': atr,
                    'rsi': rsi,
                    'price': current['Close'],
                    'position': 1 if pnl > 0 else 0
                })
            else:
                updated_trades.append(trade)

        active_trades = updated_trades

    return pd.DataFrame(trades), snapshots, active_trades


