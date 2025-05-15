from scipy.signal import find_peaks
from scipy.stats import linregress
import numpy as np

def detect_patterns(data):
    patterns = {}

    # Head & Shoulders
    detected, details = is_head_and_shoulders(data)
    if detected:
        patterns.setdefault("Head & Shoulders 😟", []).append({
            "date": details["components"]["head"]["date"],
            "price": details["components"]["head"]["price"],
            "components": details["components"]
        })

    # Inverse Head & Shoulders
    detected, details = is_inverse_head_and_shoulders(data)
    if detected:
        patterns.setdefault("Inverse Head & Shoulders 😀", []).append({
            "date": details["components"]["head"]["date"],
            "price": details["components"]["head"]["price"],
            "components": details["components"]
        })

    # Other patterns (converted below)
    for check_fn, name in [
        (is_ascending_triangle, "Ascending Triangle 📈"),
        (is_descending_triangle, "Descending Triangle 📉"),
        (is_double_top, "Double Top ⚠️"),
        (is_double_bottom, "Double Bottom 💪"),
        (is_cup_and_handle, "Cup & Handle ☕"),
        (lambda df: detect_flag_pattern(df, direction='bull'), "Bull Flag 🚩"),
        (lambda df: detect_flag_pattern(df, direction='bear'), "Bear Flag 🚩")
    ]:
        detected, details = check_fn(data)
        if detected:
            patterns.setdefault(name, []).append(details)

    return patterns

def is_ascending_triangle(df, window=20, tolerance=0.015):
    if len(df) < window:
        return False, {}

    highs = df['High'][-window:]
    lows = df['Low'][-window:]
    timestamps = df.index[-window:]

    max_high = highs.max()
    flat_highs = highs[(max_high - highs) / max_high < tolerance]
    higher_lows = all(x < y for x, y in zip(lows, lows[1:]))

    if len(flat_highs) >= window // 2 and higher_lows:
        return True, {
            "date": timestamps[-1],
            "price": df['Close'].iloc[-1],
            "components": {
                "flat_resistance": max_high,
                "recent_higher_lows": lows.tolist()
            }
        }

    return False, {}

def is_descending_triangle(df, window=20, tolerance=0.015):
    """
    Detects a Descending Triangle pattern in the given OHLC DataFrame.
    - Lower highs
    - Flat lows
    """
    if len(df) < window:
        return False, {}

    recent = df.iloc[-window:]
    highs = recent['High'].values
    lows = recent['Low'].values
    timestamps = df.index[-window:]

    # Condition 1: Lower highs
    lower_highs = all(earlier > later for earlier, later in zip(highs, highs[1:]))

    # Condition 2: Flat lows (within tolerance range of min low)
    min_low = min(lows)
    flat_lows = np.sum(np.abs(lows - min_low) / min_low < tolerance)

    if lower_highs and flat_lows >= window // 2:
        return True, {
            "date":timestamps[-1],
            "price":df['Close'].iloc[-1],
            "components": {
                "flat_support": min_low,
                "recent_lower_highs": highs.tolist()
            }
        }
    
    return False, {}

def is_head_and_shoulders(df, window=30, tolerance=0.03):
    """
    Detects a Head and Shoulders pattern in the recent price action.
    Returns a tuple: (bool, pattern_details_dict)
    """
    if len(df) < window:
        return False, {}

    recent = df.iloc[-window:]
    closes = recent['Close'].values
    timestamps = recent.index

    peaks, _ = find_peaks(closes, distance=3)
    troughs, _ = find_peaks(-closes, distance=3)

    if len(peaks) < 3 or len(troughs) < 2:
        return False, {}

    last_peaks = peaks[-3:]
    last_troughs = troughs[-2:]

    p1, p2, p3 = closes[last_peaks[0]], closes[last_peaks[1]], closes[last_peaks[2]]
    t1, t2 = closes[last_troughs[0]], closes[last_troughs[1]]

    head_higher = p2 > p1 and p2 > p3
    shoulders_similar = abs(p1 - p3) / max(p1, p3) < tolerance
    neckline_flat = abs(t1 - t2) / max(t1, t2) < tolerance
    neckline = (t1 + t2) / 2
    breakdown = closes[-1] < neckline

    if head_higher and shoulders_similar and neckline_flat and breakdown:
        return True, {
            "pattern": "Inverse Head and Shoulders",
            "components": {
                "left_shoulder": {
                    "price": p1,
                    "date": str(timestamps[last_peaks[0]])
                },
                "mid_point_1": {
                    "price": t1,
                    "date": str(timestamps[last_troughs[0]])
                },
                "head": {
                    "price": p2,
                    "date": str(timestamps[last_peaks[1]])
                },
                "mid_point_2": {
                    "price": t2,
                    "date": str(timestamps[last_troughs[1]])
                },
                "right_shoulder": {
                    "price": p3,
                    "date": str(timestamps[last_peaks[2]])
                }
            }
        }

    return False, {}


def is_inverse_head_and_shoulders(df, window=30, tolerance=0.03):
    """
    Detects an Inverse Head and Shoulders pattern in recent price action.
    - tolerance: allowed variation between shoulder lows
    """
    if len(df) < window:
        return False, {}

    recent = df.iloc[-window:]
    closes = recent['Close'].values
    timestamps = recent.index

    # Troughs become peaks of -Close
    troughs, _ = find_peaks(-closes, distance=3)
    peaks, _ = find_peaks(closes, distance=3)

    if len(troughs) < 3 or len(peaks) < 2:
        return False, {}

    # Take last 3 troughs and 2 peaks
    last_troughs = troughs[-3:]
    last_peaks = peaks[-2:]

    # Order may vary depending on signal spacing
    s1, h, s2 = closes[last_troughs[0]], closes[last_troughs[1]], closes[last_troughs[2]]
    p1, p2 = closes[last_peaks[0]], closes[last_peaks[1]]

    head_lower = h < s1 and h < s2
    shoulders_similar = abs(s1 - s2) / max(s1, s2) < tolerance
    neckline_flat = abs(p1 - p2) / max(p1, p2) < tolerance

    # Final price should break above neckline
    neckline = (p1 + p2) / 2
    breakout = closes[-1] > neckline

    if head_lower and shoulders_similar and neckline_flat and breakout:
        return True, {
            "pattern": "Head and Shoulders",
            "components": {
#                "left_shoulder": {"time": timestamps[last_troughs[0]], "price": s1},
#                "head": {"time": timestamps[last_troughs[1]], "price": h},
#                "right_shoulder": {"time": timestamps[last_troughs[2]], "price": s2},
#                "neckline": {
#                    "times": (timestamps[last_peaks[0]], timestamps[last_peaks[1]]),
#                    "prices": (p1, p2)
                "left_shoulder": {
                    "price": s1,
                    "date": str(timestamps[last_troughs[0]])
                },
                "mid_point_1": {
                    "price": p1,
                    "date": str(timestamps[last_peaks[0]])
                },
                "head": {
                    "price": h,
                    "date": str(timestamps[last_troughs[1]])
                },
                "mid_point_2": {
                    "price": p2,
                    "date": str(timestamps[last_peaks[1]])
                },
                "right_shoulder": {
                    "price": s2,
                    "date": str(timestamps[last_troughs[2]])
                }
            }
        }
    
    return False, {}

def is_double_top(df, window=30, tolerance=0.02):
    if len(df) < window:
        return False, {}

    recent = df.iloc[-window:]
    closes = recent['Close'].values
    timestamps = recent.index

    # Find peaks and troughs
    peaks, _ = find_peaks(closes, distance=3)
    troughs, _ = find_peaks(-closes, distance=3)

    if len(peaks) < 2 or len(troughs) < 1:
        return False, {}

    # Loop through all valid peak pairs
    for i in range(len(peaks) - 1):
        idx1, idx2 = peaks[i], peaks[i+1]
        p1, p2 = closes[idx1], closes[idx2]

        # Check price similarity
        if abs(p1 - p2) / max(p1, p2) < tolerance:
            # Find a trough between these two peaks
            middle_troughs = [t for t in troughs if idx1 < t < idx2]
            if not middle_troughs:
                continue

            trough_idx = middle_troughs[np.argmin(closes[middle_troughs])]
            t = closes[trough_idx]

            # Breakdown condition (recent close below trough)
            if closes[-1] < t:
                return True, {
                    "date": timestamps[idx2],
                    "price": p2,
                    "components": {
                        "first_top": {"price": p1, "time": timestamps[idx1]},
                        "mid_point": {"price": t, "time": timestamps[trough_idx]},
                        "second_top": {"price": p2, "time": timestamps[idx2]}
                    }
                }
    
    return False, {}

def is_double_bottom(df, window=30, tolerance=0.02):
    """
    Detects Double Bottom pattern: two similar lows with a peak (neckline) in between.
    """
    if len(df) < window:
        return False, {}

    recent = df.iloc[-window:]
    closes = recent['Close'].values
    timestamps = recent.index

    troughs, _ = find_peaks(-closes, distance=3)
    peaks, _ = find_peaks(closes, distance=3)

    if len(troughs) < 2 or len(peaks) < 1:
        return False, {}

    # Get last 2 troughs and the peak between them
    b1, b2 = closes[troughs[-2]], closes[troughs[-1]]
    neckline = closes[peaks[-1]]

    bottoms_similar = abs(b1 - b2) / max(b1, b2) < tolerance
    breakout = closes[-1] > neckline

    if bottoms_similar and breakout:
        return True, {
            "date": timestamps[peaks[-1]],
            "price": b2,
            "components": {
                "first_bottom": {"time": timestamps[troughs[-2]], "price": b1},
                "peaks": {"time": timestamps[peaks[-1]], "price": neckline},
                "second_bottom": {"time": timestamps[troughs[-1]], "price": b2},
            }
        }

    return False, {}

def is_cup_and_handle(df, window=40, tolerance=0.03):
    """
    Detects a basic Cup and Handle pattern.
    Cup = rounded bottom, Handle = small pullback
    """
    if len(df) < window:
        return False, {}

    closes = df['Close'].values[-window:]
    midpoint = window // 2

    cup = closes[:midpoint]
    handle = closes[midpoint:]

    # Cup: rounded bottom
    min_cup = cup.min()
    cup_valid = all(c > min_cup for c in cup[:len(cup)//2]) and all(c > min_cup for c in cup[len(cup)//2:])

    # Handle: short dip then rise
    dip = handle[:len(handle)//2].min()
    end = handle[-1]

    handle_shallow = (end - dip) / dip < 0.1
    breakout = end > cup[0]  # handle end > cup start = breakout

    return cup_valid and handle_shallow and breakout, {}

def detect_flag_pattern(df, window=20, direction='bull'):
    """
    Detects Bull or Bear Flag patterns with breakout.
    Returns (True/False, components) where components contains:
        - top_1, top_2, bottom_1, bottom_2 (each with price/date)
    """

    if len(df) < window:
        return False, {}

    recent = df[-window:].copy()
    recent["Index"] = range(len(recent))  # needed for linregress
    close_prices = recent['Close'].values

    pole_window = window // 2
    flag_window = window - pole_window

    pole = close_prices[:pole_window]
    flag = close_prices[pole_window:]

    pole_slope = linregress(range(pole_window), pole).slope
    if direction == 'bull' and pole_slope <= 0:
        return False, {}
    if direction == 'bear' and pole_slope >= 0:
        return False, {}

    flag_slope = linregress(range(flag_window), flag).slope
    if direction == 'bull' and flag_slope > 0:
        return False, {}
    if direction == 'bear' and flag_slope < 0:
        return False, {}

    breakout = (
        close_prices[-1] > pole[-1] if direction == 'bull'
        else close_prices[-1] < pole[-1]
    )

    if not breakout:
        return False, {}

    # --- Find top 2 highs and bottom 2 lows with date ---
    highs = recent.nlargest(2, 'High')[['High']].copy()
    lows = recent.nsmallest(2, 'Low')[['Low']].copy()

    tops = {}
    for i, (idx, row) in enumerate(highs.iterrows(), 1):
        tops[f"top_points_{i}"] = {
            "price": round(row["High"], 2),
            "time": row.name  # <- use row.name instead of df.index[idx]
        }

    bottoms = {}
    for i, (idx, row) in enumerate(lows.iterrows(), 1):
        bottoms[f"bottom_points_{i}"] = {
            "price": round(row["Low"], 2),
            "time": row.name
        }

    return True, {
        "pattern": "Flag",
        "price": close_prices[-1],
        "components": format_components(tops, bottoms)
    }

def format_components(tops, bottoms):
    components = {}
    for k, v in tops.items():
        components[k] = {"price": v["price"], "time": v["time"]}
    for k, v in bottoms.items():
        components[k] = {"price": v["price"], "time": v["time"]}
    return components
