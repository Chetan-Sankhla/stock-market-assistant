import pandas as pd
import talib

def detect_all_candlestick_patterns(df):
    patterns = {}
    for func in dir(talib):
        if func.startswith("CDL"):
            pattern_func = getattr(talib, func)
            pattern_name = func.replace("CDL", "")
            result = pattern_func(df['Open'], df['High'], df['Low'], df['Close'])
            patterns[pattern_name] = result
    return pd.DataFrame(patterns, index=df.index)

def detect_candle_patterns(df):
    pattern_df = detect_all_candlestick_patterns(df)

    # Keep only columns where any value is non-zero
    detected_patterns = pattern_df.loc[:, (pattern_df != 0).any()]

    # Ensure all values are numbers (not Series or DataFrames)
    if not detected_patterns.empty and len(detected_patterns) == len(df):
        df['patterns'] = detected_patterns.apply(
            lambda row: [col for col in detected_patterns.columns if pd.notna(row[col]) and row[col] != 0],
            axis=1
        )
    else:
        df['patterns'] = [[] for _ in range(len(df))]

    return df

