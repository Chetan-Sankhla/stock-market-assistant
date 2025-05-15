def show_candlestick_chart(data, patterns=None, candlestick_patterns=True):
    st.markdown("## 📈 Candlestick Chart with Pattern Filter and Zoom")

    # Extract detected patterns
    if "patterns" not in data.columns:
        st.warning("No candlestick patterns detected.")
        return

    # Extract all unique pattern names from data['patterns']
    all_patterns = sorted({
        pattern
        for pattern_list in data["patterns"]
        if isinstance(pattern_list, list)
        for pattern in pattern_list
    })

    # Multi-select filter
    selected_patterns = st.multiselect("📌 Filter by Candlestick Pattern(s):", all_patterns)

    # Filter rows based on selected patterns
    if selected_patterns:
        filtered_rows = []
        for idx, p in data["patterns"].items():
            if any(sp in p for sp in selected_patterns):
                tag = []
                for pat in p:
                    if pat in bullish_patterns:
                        tag.append(f"{pat} 🟢")
                    elif pat in bearish_patterns:
                        tag.append(f"{pat} 🔴")
                    else:
                        tag.append(pat)
                filtered_rows.append({"Date": idx.strftime("%Y-%m-%d"), "Patterns": ", ".join(tag)})
    else:
        filtered_rows = []
        for idx, p in data["patterns"].items():
            if p:
                tag = []
                for pat in p:
                    if pat in bullish_patterns:
                        tag.append(f"{pat} 🟢")
                    elif pat in bearish_patterns:
                        tag.append(f"{pat} 🔴")
                    else:
                        tag.append(pat)
                filtered_rows.append({"Date": idx.strftime("%Y-%m-%d"), "Patterns": ", ".join(tag)})

    pattern_df = pd.DataFrame(filtered_rows)

    if pattern_df.empty:
        st.info("No patterns matched the selected filter.")
        return

    # Convert to datetime for syncing with chart
    pattern_df["Date"] = pd.to_datetime(pattern_df["Date"])

    # Let user select a date to zoom on
    selected_date = st.selectbox(
        "📅 Jump to date on chart:",
        pattern_df["Date"].dt.strftime("%Y-%m-%d").tolist()
    )
    selected_datetime = pd.to_datetime(selected_date)
    zoom_start = selected_datetime - pd.Timedelta(days=2)
    zoom_end = selected_datetime + pd.Timedelta(days=2)

    # ================== PLOTLY CHART ==================
    fig = make_full_chart(data, zoom_start, zoom_end)
    st.plotly_chart(fig, use_container_width=True)

    # ========== TABLE ========== #
    st.markdown("### 📋 Matched Candlestick Patterns")
    st.dataframe(pattern_df, height=400, use_container_width=True)

def make_full_chart(data, zoom_start, zoom_end):
    # Create 3-row subplot: Candles, Volume, RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume", "RSI")
    )

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Candlesticks"
    ), row=1, col=1)

    # EMA traces
    if "EMA_20" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_20"],
                                 mode='lines', name='EMA 20',
                                 line=dict(color='blue', width=1)), row=1, col=1)

    if "EMA_50" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA_50"],
                                 mode='lines', name='EMA 50',
                                 line=dict(color='red', width=1)), row=1, col=1)

    # Volume bars
    if "Volume" in data.columns:
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"],
                             name="Volume", marker_color="gray"), row=2, col=1)

    # RSI
    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["RSI"],
            mode='lines', name='RSI',
            line=dict(color='purple', width=1)), row=3, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=3, col=1)

    # Layout
    fig.update_layout(
        title="📊 Candlestick Chart with Volume and RSI",
        xaxis_rangeslider_visible=False,
        xaxis_range=[zoom_start, zoom_end],
        height=900,
        margin=dict(t=60, b=40),
        showlegend=False
    )

    return fig
    

def abc(data, patterns=None, candlestick_patterns=True):
    # Extract all unique patterns from the DataFrame
    all_patterns = sorted({
        pattern
        for pattern_list in data['patterns'] if pattern_list
        for pattern in pattern_list
    })

    selected_patterns = st.multiselect(
        "📌 Filter by Candlestick Pattern(s):", all_patterns)

    # Filter rows based on selected pattern(s)
    if selected_patterns:
        filtered_rows = [
            {"Date": idx.strftime("%Y-%m-%d"), "Patterns": ", ".join(p)}
            for idx, p in data["patterns"].items()
            if any(sp in p for sp in selected_patterns)
        ]
    else:
        filtered_rows = [
            {"Date": idx.strftime("%Y-%m-%d"), "Patterns": ", ".join(p)}
            for idx, p in data["patterns"].items()
            if p
        ]

    pattern_df = pd.DataFrame(filtered_rows)