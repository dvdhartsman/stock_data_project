import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots


def plot_stock_ta(ticker:str, start_date:str="2024-01-01", end_date:str="2024-08-18", interval:str="1d") -> None:
    """Plot a given asset's price action between specified dates. Also includes: Bollinger Bands, 20/50/100 Day Moving Averages, 
    and subplot showing the RSI - a measure of whether a stock is over-bought or over-sold

    Args:
    -------------------
    ticker:str | uppercase string that matches the stock abbreviation for the asset of interest
    start_date:str | starting date of analysis
    end_date:str | ending date of analysis

    Returns:
    -------------------
    None: plotly.graph_objects.Figure | a plotly figure with 2 subplots, one showing price action and technical indicators, the other RSI

    Errors:
    -------------------
    Certain intervals are only available for recent periods of time per the yfinance API, keep shorter intervals < 60 days
    
    ex: plot_stock_ta("GOOGL", "2024-01-01", "2024-08-01", interval="60m")
    """
    

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Create different technical indicators
    data["50-Day MA"] = data["Close"].rolling(50).mean()
    data["100-Day MA"] = data["Close"].rolling(100).mean()
    
    bollingers = ta.bbands(data["Close"], length=20).iloc[:, 0:3]
    data["lower_bb"], data["simple_moving_average"], data["upper_bb"] = bollingers.iloc[:, 0], bollingers.iloc[:, 1], bollingers.iloc[:, 2]
    data["rsi"] = ta.rsi(data["Close"], length=14)
    
    
    # Plot code
    fig = make_subplots(rows=2, cols=1, row_heights=[.7, .3], subplot_titles=[f"{ticker} Price Action (Daily Freq)", f"{ticker} RSI"], 
                        vertical_spacing=.1, shared_xaxes=True)
    candles = go.Candlestick(x=data.index, open=data["Open"], close=data["Close"], high=data["High"], 
                                 low=data["Low"], name="Daily Candle")
    
    fig.add_trace(candles, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data["rsi"], name="Relative Strength Index"), row=2, col=1)
    
    colors = ["violet", "lightblue"]
    styles = ["solid", "dot"]
    widths = [.5, 1]
    
    # Bollinger Band Plots
    for idx, col in enumerate(["lower_bb", "simple_moving_average", "upper_bb"]):
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col.replace("_", " ")[:-3].title() + " BB",
                                line={"color":colors[idx%2], "width":widths[idx%2], "dash":styles[idx%2]}), row=1, col=1)

    ma_colors = ["royalblue", "yellow"]
    for idx, col in enumerate(["50-Day MA", "100-Day MA"]):
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col,
                                line={"color":ma_colors[idx%2], "width":.5, "dash":"dash"}), row=1, col=1)
    
    fig.update_layout(
        title = {"text":f"{ticker} Technical Analysis", "x":0.05, "font_size":30},
        yaxis_title = "Price",
        xaxis2 = {"title":"Date"},
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible = False
    )
    
    limits = [30, 70]
    
    for lim in limits:
        fig.add_trace(go.Scatter(x=data.index, y = np.array([lim for i in range(len(data.index))]), line={"color":"gray", "dash":"dash"},
                                 name=f"{lim} RSI", showlegend=False), row=2, col=1)
    
    fig.update_layout(yaxis2={"range":[0, 100]})
        
    
    return fig