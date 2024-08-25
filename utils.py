import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots

from urllib.request import urlopen  # web request
import certifi  #
import json
from sqlalchemy import create_engine
import psycopg2
import os
import configparser
import ssl


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


# Function to create the table of stock profiles
def create_daily_stock_prices_table():
    """
    Initialize a database for the FMP data about company profiles

    Args:
    ------------------
    table_name| system variables are required in the call, but not passed directly to the function
    
    Returns:
    ------------------
    None| creates postgreSQL database, managed by postgreSQL

    Errors:
    ------------------
    Authorization Error: if system/environment variables are not already loaded in the script connection may not be made to db
    """
    with psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    ) as connection:
        with connection.cursor() as cursor:
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS daily_stock_prices (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                open FLOAT NOT NULL,
                close FLOAT NOT NULL,
                high FLOAT NOT NULL,
                low FLOAT NOT NULL,
                volume BIGINT
            );
            """
            cursor.execute(create_table_query)
            connection.commit()
            print("Table daily_stock_prices created successfully.")



# Function to insert data into PostgreSQL table
def insert_data_to_daily_stock_prices(connection, df):
    """
    Insert new rows of data into the FMP-sourced table, using API call

    Args:
    ----------------
    connection:psycopg2.extensions.connection | connection to the database via psycopg2
    stock_profile:dict | api json response from FMP

    Returns:
    ------------------
    None | inserts rows into the project dataframe

    Errors:
    ------------------
    Authorization Error: if system/environment variables are not already loaded in the script connection may not be made to db
    """
    with connection.cursor() as cursor:
        insert_query = """
        INSERT INTO daily_stock_prices (date, symbol, open, close, high, low, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        for index, row in df.iterrows():
            cursor.execute(insert_query, (
                row["Date"],
                row["Ticker"],
                row["Open"],
                row["Close"],
                row["High"],
                row["Low"],
                row["Volume"]
            ))
        connection.commit()



def get_jsonparsed_data(url):
    """
    The cafile=certifi.where() part ensures that urlopen uses the certificate bundle provided by certifi 
    to verify the SSL certificate of the URL you're trying to access. Then read the api response and return it in json-form

    Args:
    ------------
    url:str | url with api key to successfully download information

    Returns:
    ------------
    json.loads(data):json | json object with the results of the query

    Raises:
    ------------
    HTTPError | if any part of the query string is inaccurate, users must have their own API key "HTTP Error 401: Unauthorized"

    Example : 
    url = ("https://financialmodelingprep.com/api/v3/search?query=AA&apikey=YOUR_API_KEY")
    get_jsonparsed_data(url)
    """

    # Create a custom SSL context
    context = ssl.create_default_context(cafile=certifi.where())
    
    # Initialize response from FMP
    response = urlopen(url, context=context)  
    
    # Parse the response
    data = response.read().decode("utf-8")  # read the API response
    
    # Return as JSON object
    return json.loads(data)



# Function to create the table of stock profiles
def create_sp500profiles_table():
    """
    Initialize a database for the FMP data about company profiles

    Args:
    ------------------
    table_name| system variables are required in the call, but not passed directly to the function
    
    Returns:
    ------------------
    None| creates postgreSQL database, managed by postgreSQL

    Errors:
    ------------------
    Authorization Error: if system/environment variables are not already loaded in the script connection may not be made to db
    """
    with psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    ) as connection:
        with connection.cursor() as cursor:
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS sp500_profiles (
                asof_date DATE,
                symbol VARCHAR(10) PRIMARY KEY,
                company_name TEXT,
                exchange TEXT,
                sector TEXT,
                industry TEXT,
                mktCap BIGINT,
                description TEXT,
                ceo TEXT,
                ipo_date DATE,
                website TEXT
            );
            """
            cursor.execute(create_table_query)
            connection.commit()
            print("Table sp500_profiles created successfully.")



# Function to insert data into PostgreSQL table
def insert_data_to_postgres(connection, stock_profile):
    """
    Insert new rows of data into the FMP-sourced table, using API call

    Args:
    ----------------
    connection:psycopg2.extensions.connection | connection to the database via psycopg2
    stock_profile:dict | api json response from FMP

    Returns:
    ------------------
    None | inserts rows into the project dataframe

    Errors:
    ------------------
    Authorization Error: if system/environment variables are not already loaded in the script connection may not be made to db
    """
    with connection.cursor() as cursor:
        insert_query = """
        INSERT INTO sp500_profiles (symbol, asof_date, company_name, exchange, sector, industry, mktCap, description, ceo, ipo_date, website)
        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            stock_profile.get("symbol"),
            stock_profile.get("companyName"),
            stock_profile.get("exchange"),
            stock_profile.get("sector"),
            stock_profile.get("industry"),
            stock_profile.get("mktCap"),
            stock_profile.get("description"),
            stock_profile.get("ceo"),
            stock_profile.get("ipoDate"),
            stock_profile.get("website"),
        ))
        connection.commit()