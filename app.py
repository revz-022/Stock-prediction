import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from textblob import TextBlob
import requests

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to fetch news articles from News API
def fetch_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])
    else:
        st.error("Failed to fetch news articles.")
        return []

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to fetch stock tickers from Finnhub API
@st.cache_data
def fetch_tickers():
    api_key = "YOUR_FINNHUB_API_KEY"  # Replace with your Finnhub API key
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [ticker['symbol'] for ticker in data]
    else:
        return []

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}. Please check your inputs.")
        return None

# Function to train a simple PyTorch model
def train_model(X_train, y_train):
    class DTAML(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=1):
            super(DTAML, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    model = DTAML(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    st.write("Model training completed.")

# Function for Exploratory Data Analysis
def perform_eda(stock_data):
    st.subheader("Stock Data Overview")
    st.write(stock_data.describe())
    st.line_chart(stock_data['Close'])

    st.subheader("Time Series Decomposition")
    decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative', period=30)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    axes[0].plot(decomposition.trend, label='Trend')
    axes[1].plot(decomposition.seasonal, label='Seasonality')
    axes[2].plot(decomposition.resid, label='Residuals')
    axes[3].plot(decomposition.observed, label='Observed')
    plt.tight_layout()
    st.pyplot(fig)

# Function for Feature Importance
def feature_importance_plot(stock_data):
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Close']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, rf.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')
    st.pyplot()

# Function for Candlestick Chart
def candlestick_chart(stock_data):
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Define the News API key as a constant variable
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Replace with your News API key

# Streamlit app entry point
def main():
    st.title("Stock Analytics and Insight Hub")
    st.sidebar.title("Options")
    
    # Fetch real-time ticker symbols
    ticker_symbols = fetch_tickers()
    selected_tickers = st.sidebar.multiselect("Select Ticker Symbols", ticker_symbols, default=ticker_symbols[:3])
    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)")
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)")
    
    if selected_tickers and start_date and end_date:
        for symbol in selected_tickers:
            stock_data = fetch_stock_data(symbol, start_date, end_date)
            if stock_data is not None:
                st.subheader(f"Stock Data for {symbol}")
                perform_eda(stock_data)
                feature_importance_plot(stock_data)
                candlestick_chart(stock_data)

    query = st.sidebar.text_input("Search for news")
    num_articles = st.sidebar.slider("Number of articles", 1, 10, 5)
    if query:
        news_articles = fetch_news(NEWS_API_KEY, query)
        if news_articles:
            st.subheader("News Articles")
            sentiments = [analyze_sentiment(article["description"] or "") for article in news_articles[:num_articles]]
            for i, article in enumerate(news_articles[:num_articles]):
                st.write(f"**{i+1}. {article['title']}** - {article['description']}")
            
            st.subheader("Sentiment Analysis")
            plt.bar(range(len(sentiments)), sentiments)
            st.pyplot()

if __name__ == "__main__":
    main()
