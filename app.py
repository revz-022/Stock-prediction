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

# Function to fetch news articles from the News API
def fetch_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data["articles"]
    else:
        st.error("Failed to fetch news articles.")
        return []

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to fetch real-time ticker symbols from Finnhub API
@st.cache_data
def fetch_tickers():
    api_key = "cov570hr01ql1b01ot70cov570hr01ql1b01ot7g"  # Replace with your Finnhub API key
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
        st.error(f"Failed to fetch data for {symbol} from Yahoo Finance. Please check your inputs and try again.")
        st.stop()

# Function for DTAML model training
def train_model(X_train, y_train):
    st.subheader("Training DTAML Model:")
    # Define DTAML model
    class DTAML(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(DTAML, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Prepare data for training
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 1

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    # Initialize model, criterion, optimizer
    model = DTAML(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    st.write("Model training completed.")
    
    #Display Background
def display_background():
    st.markdown(
        """
        <style>
        .image-container {
            width: 100%;
            height: 400px; /* Adjust height as needed */
            background-image: url('https://akm-img-a-in.tosshub.com/businesstoday/images/story/202301/stock-market-vs-real-estate_2_1-sixteen_nine.jpg?size=948:533');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }
        </style>
        <div class="image-container"></div>
        """,
        unsafe_allow_html=True
    )
    
    #Log Session
def log_session(symbol, start_date, end_date):
    session_data = pd.DataFrame({
        'Symbol': [symbol],
        'Start Date': [start_date],
        'End Date': [end_date]
    })
    if 'session_history' not in st.session_state:
        st.session_state.session_history = pd.DataFrame()
    st.session_state.session_history = pd.concat([st.session_state.session_history, session_data], ignore_index=True)

# Function for EDA and feature engineering
def perform_eda(stock_data):
    st.subheader("Exploratory Data Analysis:")
    st.write(stock_data.head())
    st.write(stock_data.describe())
    st.write("Data Types:")
    st.write(stock_data.dtypes)

    st.subheader("Visualization:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Stock Close Price Over Time')
    st.pyplot(fig)

    # Time Series Decomposition
    st.subheader("Time Series Decomposition:")
    decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative', period=30)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    ax1.plot(decomposition.trend)
    ax1.set_title('Trend')
    
    ax2.plot(decomposition.seasonal)
    ax2.set_title('Seasonality')
    
    ax3.plot(decomposition.resid)
    ax3.set_title('Residuals')
    
    ax4.plot(decomposition.observed)
    ax4.set_title('Observed')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Autocorrelation Plot
    st.subheader("Autocorrelation Plot:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(stock_data['Close'], ax=ax1, lags=30)
    ax1.set_title('Autocorrelation Plot')
    plot_pacf(stock_data['Close'], ax=ax2, lags=30)
    ax2.set_title('Partial Autocorrelation Plot')
    st.pyplot(fig)

# Function for PCA feature engineering
def perform_pca(stock_data):
    st.subheader("Feature Engineering using PCA:")
    X = stock_data[['Open', 'High', 'Low', 'Volume']] # Select relevant features
    y = stock_data['Close'] # Target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    st.write("Explained Variance Ratio:")
    st.write(pca.explained_variance_ratio_)

# Function for Rolling Statistics
def rolling_statistics(stock_data):
    st.subheader("Rolling Statistics:")
    rolling_mean = stock_data['Close'].rolling(window=30).mean()
    rolling_std = stock_data['Close'].rolling(window=30).std()
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
    plt.plot(stock_data.index, rolling_mean, label='Rolling Mean (30 days)')
    plt.plot(stock_data.index, rolling_std, label='Rolling Std (30 days)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Rolling Statistics')
    plt.legend()
    st.pyplot()

# Function for Volume Analysis
def volume_analysis(stock_data):
    st.subheader("Volume Analysis:")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(stock_data.index, stock_data['Close'], 'b-')
    ax2.plot(stock_data.index, stock_data['Volume'], 'r-')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='b')
    ax2.set_ylabel('Volume', color='r')
    ax1.set_title('Price and Volume Over Time')
    st.pyplot(fig)

# Function for Feature Importance Plot
def feature_importance_plot(stock_data):
    st.subheader("Feature Importance Plot:")
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Close']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Plot')
    st.pyplot()


# Function for Candlestick Chart
def candlestick_chart(stock_data):
    st.subheader("Candlestick Chart:")
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])
    fig.update_layout(title='Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)

# Function for Density Plot
def density_plot(stock_data):
    st.subheader("Density Plot:")
    stock_data.plot(kind='density', subplots=True, layout=(2, 3), sharex=False, figsize=(15, 8))
    st.pyplot()

# Define the News API key as a constant variable
NEWS_API_KEY = "775ef4e44e7940e282af7f4056713dda"

# Streamlit app
def main():
    state = st.session_state
    st.title("Stock Analytics and Insight Hub")
    display_background()
    st.sidebar.title("Options")
    
    # Fetch real-time ticker symbols
    ticker_symbols = fetch_tickers()
    
    # Retrieve the selected ticker symbols from the browser's local storage
    selected_tickers = st.sidebar.multiselect("Select Ticker Symbols", ticker_symbols, default=ticker_symbols[:3], key="ticker_multiselect")

    # Update the selected ticker symbols in the browser's local storage
    if selected_tickers:
        st.write("You selected:", selected_tickers)

    # Sidebar inputs for date range
    start_date = st.sidebar.text_input("Enter the start date (YYYY-MM-DD):")
    end_date = st.sidebar.text_input("Enter the end date (YYYY-MM-DD):")

    # Fetch stock data for selected ticker symbols
    if selected_tickers and start_date and end_date:
        stock_data = {}
        for symbol in selected_tickers:
            stock_data[symbol] = fetch_stock_data(symbol, start_date, end_date)
            
    # Fetch news articles using the predefined API key
    query = st.sidebar.text_input("Enter search query")
    num_articles = st.sidebar.slider("Number of articles", min_value=1, max_value=10, value=5)

    if query:
        news_articles = fetch_news(NEWS_API_KEY, query)
        if news_articles:
            st.subheader("News Articles")
            sentiments = []
            for i, article in enumerate(news_articles[:num_articles]):
                st.write(f"**{i + 1}. {article['title']}** - {article['description']}")
                sentiment_score = analyze_sentiment(article["description"])
                sentiments.append(sentiment_score)

            # Plot sentiment scores
            if sentiments:
                st.subheader("Sentiment Analysis Chart")
                plt.bar(range(len(sentiments)), sentiments)
                plt.xlabel("News Article")
                plt.ylabel("Sentiment Score")
                st.pyplot()

    # Display comparative visualizations
    for symbol, data in stock_data.items():
            st.subheader(f"Stock Data for {symbol}:")
            st.write(data.head())
            perform_eda(data)
            perform_pca(data)
            rolling_statistics(data)
            volume_analysis(data)
            feature_importance_plot(data)
            candlestick_chart(data)
            density_plot(data)

if __name__ == "__main__":
    main()
