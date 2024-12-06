import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Sidebar for user inputs
st.sidebar.title("Backtest Moving Average Crossover Strategy")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
short_window = st.sidebar.slider("Short Moving Average Window", 1, 50, 20)
long_window = st.sidebar.slider("Long Moving Average Window", 51, 200, 100)

# Fetch stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate moving averages
data['Short_MA'] = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()
data['Long_MA'] = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()

# Generate buy/sell signals (1 for buy, -1 for sell)
data['Signal'] = 0
data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, -1)

# Calculate daily returns
data['Daily_Return'] = data['Adj Close'].pct_change()
data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']  # Strategy performance

# Cumulative returns
data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

# Performance metrics
total_return = data['Cumulative_Strategy_Return'][-1] - 1
annualized_return = (1 + total_return) ** (1 / ((data.index[-1] - data.index[0]).days / 365)) - 1
volatility = data['Strategy_Return'].std() * np.sqrt(252)  # Assuming 252 trading days per year
sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

# Max Drawdown Calculation
data['Rolling_Max'] = data['Cumulative_Strategy_Return'].cummax()
data['Drawdown'] = data['Cumulative_Strategy_Return'] / data['Rolling_Max'] - 1
max_drawdown = data['Drawdown'].min()

# Display performance metrics
st.subheader("Performance Metrics")
st.write(f"Total Return: {total_return * 100:.2f}%")
st.write(f"Annualized Return: {annualized_return * 100:.2f}%")
st.write(f"Volatility: {volatility * 100:.2f}%")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
st.write(f"Max Drawdown: {max_drawdown * 100:.2f}%")

# Plotting
st.subheader("Performance Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.index, data['Cumulative_Return'], label='Buy and Hold Strategy')
ax.plot(data.index, data['Cumulative_Strategy_Return'], label='Moving Average Crossover Strategy', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title(f"Cumulative Returns: {ticker} ({start_date} to {end_date})")
ax.legend()
st.pyplot(fig)

# Plot signals and moving averages
st.subheader("Trading Signals and Moving Averages")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.index, data['Adj Close'], label='Adj Close Price')
ax.plot(data.index, data['Short_MA'], label=f'Short {short_window}-Day MA', linestyle='--')
ax.plot(data.index, data['Long_MA'], label=f'Long {long_window}-Day MA', linestyle='--')

# Plot buy/sell signals
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]
ax.scatter(buy_signals.index, buy_signals['Adj Close'], marker='^', color='g', label='Buy Signal', alpha=1)
ax.scatter(sell_signals.index, sell_signals['Adj Close'], marker='v', color='r', label='Sell Signal', alpha=1)

ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.set_title(f"Moving Average Crossover Strategy: {ticker}")
ax.legend()
st.pyplot(fig)
