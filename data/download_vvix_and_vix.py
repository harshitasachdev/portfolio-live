import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set the date range
start_date = "2014-01-01"
end_date = "2025-06-06"

# Download VIX and VVIX
tickers = ['^VIX', '^VVIX']
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Rename columns for clarity
data.columns = ['VIX', 'VVIX']

# Drop missing values
data.dropna(inplace=True)

# Preview data
print(data.tail())

# Optional: Save to CSV
data.to_csv("vix_vvix_data.csv")

