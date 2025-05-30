import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import os

st.title("📈 Stock Price Predictor using LSTM")
st.markdown("Train a deep learning model to predict closing prices of a selected stock.")

# Sidebar for inputs
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, GOOG)", value='AAPL')
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Load and validate data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

df = load_data(stock_symbol, start_date, end_date)

if df.empty:
    st.error(f"No data found for {stock_symbol} between {start_date} and {end_date}")
    st.stop()

st.subheader(f"{stock_symbol} Closing Price")
st.line_chart(df["Close"])

# Preprocess
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

training_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:training_size]

x_train, y_train = [], []
for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Load or train model
MODEL_PATH = "stock_predictor_model.h5"

st.subheader("Model Status")
if os.path.exists(MODEL_PATH):
    with st.spinner("Loading pre-trained model..."):
        model = load_model(MODEL_PATH)
    st.success("Model loaded from file!")
else:
    with st.spinner("Training LSTM model..."):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(100, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        model.save(MODEL_PATH)
    st.success("Model trained and saved!")

# Predict on test data
test_data = scaled_data[training_size - 100:]
x_test, y_test = [], data[training_size:]

for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i])

x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Visualize predictions
st.subheader("📈 Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test, label='Actual Price')
ax.plot(predictions, label='Predicted Price')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Predict the Next Day Closing Price
last_100_days = scaled_data[-100:]
last_100_days = last_100_days.reshape((1, 100, 1))
next_day_pred = model.predict(last_100_days)
next_day_price = scaler.inverse_transform(next_day_pred)

st.subheader("📌 Next Day Predicted Closing Price")
st.metric(label=f"{stock_symbol} Next Day Prediction", value=f"${next_day_price[0][0]:.2f}")
