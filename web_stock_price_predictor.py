import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error("No data found for this stock symbol. Please try a different one.")
        st.stop()
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("Stock Data")
st.write(data)

data['MA_for_250_days'] = data.Close.rolling(250).mean()
data['MA_for_200_days'] = data.Close.rolling(200).mean()
data['MA_for_100_days'] = data.Close.rolling(100).mean()

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label=values.name)
    plt.plot(full_data.Close, 'b', label='Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'g', label=extra_dataset.name)
    plt.legend()
    return fig

st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), data['MA_for_250_days'], data))

st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), data['MA_for_200_days'], data))

st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), data['MA_for_100_days'], data))

st.subheader('Original Close Price, MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), data['MA_for_100_days'], data, 1, data['MA_for_250_days']))

splitting_len = int(len(data) * 0.7)
train_data = data[['Close']].iloc[:splitting_len]
test_data = data[['Close']].iloc[splitting_len:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)

scaled_test = scaler.transform(test_data)
scaled_train = scaler.transform(train_data)

scaled_data = np.concatenate((scaled_train[-100:], scaled_test), axis=0)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

try:
    predictions = model.predict(x_data)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

num_predictions = len(inv_pre)
start_index = splitting_len + 100
end_index = start_index + num_predictions

if end_index > len(data):
    end_index = len(data)
    num_predictions = end_index - start_index
    inv_pre = inv_pre[:num_predictions]
    inv_y_test = inv_y_test[:num_predictions]

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1)[:num_predictions],
        'predictions': inv_pre.reshape(-1)[:num_predictions]
    },
    index=data.index[start_index:end_index]
)

st.subheader("Original vs Predicted")
st.write(ploting_data)

st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(data.index[:splitting_len], data.Close[:splitting_len], 'b', label="Training Data")
plt.plot(ploting_data.index, ploting_data['original_test_data'], 'g', label="Actual Price")
plt.plot(ploting_data.index, ploting_data['predictions'], 'r', label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)