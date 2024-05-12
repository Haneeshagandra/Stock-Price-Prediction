# Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objs as go
import os


# Function to yesterday's stock price
def yesterday_stock_price(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    filename = "last_7_days_stock_prices.csv"
    data.to_csv(filename)
    # Read the CSV file containing the stock prices
    df = pd.read_csv("last_7_days_stock_prices.csv")

    # Get the last row's close price
    last_close_price = df.iloc[-1]['Close']
    print(last_close_price)
    return last_close_price


# Function to download stock price data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    # Save data to a CSV file
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    data.to_csv(filename)

    return data

# Function to create and train the LSTM model
def create_train_lstm_model(data, prediction_days):
    if len(data) < prediction_days:
        return None, None, None

    data = data['Close'].values  # Convert DataFrame to Numpy array
    data = data.reshape(-1, 1)  # Reshape as a 2D array
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train, y_train = [], []

    for i in range(prediction_days, len(data)):
        x_train.append(data[i - prediction_days:i, 0])
        y_train.append(data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1, batch_size=1)

    return model, scaler, prediction_days

# Function to predict stock prices
def predict_stock_price(model, data, scaler, prediction_days):
    last_x_days = data['Close'].values[-prediction_days:]
    last_x_days = last_x_days.reshape(-1, 1)
    last_x_days = scaler.transform(last_x_days)

    x_test = []
    x_test.append(last_x_days)
    x_test = np.array(x_test)
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price



# Streamlit frontend
def main():
    st.title("Stock Price Prediction App")
    st.write("Predict the future stock price using an LSTM model")

    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL for Apple)", "AAPL")
     
    
    yesterday_date = datetime.now() - timedelta(1)

    # User can specify the start date (default to 2 years back)
    start_date = st.date_input("Select Start Date", yesterday_date - timedelta(730))

    # Set the "End Date" to yesterday's date by default
    end_date = st.date_input("Select End Date", datetime.now() - timedelta(1))

    
    # Show yesterday's stock price dynamically
    yesterday_price = None

    Recent_start_date = datetime.now() - timedelta(days=7)

    try:
        # Retrieve yesterday's stock price if available
        Recent_price = yesterday_stock_price(ticker, Recent_start_date, yesterday_date)
        if Recent_price:
            st.write(f"Recent Stock Price: $ {Recent_price:.4f}")
        else:
            st.write("Recent stock price data is not available.")
    except Exception as e:
        st.write(f"An error occurred while retrieving yesterday's stock price: {e}")

    

    prediction_period = st.selectbox("Select Prediction Period", ["Days", "Months"])
    
    if prediction_period == "Days":
        prediction_range = list(range(1, 31))  # Display 1 to 30 days
        prediction_days = st.selectbox("Select Number of Prediction Days (Note: Count Start from the End Date)", prediction_range)
    else:
        selected_months = st.selectbox("Select Number of Prediction Months (Note: Count Start from the End Date)", [1, 2, 3, 4, 5, 6])
        
        # Calculate the end date for the selected number of months
        # end_date = end_date + pd.DateOffset(months=selected_months)

        # Calculate the prediction days based on selected months
        prediction_days = selected_months * 30
    

    if st.button("Predict"):
        data = download_stock_data(ticker, start_date, end_date)
        model, scaler, _ = create_train_lstm_model(data, prediction_days)
        if model is None:
            st.write("Not enough data to make predictions.")
        else:
            predicted_price = predict_stock_price(model, data, scaler, prediction_days)
             # Calculate the prediction end date
            prediction_end_date = end_date + timedelta(days=prediction_days)
        
            
        # Plotting stock performance
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
        fig.add_trace(go.Scatter(x=[prediction_end_date], y=predicted_price[0], mode='markers', name='Predicted Price'))

        fig.update_layout(
            title=f"Stock Performance for {ticker} from {start_date} to {end_date}",
            xaxis_title='Date',
            yaxis_title='Stock Price',
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
            )

        st.plotly_chart(fig)

        st.write(f"Predicted Price for the Date: {prediction_end_date.strftime('%Y-%m-%d')}")
        st.write(f"Predicted Price for the selected period: $ {predicted_price[0][0]:.4f}")


if __name__ == '__main__':
    main()
