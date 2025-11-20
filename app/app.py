import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from preproccesing import fetch_stock_data, prepare_prediction_data
from model_utils import load_model, predict, evaluate_predictions

# Streamlit app configuration
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("Stock Trend Prediction App")
st.markdown("""
This app uses pre-trained LSTM and GRU models to predict stock trend for selected tickers.
Choose a stock and model to see the predicted price trend and performance metrics. \n
**Note**: Predictions are for analysis only and not financial advice.
""")

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")
ticker = st.sidebar.selectbox("Select Stock Ticker", ["TSLA", "AAPL", "AMZN", "GOOGL"])
model_type = st.sidebar.selectbox("Select Model", ["LSTM", "GRU"])
predict_button = st.sidebar.button("Predict")

# Main content
if predict_button:
    with st.spinner("Fetching data and generating predictions..."):
        try:
            # Fetch data
            start_date = '2020-01-01'
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            end_date = yesterday.strftime('%Y-%m-%d')
            df = fetch_stock_data(ticker, start_date, end_date)

            # Prepare data
            seq_length = 20
            X_scaled, y_scaled, y_true, feature_scaler, target_scaler = prepare_prediction_data(df, seq_length)

            # Load model and predict
            model = load_model(model_type, ticker)
            preds = predict(model, X_scaled, target_scaler)

            # Evaluate predictions
            mae, r2, rmse = evaluate_predictions(preds, y_true)

            # Plot results
            st.subheader(f"{model_type} Predictions for {ticker}")
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = df.index[seq_length:][:len(preds)]  
            ax.plot(dates, y_true, label="Actual Prices", color="blue", linewidth=2)
            ax.plot(dates, preds, label=f"{model_type} Predictions", color="red", linestyle="--", linewidth=2)
            ax.set_title(f"{model_type} Predicted vs Actual Prices for {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
            col2.metric("R² Score", f"{r2:.4f}")
            col3.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")

            # Trend analysis
            st.subheader("Trend Analysis")
            recent_preds = preds[-10:]  # Last 10 predictions
            trend = "upward" if recent_preds[-1] > recent_preds[0] else "downward"
            st.write(f"The {model_type} model predicts a **{trend} trend** for {ticker} based on the last 10 predictions.")
            st.write("**Note**: Stock price predictions are uncertain and should not be the sole basis for investment decisions.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("**Debug Info**:")
            st.write("- Check if the ticker is valid and has sufficient data.")
            st.write("- Ensure an internet connection for yfinance.")
            st.write("- Verify that the model weights exist in `models/<TICKER>/.`")
            st.stop()

else:
    st.info("Select a stock ticker and model, then click 'Predict' to see results.")