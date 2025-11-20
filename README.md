# Stock Prediction App — Deep Learning Forecasts (LSTM & GRU)

**Deep learning–based stock price prediction using LSTM and GRU (PyTorch) with feature engineering and a Streamlit app for live forecasting.**

---

## 🔎 Project overview
This project implements and compares recurrent neural networks (LSTM and GRU) to forecast short-term stock price trends using historical daily data from Yahoo Finance. It includes:
- Feature engineering: SMA_10, SMA_50, RSI.  
- Sequence windowing (20 time steps).  
- Models trained with PyTorch using Huber Loss.  
- Streamlit web app for live prediction and visualization.

See the full project report and presentation (PDFs included in this repository):  
- **Project Report** — `project_report_arawat3.pdf`. :contentReference[oaicite:5]{index=5}  

---

## 🔧 Tech stack
- Python, PyTorch, NumPy, pandas, scikit-learn. :contentReference[oaicite:7]{index=7}  
- Streamlit for the interactive demo. :contentReference[oaicite:8]{index=8}

---

## 🚀 Quickstart (local)
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/stock_prediction_app.git
cd stock_prediction_app
