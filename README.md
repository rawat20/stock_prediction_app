stock_prediction_app/
│
├── app/
│   ├── app.py                      # Streamlit app
│   ├── model_utils.py              # Model loading and prediction utilities
│   ├── preprocessing.py            # Data fetching and preprocessing
│   ├── models/
│   │   ├── TSLA/
│   │   │   ├── best_StockLSTM.pth  # Pre-trained LSTM weights for TSLA
│   │   │   ├── best_StockGRU.pth   # Pre-trained GRU weights for TSLA
│   │   ├── AAPL/
│   │   │   ├── best_StockLSTM.pth
│   │   │   ├── best_StockGRU.pth
│   │   ├── AMZN/
│   │   │   ├── best_StockLSTM.pth
│   │   │   ├── best_StockGRU.pth
│   │   ├── GOOGL/
│   │   │   ├── best_StockLSTM.pth
│   │   │   ├── best_StockGRU.pth
│   └── data/
│       └── live from yfinance
│
├── notebooks/
│   └── Project_arawat3.ipynb                  # Model training notebook
│
├── requirements.txt
└── README.md

// all references are mentioned in final project report
