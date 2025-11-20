import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_layers=1):
        super().__init__()
        # Processes sequences to capture temporal patterns
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True) 
        #Maps LSTM output to prediction (e.g. next price)
        self.fc = nn.Linear(hidden_dim, output_dim)

    # Defines how input flows through the model
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1])

class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1])

def load_model(model_type, ticker, model_dir='models'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 4  # Close, SMA_10, SMA_50, RSI
    if model_type.lower() == 'lstm':
        model = StockLSTM(input_dim).to(device)
        model_path = os.path.join(model_dir, ticker, 'best_StockLSTM.pth')
    elif model_type.lower() == 'gru':
        model = StockGRU(input_dim).to(device)
        model_path = os.path.join(model_dir, ticker, 'best_StockGRU.pth')
    else:
        raise ValueError("Model type must be 'lstm' or 'gru'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, X, target_scaler):
    model.eval()
    # Get device from model's parameters
    device = next(model.parameters()).device
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    preds_raw = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    return preds_raw

def evaluate_predictions(preds, true_values):
    mae = mean_absolute_error(true_values, preds)
    r2 = r2_score(true_values, preds)
    rmse = np.sqrt(mean_squared_error(true_values, preds))
    return mae, r2, rmse