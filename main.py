from fastapi import FastAPI
from models.xgboost_rnn_lstm_predictor import run_stock_prediction
from models.patchtst_predictor import run_finance_prediction

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "Welcome to the Finance Predictor API!",
        "options": ["/predict/stock", "/predict/finance"]
    }

@app.get("/predict/stock")
def predict_stock():
    run_stock_prediction("data/S&P 500 Historical Data.csv")
    return {"status": "✅ XGBoost + LSTM prediction done!"}

@app.get("/predict/finance")
def predict_finance():
    run_finance_prediction("data/Amazon.com Stock Price History.csv")
    return {"status": "✅ Transformer (PatchTST) prediction done!"}
