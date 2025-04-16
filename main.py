# main.py

from models.xgboost_rnn_lstm_predictor import run_stock_prediction
from models.patchtst_predictor import run_finance_prediction

def main():
    print("\n📊 Welcome to the Finance Predictor!")
    print("Choose an option:")
    print("1. Stock Price Prediction using XGBoost")
    print("2. Financial Forecasting using Transformer (PatchTST)")
    
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        run_stock_prediction("data/S&P 500 Historical Data.csv")  
    elif choice == '2':
        run_finance_prediction("data/Amazon.com Stock Price History.csv")  
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()