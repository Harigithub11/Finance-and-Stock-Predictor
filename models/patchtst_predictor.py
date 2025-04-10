import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PatchTST(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PatchTST, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def run_finance_prediction(csv_path):
    df = pd.read_csv(csv_path)

    # Rename 'Price' to 'Close' if necessary
    if 'Price' in df.columns and 'Close' not in df.columns:
        df.rename(columns={'Price': 'Close'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']].dropna()

    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    sequence_length = 10
    data = df['Close'].values
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    sequences = torch.tensor(sequences).float().unsqueeze(-1)
    targets = torch.tensor(targets).float().unsqueeze(-1)

    dataset = TensorDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PatchTST(input_dim=1, hidden_dim=50, output_dim=1, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for batch in dataloader:
            sequences_batch, targets_batch = batch
            outputs = model(sequences_batch)
            loss = criterion(outputs, targets_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(sequences)
        for i in range(len(predictions)):
            print(f"Date: {df.index[i + sequence_length].date()}, Predicted Close Price: {predictions[i].item():.2f}")
