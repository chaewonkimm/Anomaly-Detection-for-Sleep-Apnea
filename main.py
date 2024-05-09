import torch
import torch.nn as nn
import os
import torch.optim as optim
from model import TimeSeriesTransformer
from preprocessing import preprocess_data

base_dir = "./"
data_path = os.path.join(base_dir, "chat-baseline-hrv-5min-0.14.0.csv")

train_loader, test_loader = preprocess_data(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_slices = 3
num_channels = 10
d_model = 512
nhead = 8
num_layers = 6
model = TimeSeriesTransformer(num_slices, num_channels, d_model, nhead, num_layers).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (features, labels) in enumerate(train_loader):
            print(f"Features original shape: {features.shape}")

            if features.dim() == 3 and features.size(1) != num_channels:
                features = features.transpose(1, 2)
                print(f"Features transposed to: {features.shape}")

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def evaluate_model():
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for features, labels in test_loader:
            batch_size = len(features)
            max_seq_len = max([f.shape[1] for f in features])
            padded_features = torch.zeros(batch_size, num_slices, num_channels, max_seq_len, device=device)
            for i, f in enumerate(features):
                seq_len = f.shape[1]
                padded_features[i, :, :, :seq_len] = f.to(device)
            targets = torch.cat([l.unsqueeze(0) for l in labels]).to(device)
            outputs = model(padded_features)
            predicted = (outputs.squeeze() > 0.5).float()
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    print(f'Accuracy: {total_correct / total_samples * 100:.2f}%')

if __name__ == "__main__":
    train_model(model, train_loader, optimizer, criterion, 100, device)
    evaluate_model()
