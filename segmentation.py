import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ApneaDataset(Dataset):
    def __init__(self, csv_files, window_size=18000, step_size=15000):
        """
        window_size: 30분 segment 크기 (18000 samples)
        step_size: 중첩 간격 (25분 중첩, 15000 samples)
        """
        self.data = []
        self.labels = []
        
        for file in csv_files:
            df = pd.read_csv(file)
            signals = df[['Airflow', 'SAO2']].values
            labels = df['apnea'].values

            for start in range(0, len(signals) - window_size + 1, step_size):
                segment = signals[start:start + window_size]  # segment, total 18000 samples
                apnea_events = labels[start:start + window_size].sum()
                
                segment = np.reshape(segment, (6, 3000, 2))  # (6 epochs, 3000 samples, 2 signals)
                self.data.append(segment)
                self.labels.append(apnea_events)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        segment = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return segment, label

def load_apnea_data(csv_files, batch_size=32, shuffle=False):
    dataset = ApneaDataset(csv_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
