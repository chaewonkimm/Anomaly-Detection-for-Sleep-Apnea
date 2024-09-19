import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class ApneaDataset(Dataset):
    def __init__(self, csv_files, window_size=18000, step_size=15000, return_filename=False):
        self.data = []
        self.labels = []
        self.filenames = []
        self.return_filename = return_filename
        
        for file in csv_files:
            df = pd.read_csv(file)
            signals = df[['Airflow', 'SAO2']].values
            labels = df['apnea'].values

            for start in range(0, len(signals) - window_size + 1, step_size):
                segment = signals[start:start + window_size]
                apnea_events = labels[start:start + window_size].sum()
                
                segment = np.reshape(segment, (6, 3000, 2))
                self.data.append(segment)
                self.labels.append(apnea_events)
                self.filenames.append(os.path.basename(file))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        segment = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.return_filename:
            filename = self.filenames[idx]
            return segment, label, filename
        else:
            return segment, label

def load_apnea_data(csv_files, batch_size=32, shuffle=False, return_filename=False):
    dataset = ApneaDataset(csv_files, return_filename=return_filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

