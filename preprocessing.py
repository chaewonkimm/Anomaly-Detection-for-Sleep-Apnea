import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class HRVDataset(Dataset):
    def __init__(self, data, feature_columns, label_column):
        self.data = data
        self.feature_columns = feature_columns
        self.label_column = label_column

    def __len__(self):
        return len(self.data['nsrrid'].unique())

    def __getitem__(self, idx):
        nsrrid = self.data['nsrrid'].unique()[idx]
        nsrrid_data = self.data[self.data['nsrrid'] == nsrrid]
        features = torch.tensor(nsrrid_data[self.feature_columns].values, dtype=torch.float32)
        labels = torch.tensor(nsrrid_data[self.label_column].values, dtype=torch.float32)
        return features, labels 

## nsrrid sequence padding
def collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)
    return features_padded, labels_padded

def load_data(file_path):
    data_frame = pd.read_csv(file_path)
    # data_frame['apnea'] = data_frame['event01start'].apply(lambda x: 1.0 if pd.notnull(x) else 0.0)
    feature_columns = [col for col in data_frame.columns if col not in ['hasrespevent', 'nsrrid', 'Start_sec', 'End_sec']]
    label_column = 'hasrespevent'
    return data_frame, feature_columns, label_column

def preprocess_data(file_path):
    data_frame, feature_columns, label_column = load_data(file_path)
    train = data_frame[data_frame['nsrrid'] <= 300862].copy()
    test = data_frame[data_frame['nsrrid'] > 300862].copy()
    train_dataset = HRVDataset(train, feature_columns, label_column)
    test_dataset = HRVDataset(test, feature_columns, label_column)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=None, collate_fn=collate_fn)
    return train_loader, test_loader