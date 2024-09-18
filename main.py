import os
import random
from segmentation import load_apnea_data
from train import train_and_test_model, split_csv_files, create_dataloaders

if __name__ == "__main__":
    data_dir = 'chat-csv'

    train_files, val_files, test_files = split_csv_files(data_dir)

    train_loader, val_loader, test_loader = create_dataloaders(train_files, val_files, test_files, batch_size=32)

    ahi_value = train_and_test_model(train_loader, val_loader, test_loader)
