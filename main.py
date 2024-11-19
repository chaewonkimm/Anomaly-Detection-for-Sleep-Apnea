import os
import random
import numpy as np
import torch
from segmentation import load_apnea_data
from train import train_and_test_model, split_csv_files, create_dataloaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(1004)

    data_dir = 'chat-csv'

    train_files, val_files, test_files = split_csv_files(data_dir)

    train_loader, val_loader, test_loader = create_dataloaders(train_files, val_files, test_files, batch_size=64)

    ahi_value = train_and_test_model(train_loader, val_loader, test_loader)

