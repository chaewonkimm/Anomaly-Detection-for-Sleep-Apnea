import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from segmentation import load_apnea_data
from model import CNN_RNN_Model
from sklearn.metrics import confusion_matrix, cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_csv_files(data_dir, train_ratio=0.6, val_ratio=0.2):
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    random.shuffle(csv_files)
    
    total_files = len(csv_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    
    train_files = csv_files[:train_size]
    val_files = csv_files[train_size:train_size + val_size]
    test_files = csv_files[train_size + val_size:]
    
    return train_files, val_files, test_files


def create_dataloaders(train_files, val_files, test_files, batch_size=32):
    train_loader = load_apnea_data(train_files, batch_size=batch_size, shuffle=True)
    val_loader = load_apnea_data(val_files, batch_size=batch_size, shuffle=False)
    test_loader = load_apnea_data(test_files, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


class ApneaTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader=None, num_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.HuberLoss()
        self.linear_reg = torch.nn.Linear(1, 1).to(device) 
        
        # 10 epoch 동안 validation loss가 개선되지 않으면 감소
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        # 30 epoch 동안 개선되지 않으면 학습 중단
        self.early_stopping_patience = 30
        self.early_stopping_counter = 0

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_loader)}')

            val_loss = self.validate()

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.early_stopping_counter = 0
                print(f"Validation loss improved at epoch {epoch+1}")
                self.save_model("best_model.pth")
            else:
                self.early_stopping_counter += 1
                print(f"No improvement in validation loss for {self.early_stopping_counter} epochs")
                
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / len(self.val_loader)}')
        return total_loss / len(self.val_loader)

    def test(self):
        self.model.eval()
        total_ahi = 0.0
        count = 0
        predicted_classes = []
        true_classes = []
        true_ahi_values = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                predicted_ahi = self.calculate_ahi(outputs)
                total_ahi += predicted_ahi
                count += 1

                true_ahi = labels.sum().item() / (len(labels) / 10.0 / 60.0) ##!
                true_ahi_values.append(true_ahi)

                true_class = self.classify_ahi(true_ahi)
                predicted_class = self.classify_ahi(predicted_ahi.item())
                
                true_classes.append(true_class)
                predicted_classes.append(predicted_class)

        cm = confusion_matrix(true_classes, predicted_classes, labels=["No OSA", "Mild OSA", "Moderate OSA", "Severe OSA"])
        print("Confusion Matrix:")
        print(cm)

        # Cohen's Kappa
        kappa = cohen_kappa_score(true_classes, predicted_classes, labels=["No OSA", "Mild OSA", "Moderate OSA", "Severe OSA"])
        print(f"Cohen's Kappa: {kappa}")

        # 4-class accuracy
        total_correct = sum(cm[i][i] for i in range(len(cm)))
        total_samples = cm.sum()
        acc4 = total_correct / total_samples
        print(f"4-Class Accuracy (Acc4): {acc4}")

        corrected_ahi = self.linear_reg(total_ahi / count)
        return corrected_ahi

    def classify_ahi(self, ahi_value):
        if ahi_value < 1:
            return "No OSA"
        elif 1 <= ahi_value < 5:
            return "Mild OSA"
        elif 5 <= ahi_value < 10:
            return "Moderate OSA"
        else:
            return "Severe OSA"
            

    def calculate_ahi(self, predicted_values, segment_duration=0.5):
        ahi = predicted_values / segment_duration
        return ahi.mean()

    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))


def train_and_test_model(train_loader, val_loader, test_loader):
    model = CNN_RNN_Model().to(device)
    trainer = ApneaTrainer(model, train_loader, val_loader, test_loader)
    
    trainer.train()
    
    ahi_value = trainer.test()
    
    return ahi_value