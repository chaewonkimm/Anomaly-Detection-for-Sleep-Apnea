import torch
import torch.nn as nn

class CNN_RNN_Model(nn.Module):
    def __init__(self):
        super(CNN_RNN_Model, self).__init__()
        
        self.first_cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(17, 2), stride=1, padding=(8, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.1)
        )

        self.additional_cnn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(17, 2), stride=1, padding=(8, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout(0.1)
            ) for _ in range(7)
        ])
        
        self.flatten = nn.Flatten()

        self.gru_input_size = 64 * 11 * 2

        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=4, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        
        self.fc = nn.Linear(4 * 2, 1)

    def forward(self, x):
        batch_size, epochs, samples, channels = x.size()
        
        cnn_out = []
        for i in range(epochs):
            epoch_data = x[:, i, :, :].permute(0, 2, 1).unsqueeze(3)
            epoch_data = self.first_cnn_block(epoch_data)
            
            for idx, cnn_block in enumerate(self.additional_cnn_blocks):
                epoch_data = cnn_block(epoch_data)

            epoch_data = self.flatten(epoch_data)
            cnn_out.append(epoch_data)

        cnn_out = torch.stack(cnn_out, dim=1)
        
        gru_out, _ = self.gru(cnn_out)

        output = self.fc(gru_out[:, -1, :])
        return output
