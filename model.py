import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, reduction=16):
        super(CNN_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.1)
        )
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

class CNN_RNN_Model(nn.Module):
    def __init__(self):
        super(CNN_RNN_Model, self).__init__()
        
        self.first_cnn_block = CNN_Block(
            in_channels=2,
            out_channels=64,
            kernel_size=(17, 1),
            padding=(8, 0)
        )

        self.additional_cnn_blocks = nn.ModuleList([
            CNN_Block(
                in_channels=64,
                out_channels=64,
                kernel_size=(17, 1),
                padding=(8, 0)
            ) for _ in range(7)
        ])
        
        self.flatten = nn.Flatten()

        self.gru_input_size = 64 * 11 * 1
        self.gru_hidden_size = 4
        self.gru_num_layers = 2
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Linear(self.gru_hidden_size * 2, 1)

    def forward(self, x):
        batch_size, epochs, samples, channels = x.size()
        
        cnn_out = []
        for i in range(epochs):
            epoch_data = x[:, i, :, :].permute(0, 2, 1).unsqueeze(3)
            epoch_data = self.first_cnn_block(epoch_data)

            for cnn_block in self.additional_cnn_blocks:
                epoch_data = cnn_block(epoch_data)

            epoch_data = self.flatten(epoch_data)
            cnn_out.append(epoch_data)

        cnn_out = torch.stack(cnn_out, dim=1)
        
        gru_out, _ = self.gru(cnn_out)

        output = self.fc(gru_out[:, -1, :])
        return output