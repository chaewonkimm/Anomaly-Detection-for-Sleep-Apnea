import torch
import torch.nn as nn
import math

class LocalAttention(nn.Module):
    def __init__(self, hidden_size, window_size):
        super(LocalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale 
        
        mask = torch.zeros(seq_len, seq_len).to(x.device)
        mask.fill_(float('-inf'))
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0
        
        attention_scores = attention_scores + mask.unsqueeze(0)
        
        attention_weights = torch.softmax(attention_scores, dim=-1) 
        
        context = torch.matmul(attention_weights, values)
        context_vector = context.mean(dim=1)
        return context_vector

class CNN_RNN_Model(nn.Module):
    def __init__(self):
        super(CNN_RNN_Model, self).__init__()
        
        self.first_cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(17, 1), stride=1, padding=(8, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.1)
        )

        self.additional_cnn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(17, 1), stride=1, padding=(8, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout(0.1)
            ) for _ in range(7)
        ])
        
        self.flatten = nn.Flatten()

        self.gru_input_size = 64 * 11 * 1
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=4, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        
        self.local_attention = LocalAttention(hidden_size=8, window_size=2)
        
        self.fc = nn.Linear(8, 1)

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

        attn_out = self.local_attention(gru_out)

        output = self.fc(attn_out)
        return output
