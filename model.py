import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = src + self.dropout(self.linear(src))
        src = self.norm(src)
        return src

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_slices, num_channels, d_model, nhead, num_layers):
        super(TimeSeriesTransformer, self).__init__()
        self.num_slices = num_slices
        print(f"Initializing TimeSeriesTransformer with {self.num_slices} slices")
        self.slice_attention = nn.ModuleList([ConvBlock(num_channels, d_model, 3) for _ in range(num_slices)])
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        outs = []
        for i in range(self.num_slices):
            slice_x = x[:, i, :, :]
            slice_x = slice_x.transpose(1, 2)
            out = self.slice_attention[i](slice_x)
            outs.append(out)
        outs = torch.stack(outs, dim=1).mean(dim=1)
        for layer in self.encoder_layers:
            outs = layer(outs)
        outs = self.classifier(outs.mean(dim=1))
        return torch.sigmoid(outs)

model = TimeSeriesTransformer(num_slices=3, num_channels=10, d_model=512, nhead=8, num_layers=6)