import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Feature2DEncoder(nn.Module):
    def __init__(self, num_channels, num_cnn_out=7, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=25, kernel_size=kernel_size, padding=padding), # num_cnn_out dependent
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=25, out_channels=50, kernel_size=kernel_size, padding=padding), # num_cnn_out dependent
            nn.BatchNorm1d(num_features=50),
            nn.SELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=kernel_size, padding=padding), # num_cnn_out dependent
            nn.BatchNorm1d(num_features=50),
            nn.Dropout(p=0.4),
            nn.SELU()
        )
        self.pool = nn.AvgPool1d(kernel_size=5)
        self.linear = nn.Sequential(
            nn.Linear(in_features=50, out_features=num_cnn_out),
            nn.SELU()
        )

    def __call__(self, inputs):
        if inputs.dim() == 4:
            N, T, A, F = inputs.size()
            inputs = inputs.reshape(N * T, A, F)
            outputs = self.forward(inputs)
            return outputs.reshape(N, T, -1)
        else:
            outputs = self.forward(inputs)
            return outputs
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.reshape(x.size()[:2])
        outputs = self.linear(x)
        return outputs

class SequentialFeatureDecoder(nn.Module):
    
    def __init__(self, num_cnn_out, num_lstm_out, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_cnn_out+num_lstm_out, hidden_size=hidden_size, batch_first=True)
        self.affine = nn.Linear(in_features=hidden_size, out_features=num_lstm_out)
        
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        x, (h0, c0) = self.lstm(inputs)
        return self.affine(x)

class FeatureSeqNet(nn.Module):

    def __init__(self, num_aws, num_features, num_cnn_out, time_size, hidden_size, num_lstm_out):
        super().__init__()
        self.spatial_encoder = Feature2DEncoder(num_aws, num_cnn_out)
        self.sequential_encoder = Feature2DEncoder(time_size, num_cnn_out)
        self.decoder = SequentialFeatureDecoder(num_cnn_out, num_lstm_out, hidden_size)

    def __call__(self, inputs, targets):
        return self.forward(inputs, targets)
    
    def forward(self, inputs, targets):
        spatial_x = self.spatial_encoder(inputs)
        N, T2, F = spatial_x.size()
        input_t0 = spatial_x[:, :T2 // 2, :]
        cnn_out = spatial_x[:, T2 // 2:, :]
        decoder_input_t0 = self.sequential_encoder(input_t0).reshape(N, 1, F)
        decoder_input = torch.concat([decoder_input_t0, targets[:, :-1, :]], dim=1)
        decoder_input = torch.concat([cnn_out, decoder_input], dim=2)
        out = self.decoder(decoder_input)
        return out