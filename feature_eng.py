import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 모델
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
        self.lstm = nn.LSTM(input_size=num_lstm_out + num_cnn_out, hidden_size=hidden_size, batch_first=True)
        self.affine = nn.Linear(in_features=num_cnn_out + hidden_size, out_features=num_lstm_out)

    def __call__(self, inputs, cnn_out):
        return self.forward(inputs, cnn_out)

    def forward(self, inputs, cnn_out):
        N, T, F = inputs.shape
        peek = cnn_out.repeat(T, 1).reshape(N, T, F)
        x = torch.concat([peek, inputs], dim=2)
        x, _ = self.lstm(x)
        affine_inputs = torch.concat([peek, x], dim=2)
        return self.affine(affine_inputs)

class FeatureSeqNet(nn.Module):

    def __init__(self, num_aws, num_features, time_size, hidden_size, num_lstm_out):
        super().__init__()
        self.spatial_encoder = Feature2DEncoder(num_aws, num_features)
        self.sequential_encoder = Feature2DEncoder(time_size, num_features - 2)
        self.decoder = SequentialFeatureDecoder(num_features - 2, num_lstm_out, hidden_size)

    def __call__(self, inputs, targets):
        return self.forward(inputs, targets)

    def forward(self, inputs, targets):
        spatial_x = self.spatial_encoder(inputs)
        N, T, F = spatial_x.size()
        pred_t0 = self.sequential_encoder(spatial_x)
        decoder_input = torch.concat([pred_t0.reshape(N, 1, F - 2), targets[:, :-1, :]], dim=1)
        out = self.decoder(decoder_input, pred_t0)
        return out