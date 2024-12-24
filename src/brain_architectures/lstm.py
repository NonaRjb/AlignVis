import torch
from torch import Tensor
from einops.layers.torch import Rearrange
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import math
from functools import partial


class lstm(nn.Module):
    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128, n_classes=2, device='cuda'):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, n_classes if n_classes > 2 else 1)
        self.device = device
    def forward(self, x):
        # Forward LSTM and get final state
        h_0 = torch.randn(self.lstm_layers, x.shape[0], self.lstm_size).to(self.device)
        c_0 = torch.randn(self.lstm_layers, x.shape[0], self.lstm_size).to(self.device)
        x = x.to(self.device)
        x = x.squeeze(dim=1).permute((0, 2, 1))
        x, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Forward output
        x = nn.functional.relu(self.output(hn[-1, ...]))
        x = self.classifier((x))

        return x