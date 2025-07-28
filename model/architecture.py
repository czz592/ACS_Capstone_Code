# architecture.py
# Version: 1.0

import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        """
        Define model architecture here.\n

        Version 1: 03/07
        - Basic Fully Connected Neural Network
        """
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self):
        """
        Define model architecture here.


        Version 2: 07/07
        - Refined boilerplate model
        """
        super(CNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # TODO: calculate correct layer sizes instead of using LazyLinear
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension
        x = self.feature_extractor(x)
        return self.classifier(x)
