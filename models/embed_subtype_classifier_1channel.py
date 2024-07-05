import torch.nn as nn

class EmbedSubtypeClassifier1Channel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbedSubtypeClassifier1Channel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.fc(x)
