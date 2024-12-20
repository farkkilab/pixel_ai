import torch.nn as nn
import ipdb
class EmbedSubtypeClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbedSubtypeClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (input_dim // 2), 256),  # Adjust based on pooling
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Shape: (batch size, 14, 768)
        # Pass through convolutional layers
        x = self.conv_layers(x)
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
        return self.fc_layers(x)
