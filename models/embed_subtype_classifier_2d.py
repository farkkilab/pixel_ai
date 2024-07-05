import torch.nn as nn
import ipdb

class EmbedSubtypeClassifier2D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbedSubtypeClassifier2D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (input_dim[0] // 2) * (input_dim[1] // 2), 256),  # Adjust based on pooling
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # x should have shape (batch_size, 1024, H, W)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

