import torch.nn as nn
import ipdb

class EmbedSubtypeClassifier2D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbedSubtypeClassifier2D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        conv_output_size = (input_dim[0] // 2) * (input_dim[1] // 2) * 256
        #self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),  # Adjust based on pooling
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
        #batch_size, channels, height, width = x.size()

        #x = x.view(batch_size, channels, -1).permute(2, 0, 1)
        #x, _ = self.attention(x, x, x)
        #x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)  # Reshape back for fully connected layers
        x = x.reshape(x.size(0), -1)
        return self.fc_layers(x)

