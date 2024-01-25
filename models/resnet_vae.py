import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import ipdb
# Define a custom VAE class with symmetric ResNet50 architecture
class ResNetVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, input_dimensions):
        super(ResNetVAE, self).__init__()
        self.input_dimensions = input_dimensions
        self.in_channels = in_channels
        # Encoder and Decoder share ResNet50 architecture
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Latent space layers
        self.fc_mu = nn.Linear(1000, latent_dim)
        self.fc_logvar = nn.Linear(1000, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, int((self.input_dimensions[0]/(2**5))*(self.input_dimensions[1]/(2**5))))
        self.decoder_resnet50 = models.resnet50(pretrained=False)
        self.decoder_resnet50.conv1 = nn.Conv2d(int((self.input_dimensions[0]/(2**5))*(self.input_dimensions[1]/(2**5))), 64, kernel_size=7, stride=2, padding=3, bias=False)

    def encode(self, x):
        #ipdb.set_trace()
        x = self.resnet50(x)
        x = torch.flatten(x, start_dim=1)
        #ipdb.set_trace()
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, int((self.input_dimensions[0]/(2**5))*(self.input_dimensions[1]/(2**5))), 1, 1)
        #ipdb.set_trace()
        x_recon = self.decoder_resnet50(z)
        return x_recon

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]



    # Loss function (VAE loss)
    def loss_function(self, x_recon, x, mu, log_var, kld_weight):
        ipdb.set_trace()
        reconstruction_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kld_weight *  kl_divergence

