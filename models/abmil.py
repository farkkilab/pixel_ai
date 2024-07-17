import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class InstanceEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(InstanceEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim=512):
        super(AttentionMechanism, self).__init__()
        self.attention_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        if mask:
            raw_attention_weights = self.attention_fc(x).squeeze(-1)
            raw_attention_weights = raw_attention_weights.masked_fill(mask == 0, -1e9)  # Apply mask
        else:
            raw_attention_weights = self.attention_fc(x)
        attention_weights = F.softmax(raw_attention_weights, dim=1)
        return attention_weights


class ABMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, multibatch=None):
        super(ABMIL, self).__init__()
        self.instance_encoder = InstanceEncoder(input_dim, hidden_dim)
        self.attention_mechanism = AttentionMechanism(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.multibatch = multibatch

    def forward(self, bags, mask=None):
        # Process each instance in each bag
        if self.multibatch:
            instance_encodings = [self.instance_encoder(instance) for instance in bags.view(-1, bags.size(-1))]
            instance_encodings = torch.stack(instance_encodings).view(bags.size(0), bags.size(1), -1)
        else:
            instance_encodings = [self.instance_encoder(instance) for instance in bags]
            instance_encodings = torch.stack(instance_encodings)
        # Apply attention mechanism
        attention_weights = self.attention_mechanism(instance_encodings)

        if self.multibatch:
            # Compute bag representation
            bag_representation = torch.sum(attention_weights.unsqueeze(-1) * instance_encodings, dim=1)
            # Classify the bag representation
            output = self.classifier(bag_representation).squeeze(-1)
        else:
            bag_representation = torch.sum(attention_weights * instance_encodings, dim=1)
            output = self.classifier(bag_representation)
        return self.sigmoid(output), attention_weights

    def calculate_objective(self, X, Y):
        Y_prob = X
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood