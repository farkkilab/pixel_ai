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
class Attention(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=500, encoding='trainable', input_size=224):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding = encoding
        if self.encoding != 'trainable':
            self.instance_encoder = InstanceEncoder(self.input_dim, self.hidden_dim)
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        if self.input_dim == 224:
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(50 * 53 * 53, self.M),
                nn.ReLU(),
            )
        elif self.input_dim == 56:
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(50 * 11 * 11, self.M),
                nn.ReLU(),
            )
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        if self.encoding == 'trainable':
            H = self.feature_extractor_part1(x)
            if self.input_dim == 224:
                H = H.view(-1, 50 * 53 * 53)
            elif self.input_dim == 56:
                H = H.view(-1, 50 * 11 * 11)
            H = self.feature_extractor_part2(H)  # KxM
        else:
            H = torch.stack([self.instance_encoder(instance) for instance in x])

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y, eval_mode=False):
        if eval_mode and hasattr(self, 'instance_encoder'):
            self.instance_encoder.eval()
        elif hasattr(self, 'instance_encoder'):
            self.instance_encoder.train()
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y,eval_mode=False):
        if eval_mode and hasattr(self, 'instance_encoder'):
            self.instance_encoder.eval()
        elif hasattr(self, 'instance_encoder'):
            self.instance_encoder.train()
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=500, encoding='trainable'):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding = encoding
        if self.input_dim:
            self.instance_encoder = InstanceEncoder(self.input_dim, self.hidden_dim)
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        #H = self.feature_extractor_part1(x)
        #H = H.view(-1, 50 * 4 * 4)
        #H = self.feature_extractor_part2(H)  # KxM
        if self.encoding == 'trainable':
            H = self.feature_extractor_part1(x)
            H = H.view(-1, 50 * 53 * 53)
            H = self.feature_extractor_part2(H)  # KxM
        else:
            H = torch.stack([self.instance_encoder(instance) for instance in x])
        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y, eval_mode=False):
        if eval_mode and hasattr(self, 'instance_encoder'):
            self.instance_encoder.eval()
        elif hasattr(self, 'instance_encoder'):
            self.instance_encoder.train()
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y, eval_mode=False):
        if eval_mode and hasattr(self, 'instance_encoder'):
            self.instance_encoder.eval()
        elif hasattr(self, 'instance_encoder'):
            self.instance_encoder.train()
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A