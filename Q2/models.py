import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        half = x.size(1) // 2
        e_spk = F.normalize(x[:, :half], p=2, dim=1)
        e_env = F.normalize(x[:, half:], p=2, dim=1)
        return e_spk, e_env


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, e_spk, e_env):
        return self.fc(torch.cat([e_spk, e_env], dim=1))


class EnvironmentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DisentanglementModel(nn.Module):
    def __init__(self, input_dim, latent_dim, env_disc_hidden, num_speakers):
        super().__init__()
        half = latent_dim // 2
        self.half = half
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.speaker_disc = nn.Linear(half, num_speakers)
        self.env_disc = EnvironmentMLP(half, env_disc_hidden, env_disc_hidden)
        self.adv_disc = EnvironmentMLP(half, env_disc_hidden, env_disc_hidden)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, e_spk, e_env):
        return self.decoder(e_spk, e_env)

    def forward(self, x):
        e_spk, e_env = self.encoder(x)
        x_recon = self.decoder(e_spk, e_env)
        return e_spk, e_env, x_recon
