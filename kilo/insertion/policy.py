"""Temporal feature extractor for the insertion policy's proprioceptive history."""

from __future__ import annotations

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ProprioHistoryExtractor(BaseFeaturesExtractor):
    """Encode a fixed-length sequence, then fuse it with the latest frame."""

    def __init__(self, observation_space: spaces.Box, history_frames: int = 30,
                 frame_size: int = 36, features_dim: int = 128):
        expected_size = history_frames * frame_size
        if observation_space.shape != (expected_size,):
            raise ValueError(
                f"Expected flattened history of {expected_size} values, got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim)
        self.history_frames = history_frames
        self.frame_size = frame_size
        self.temporal = nn.Sequential(
            nn.Conv1d(frame_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + frame_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequence = observations.reshape(-1, self.history_frames, self.frame_size)
        latent = self.temporal(sequence.transpose(1, 2))
        latest = sequence[:, -1, :]
        return self.fusion(torch.cat((latent, latest), dim=1))


class AsymmetricHistoryExtractor(BaseFeaturesExtractor):
    """Encode actor history; append privileged state for the value head only."""

    def __init__(self, observation_space, history_frames=30, frame_size=42, features_dim=128):
        if observation_space['history'].shape != (history_frames * frame_size,):
            raise ValueError('History dimensions must match history_frames * frame_size')
        privileged_size = observation_space['privileged'].shape[0]
        super().__init__(observation_space, features_dim + privileged_size)
        self.frames, self.frame_size = history_frames, frame_size
        self.temporal = nn.Sequential(
            nn.Conv1d(frame_size, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            encoded_size = self.temporal(torch.zeros(1, frame_size, history_frames)).shape[1]
        self.fusion = nn.Sequential(nn.Linear(encoded_size + frame_size, features_dim), nn.ReLU())

    def forward(self, observations):
        sequence = observations['history'].reshape(-1, self.frames, self.frame_size)
        latent = self.temporal(sequence.transpose(1, 2))
        actor = self.fusion(torch.cat((latent, sequence[:, -1]), dim=1))
        return torch.cat((actor, observations['privileged']), dim=1)


class AsymmetricHeads(nn.Module):
    def __init__(self, actor_size, privileged_size):
        super().__init__()
        self.actor_size = actor_size
        self.latent_dim_pi = self.latent_dim_vf = 128
        self.actor_net = nn.Sequential(nn.Linear(actor_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.critic_net = nn.Sequential(nn.Linear(actor_size + privileged_size, 128), nn.Tanh(),
                                        nn.Linear(128, 128), nn.Tanh())

    def forward_actor(self, features):
        return self.actor_net(features[:, :self.actor_size])

    def forward_critic(self, features):
        return self.critic_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricInsertionPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', AsymmetricHistoryExtractor)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        privileged_size = self.observation_space['privileged'].shape[0]
        self.mlp_extractor = AsymmetricHeads(self.features_dim - privileged_size, privileged_size)
