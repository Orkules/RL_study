from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Neural network controller for the 20 cm TVC rocket-drone.
    Inputs (6):  IMU state [roll, pitch, yaw, gyro_x, gyro_y, gyro_z]
    Outputs (3): [servo_1 (pitch), servo_2 (roll), motor_speed] in [-1, 1]
    """

    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.65))

        # Orthogonal weight initialization for stable RL
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.actor_net(obs)
        # For 24D GPS/Pursuit models, enforce exact physical left-right bilateral symmetry:
        # pi(obs) = 0.5 * (f(obs) + mirror_act(f(mirror_obs(obs))))
        if obs.shape[-1] == 24:
            obs_m = obs.clone()
            obs_m[..., [1, 6, 8, 9, 13, 16, 19, 22]] = -obs_m[..., [1, 6, 8, 9, 13, 16, 19, 22]]
            raw_m = self.actor_net(obs_m)
            raw_sym = torch.stack([0.5 * (raw[..., 0] + raw_m[..., 0]), 0.5 * (raw[..., 1] - raw_m[..., 1])], dim=-1)
            mean = torch.tanh(raw_sym)
        else:
            mean = torch.tanh(raw)
        value = self.critic_net(obs).squeeze(-1)
        return mean, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(torch.clamp(self.log_std, -2.5, 0.5))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mean, _ = self.forward(obs_t)
        return np.clip(mean.squeeze(0).cpu().numpy(), -1.0, 1.0)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 8
    minibatch_size: int = 256


class PPOTrainer:
    """Proximal Policy Optimization trainer for continuous TVC drone control."""

    def __init__(self, model: ActorCritic, cfg: PPOConfig | None = None):
        self.model = model
        self.cfg = cfg or PPOConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)

    def update(
        self,
        obs_buf: np.ndarray,
        act_buf: np.ndarray,
        logp_buf: np.ndarray,
        ret_buf: np.ndarray,
        adv_buf: np.ndarray,
    ) -> Dict[str, float]:
        obs_t = torch.as_tensor(obs_buf, dtype=torch.float32)
        act_t = torch.as_tensor(act_buf, dtype=torch.float32)
        old_logp_t = torch.as_tensor(logp_buf, dtype=torch.float32)
        ret_t = torch.as_tensor(ret_buf, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_buf, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n_samples = obs_t.shape[0]
        indices = np.arange(n_samples)
        policy_losses: List[float] = []
        value_losses: List[float] = []

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.cfg.minibatch_size):
                mb_idx = indices[start : start + self.cfg.minibatch_size]
                _, new_logp, entropy, new_val = self.model.get_action_and_value(
                    obs_t[mb_idx], act_t[mb_idx]
                )
                logratio = new_logp - old_logp_t[mb_idx]
                ratio = logratio.exp()

                mb_adv = adv_t[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((new_val - ret_t[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(pg_loss.item()))
                value_losses.append(float(v_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
        }
