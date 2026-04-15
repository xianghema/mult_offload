import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HATD3D3QNConfig:
    """Version ha_td3_d3qn_v4.py: pairwise standard D3QN (mean-centered dueling)."""
    # ---------- multi-agent / environment ----------
    num_agents: int = 30
    num_servers: int = 6
    obs_dim: int = 7
    server_feat_dim: int = 4
    num_regions: int = 6

    # ---------- conditional actor ----------
    cond_mode: str = "embedding"  # "onehot" or "embedding"
    region_emb_dim: int = 8

    # ---------- action bounds ----------
    xi_low: float = 0.0
    xi_high: float = 1.0
    f_low: float = 0.001
    f_high: float = 1.0
    p_low: float = 0.001
    p_high: float = 2.0

    # ---------- network dims ----------
    actor_hidden1: int = 128
    actor_hidden2: int = 128

    critic_agent_hidden: int = 128
    critic_server_hidden: int = 128
    critic_pair_hidden1: int = 256
    critic_pair_hidden2: int = 128
    critic_home_hidden: int = 128

    # ---------- training ----------
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 100000

    # TD3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    explore_noise: float = 0.1
    policy_delay: int = 2
    grad_clip: float = 1.0

    # relay-target exploration (discrete epsilon-greedy)
    relay_epsilon_start: float = 0.20
    relay_epsilon_min: float = 0.03
    relay_epsilon_decay: float = 0.995
    relay_epsilon_decay_start_it: int = 50
    relay_epsilon_decay_per_episode: bool = True
    relay_epsilon_decay_start_episode: int = 60

    # continuous-action exploration schedule
    explore_noise_start: float = 0.10
    explore_noise_min: float = 0.02
    explore_noise_decay: float = 0.999
    explore_noise_decay_per_episode: bool = True
    explore_noise_decay_start_episode: int = 60

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_orthogonal_init: bool = False
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_layer(layer: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


class RegionConditionEncoder(nn.Module):
    def __init__(self, num_regions: int, mode: str = "onehot", emb_dim: int = 8):
        super().__init__()
        if mode not in ("onehot", "embedding"):
            raise ValueError(f"Unsupported cond mode: {mode}")
        self.num_regions = num_regions
        self.mode = mode
        self.emb_dim = emb_dim
        if mode == "embedding":
            self.embedding = nn.Embedding(num_regions, emb_dim)

    @property
    def cond_dim(self) -> int:
        return self.num_regions if self.mode == "onehot" else self.emb_dim

    def forward(self, region_id: torch.Tensor) -> torch.Tensor:
        if region_id.dtype != torch.long:
            region_id = region_id.long()
        if self.mode == "onehot":
            return F.one_hot(region_id, num_classes=self.num_regions).float()
        return self.embedding(region_id)


class SharedActor(nn.Module):
    """
    设备侧局部连续控制器。
    输入:
        obs:       [B, N, obs_dim] 或 [N, obs_dim]
        region_id: [B, N] 或 [N]
    输出:
        action[..., 0] = xi
        action[..., 1] = f_L(GHz)
        action[..., 2] = p_R(W)
    """

    def __init__(self, cfg: HATD3D3QNConfig):
        super().__init__()
        self.cfg = cfg
        self.cond_encoder = RegionConditionEncoder(
            num_regions=cfg.num_regions,
            mode=cfg.cond_mode,
            emb_dim=cfg.region_emb_dim,
        )
        input_dim = cfg.obs_dim + self.cond_encoder.cond_dim

        self.fc1 = nn.Linear(input_dim, cfg.actor_hidden1)
        self.fc2 = nn.Linear(cfg.actor_hidden1, cfg.actor_hidden2)
        self.fc_out = nn.Linear(cfg.actor_hidden2, 3)

        if cfg.use_orthogonal_init:
            init_layer(self.fc1)
            init_layer(self.fc2)
            init_layer(self.fc_out, gain=0.01)

    def _map_raw_to_action(self, raw: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        z = torch.sigmoid(raw)
        xi = cfg.xi_low + (cfg.xi_high - cfg.xi_low) * z[..., 0:1]
        f = cfg.f_low + (cfg.f_high - cfg.f_low) * z[..., 1:2]
        p = cfg.p_low + (cfg.p_high - cfg.p_low) * z[..., 2:3]
        return torch.cat([xi, f, p], dim=-1)

    def forward(self, obs: torch.Tensor, region_id: torch.Tensor) -> torch.Tensor:
        cond = self.cond_encoder(region_id)
        x = torch.cat([obs, cond], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.fc_out(x)
        return self._map_raw_to_action(raw)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        region_id: torch.Tensor,
        deterministic: bool = False,
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
    ) -> torch.Tensor:
        self.eval()
        action = self.forward(obs, region_id)
        if deterministic or noise_std <= 0.0:
            return action

        noise = torch.randn_like(action) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        action = action + noise
        action[..., 0] = action[..., 0].clamp(self.cfg.xi_low, self.cfg.xi_high)
        action[..., 1] = action[..., 1].clamp(self.cfg.f_low, self.cfg.f_high)
        action[..., 2] = action[..., 2].clamp(self.cfg.p_low, self.cfg.p_high)
        return action


class PairwiseRelayCritic(nn.Module):
    """
    方案一：标准 mean-centered dueling 的 pairwise relay critic。

    对每个设备 n 和候选二次转发目标 j，显式建模：
        Q(n, j) = V(n) + A(n, j) - mean_j A(n, j)

    其中：
    - V(n) 表示在当前设备局部状态、区域身份、连续控制已确定时的基础状态价值；
    - A(n, j) 表示选择候选 relay target j 的相对优势；
    - pairwise 特征显式包含：设备编码、候选服务器编码、home 服务器编码、hop、is_home、alive。
    """

    def __init__(self, cfg: HATD3D3QNConfig):
        super().__init__()
        self.cfg = cfg
        self.num_agents = cfg.num_agents
        self.num_servers = cfg.num_servers

        self.cond_encoder = RegionConditionEncoder(
            num_regions=cfg.num_regions,
            mode=cfg.cond_mode,
            emb_dim=cfg.region_emb_dim,
        )
        cond_dim = self.cond_encoder.cond_dim

        dev_input_dim = cfg.obs_dim + cond_dim + 3
        self.dev_fc1 = nn.Linear(dev_input_dim, cfg.critic_agent_hidden)
        self.dev_fc2 = nn.Linear(cfg.critic_agent_hidden, cfg.critic_agent_hidden)

        self.server_fc1 = nn.Linear(cfg.server_feat_dim, cfg.critic_server_hidden)
        self.server_fc2 = nn.Linear(cfg.critic_server_hidden, cfg.critic_server_hidden)

        self.value_fc1 = nn.Linear(cfg.critic_agent_hidden + cfg.critic_server_hidden + 1, cfg.critic_home_hidden)
        self.value_fc2 = nn.Linear(cfg.critic_home_hidden, cfg.critic_home_hidden)
        self.value_out = nn.Linear(cfg.critic_home_hidden, 1)

        pair_input_dim = (
            cfg.critic_agent_hidden
            + cfg.critic_server_hidden
            + cfg.critic_server_hidden
            + 1
            + 1
            + 1
        )
        self.adv_fc1 = nn.Linear(pair_input_dim, cfg.critic_pair_hidden1)
        self.adv_fc2 = nn.Linear(cfg.critic_pair_hidden1, cfg.critic_pair_hidden2)
        self.adv_out = nn.Linear(cfg.critic_pair_hidden2, 1)

        if cfg.use_orthogonal_init:
            for layer in [
                self.dev_fc1,
                self.dev_fc2,
                self.server_fc1,
                self.server_fc2,
                self.value_fc1,
                self.value_fc2,
                self.adv_fc1,
                self.adv_fc2,
            ]:
                init_layer(layer)
            init_layer(self.value_out, gain=0.01)
            init_layer(self.adv_out, gain=0.01)

    @staticmethod
    def _batch_gather_rows(matrix: torch.Tensor, row_index: torch.Tensor) -> torch.Tensor:
        B = matrix.shape[0]
        batch_idx = torch.arange(B, device=matrix.device).unsqueeze(1)
        return matrix[batch_idx, row_index]

    def forward(
        self,
        obs: torch.Tensor,
        region_id: torch.Tensor,
        cont_action: torch.Tensor,
        server_feat: torch.Tensor,
        hop_matrix: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = obs.shape
        M = server_feat.shape[1]
        assert N == self.num_agents, f"N mismatch: got {N}, expect {self.num_agents}"
        assert M == self.num_servers, f"M mismatch: got {M}, expect {self.num_servers}"

        cond = self.cond_encoder(region_id)
        dev_x = torch.cat([obs, cond, cont_action], dim=-1)
        dev_h = F.relu(self.dev_fc1(dev_x))
        dev_h = F.relu(self.dev_fc2(dev_h))

        server_h = F.relu(self.server_fc1(server_feat))
        server_h = F.relu(self.server_fc2(server_h))

        home_server_h = self._batch_gather_rows(server_h, region_id)
        hop_pair = self._batch_gather_rows(hop_matrix, region_id)
        is_home = F.one_hot(region_id, num_classes=M).float()

        if alive_mask is None:
            alive_mask = torch.ones(B, N, device=obs.device)
        alive_mask = alive_mask.float()
        alive_feat = alive_mask.unsqueeze(-1)

        value_in = torch.cat([dev_h, home_server_h, alive_feat], dim=-1)
        value_h = F.relu(self.value_fc1(value_in))
        value_h = F.relu(self.value_fc2(value_h))
        value = self.value_out(value_h)

        dev_h_exp = dev_h.unsqueeze(2).expand(-1, -1, M, -1)
        cand_server_h = server_h.unsqueeze(1).expand(-1, N, -1, -1)
        home_server_h_exp = home_server_h.unsqueeze(2).expand(-1, -1, M, -1)
        hop_feat = hop_pair.unsqueeze(-1)
        is_home_feat = is_home.unsqueeze(-1)
        alive_pair_feat = alive_feat.unsqueeze(2).expand(-1, -1, M, -1)

        pair_in = torch.cat(
            [
                dev_h_exp,
                cand_server_h,
                home_server_h_exp,
                hop_feat,
                is_home_feat,
                alive_pair_feat,
            ],
            dim=-1,
        )
        adv_h = F.relu(self.adv_fc1(pair_in))
        adv_h = F.relu(self.adv_fc2(adv_h))
        adv = self.adv_out(adv_h).squeeze(-1)
        adv = adv - adv.mean(dim=-1, keepdim=True)

        q = value + adv
        q = q * alive_mask.unsqueeze(-1)
        return q


class DoublePairwiseRelayCritic(nn.Module):
    def __init__(self, cfg: HATD3D3QNConfig):
        super().__init__()
        self.q1 = PairwiseRelayCritic(cfg)
        self.q2 = PairwiseRelayCritic(cfg)

    def forward(
        self,
        obs: torch.Tensor,
        region_id: torch.Tensor,
        cont_action: torch.Tensor,
        server_feat: torch.Tensor,
        hop_matrix: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(obs, region_id, cont_action, server_feat, hop_matrix, alive_mask)
        q2 = self.q2(obs, region_id, cont_action, server_feat, hop_matrix, alive_mask)
        return q1, q2

    def q1_only(
        self,
        obs: torch.Tensor,
        region_id: torch.Tensor,
        cont_action: torch.Tensor,
        server_feat: torch.Tensor,
        hop_matrix: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.q1(obs, region_id, cont_action, server_feat, hop_matrix, alive_mask)

    def q2_only(
        self,
        obs: torch.Tensor,
        region_id: torch.Tensor,
        cont_action: torch.Tensor,
        server_feat: torch.Tensor,
        hop_matrix: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.q2(obs, region_id, cont_action, server_feat, hop_matrix, alive_mask)


class ReplayBuffer:
    def __init__(self, cfg: HATD3D3QNConfig):
        self.cfg = cfg
        self.max_size = cfg.buffer_size
        self.ptr = 0
        self.size = 0

        N = cfg.num_agents
        M = cfg.num_servers
        O = cfg.obs_dim
        S = cfg.server_feat_dim

        self.obs = np.zeros((self.max_size, N, O), dtype=np.float32)
        self.region_id = np.zeros((self.max_size, N), dtype=np.int64)
        self.cont_action = np.zeros((self.max_size, N, 3), dtype=np.float32)
        self.relay_target = np.zeros((self.max_size, N), dtype=np.int64)
        self.reward = np.zeros((self.max_size,), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, N, O), dtype=np.float32)
        self.next_region_id = np.zeros((self.max_size, N), dtype=np.int64)
        self.server_feat = np.zeros((self.max_size, M, S), dtype=np.float32)
        self.next_server_feat = np.zeros((self.max_size, M, S), dtype=np.float32)
        self.hop_matrix = np.zeros((self.max_size, M, M), dtype=np.float32)
        self.next_hop_matrix = np.zeros((self.max_size, M, M), dtype=np.float32)
        self.alive_mask = np.ones((self.max_size, N), dtype=np.float32)
        self.next_alive_mask = np.ones((self.max_size, N), dtype=np.float32)
        self.done = np.zeros((self.max_size,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        region_id: np.ndarray,
        cont_action: np.ndarray,
        relay_target: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_region_id: np.ndarray,
        server_feat: np.ndarray,
        next_server_feat: np.ndarray,
        hop_matrix: np.ndarray,
        next_hop_matrix: np.ndarray,
        alive_mask: np.ndarray,
        next_alive_mask: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.region_id[self.ptr] = region_id
        self.cont_action[self.ptr] = cont_action
        self.relay_target[self.ptr] = relay_target
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_region_id[self.ptr] = next_region_id
        self.server_feat[self.ptr] = server_feat
        self.next_server_feat[self.ptr] = next_server_feat
        self.hop_matrix[self.ptr] = hop_matrix
        self.next_hop_matrix[self.ptr] = next_hop_matrix
        self.alive_mask[self.ptr] = alive_mask
        self.next_alive_mask[self.ptr] = next_alive_mask
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[idx], device=device),
            "region_id": torch.tensor(self.region_id[idx], device=device),
            "cont_action": torch.tensor(self.cont_action[idx], device=device),
            "relay_target": torch.tensor(self.relay_target[idx], device=device),
            "reward": torch.tensor(self.reward[idx], device=device).unsqueeze(-1),
            "next_obs": torch.tensor(self.next_obs[idx], device=device),
            "next_region_id": torch.tensor(self.next_region_id[idx], device=device),
            "server_feat": torch.tensor(self.server_feat[idx], device=device),
            "next_server_feat": torch.tensor(self.next_server_feat[idx], device=device),
            "hop_matrix": torch.tensor(self.hop_matrix[idx], device=device),
            "next_hop_matrix": torch.tensor(self.next_hop_matrix[idx], device=device),
            "alive_mask": torch.tensor(self.alive_mask[idx], device=device),
            "next_alive_mask": torch.tensor(self.next_alive_mask[idx], device=device),
            "done": torch.tensor(self.done[idx], device=device).unsqueeze(-1),
        }


class HATD3D3QNAgent:
    def __init__(self, cfg: HATD3D3QNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        set_seed(cfg.seed)

        self.actor = SharedActor(cfg).to(self.device)
        self.critic = DoublePairwiseRelayCritic(cfg).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, foreach=False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, foreach=False)

        self.replay_buffer = ReplayBuffer(cfg)
        self.total_it = 0
        self.relay_epsilon = float(cfg.relay_epsilon_start)
        self.explore_noise = float(cfg.explore_noise_start)

    def _mask_dead_cont_action(self, cont_action: torch.Tensor, alive_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if alive_mask is None:
            return cont_action
        if alive_mask.ndim == cont_action.ndim - 1:
            alive_mask = alive_mask.unsqueeze(-1)
        return cont_action * alive_mask.float()

    def _mask_dead_relay_target(self, relay_target: torch.Tensor, alive_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if alive_mask is None:
            return relay_target
        return relay_target * alive_mask.long()

    def _decay_relay_epsilon(self) -> None:
        if self.total_it >= self.cfg.relay_epsilon_decay_start_it:
            self.relay_epsilon = max(self.cfg.relay_epsilon_min, self.relay_epsilon * self.cfg.relay_epsilon_decay)

    def _decay_explore_noise(self) -> None:
        self.explore_noise = max(self.cfg.explore_noise_min, self.explore_noise * self.cfg.explore_noise_decay)

    def on_episode_end(self, episode_idx: int) -> None:
        if self.cfg.relay_epsilon_decay_per_episode and episode_idx >= self.cfg.relay_epsilon_decay_start_episode:
            self.relay_epsilon = max(self.cfg.relay_epsilon_min, self.relay_epsilon * self.cfg.relay_epsilon_decay)
        if self.cfg.explore_noise_decay_per_episode and episode_idx >= self.cfg.explore_noise_decay_start_episode:
            self._decay_explore_noise()

    def _sample_random_relay_target(self, region_id: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        relay_target = torch.randint(0, self.cfg.num_servers, size=region_id.shape, device=region_id.device)
        relay_target = self._mask_dead_relay_target(relay_target, alive_mask)
        return relay_target

    @torch.no_grad()
    def select_cont_action(
        self,
        obs: np.ndarray | torch.Tensor,
        region_id: np.ndarray | torch.Tensor,
        alive_mask: Optional[np.ndarray | torch.Tensor] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device).float()

        if not torch.is_tensor(region_id):
            region_id = torch.tensor(region_id, dtype=torch.long, device=self.device)
        else:
            region_id = region_id.to(self.device).long()

        add_batch = False
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
            region_id = region_id.unsqueeze(0)
            add_batch = True

        cont_action = self.actor.act(
            obs,
            region_id,
            deterministic=deterministic,
            noise_std=0.0 if deterministic else self.explore_noise,
            noise_clip=self.cfg.noise_clip,
        )

        if alive_mask is not None:
            if not torch.is_tensor(alive_mask):
                alive_mask = torch.tensor(alive_mask, dtype=torch.float32, device=self.device)
            else:
                alive_mask = alive_mask.to(self.device).float()
            if add_batch and alive_mask.ndim == 1:
                alive_mask = alive_mask.unsqueeze(0)
            cont_action = self._mask_dead_cont_action(cont_action, alive_mask)

        if add_batch:
            cont_action = cont_action.squeeze(0)
        return cont_action.detach().cpu().numpy()

    @torch.no_grad()
    def select_relay_target(
        self,
        obs: np.ndarray | torch.Tensor,
        region_id: np.ndarray | torch.Tensor,
        server_feat: np.ndarray | torch.Tensor,
        hop_matrix: np.ndarray | torch.Tensor,
        alive_mask: Optional[np.ndarray | torch.Tensor] = None,
        cont_action: Optional[np.ndarray | torch.Tensor] = None,
        deterministic_cont: bool = False,
        deterministic_relay: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device).float()

        if not torch.is_tensor(region_id):
            region_id = torch.tensor(region_id, dtype=torch.long, device=self.device)
        else:
            region_id = region_id.to(self.device).long()

        if not torch.is_tensor(server_feat):
            server_feat = torch.tensor(server_feat, dtype=torch.float32, device=self.device)
        else:
            server_feat = server_feat.to(self.device).float()

        if not torch.is_tensor(hop_matrix):
            hop_matrix = torch.tensor(hop_matrix, dtype=torch.float32, device=self.device)
        else:
            hop_matrix = hop_matrix.to(self.device).float()

        if alive_mask is None:
            alive_mask = torch.ones(*obs.shape[:-1], device=self.device)
        elif not torch.is_tensor(alive_mask):
            alive_mask = torch.tensor(alive_mask, dtype=torch.float32, device=self.device)
        else:
            alive_mask = alive_mask.to(self.device).float()

        add_batch = False
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
            region_id = region_id.unsqueeze(0)
            server_feat = server_feat.unsqueeze(0)
            hop_matrix = hop_matrix.unsqueeze(0)
            alive_mask = alive_mask.unsqueeze(0)
            add_batch = True

        if cont_action is None:
            cont_action = self.actor.act(
                obs,
                region_id,
                deterministic=deterministic_cont,
                noise_std=0.0 if deterministic_cont else self.explore_noise,
                noise_clip=self.cfg.noise_clip,
            )
        else:
            if not torch.is_tensor(cont_action):
                cont_action = torch.tensor(cont_action, dtype=torch.float32, device=self.device)
            else:
                cont_action = cont_action.to(self.device).float()
            if cont_action.ndim == 2:
                cont_action = cont_action.unsqueeze(0)

        cont_action = self._mask_dead_cont_action(cont_action, alive_mask)
        q1, q2 = self.critic(obs, region_id, cont_action, server_feat, hop_matrix, alive_mask)
        q_min = torch.min(q1, q2)
        greedy_relay_target = torch.argmax(q_min, dim=-1)
        greedy_relay_target = self._mask_dead_relay_target(greedy_relay_target, alive_mask)

        if deterministic_relay or self.relay_epsilon <= 0.0:
            relay_target = greedy_relay_target
        else:
            random_mask = (torch.rand_like(greedy_relay_target.float()) < self.relay_epsilon) & (alive_mask > 0.5)
            random_relay_target = self._sample_random_relay_target(region_id, alive_mask)
            relay_target = torch.where(random_mask, random_relay_target, greedy_relay_target)

        if add_batch:
            return cont_action.squeeze(0).cpu().numpy(), relay_target.squeeze(0).cpu().numpy()
        return cont_action.cpu().numpy(), relay_target.cpu().numpy()

    @torch.no_grad()
    def select_disc_action(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """向后兼容：discrete 动作现在语义上等同于 relay target。"""
        return self.select_relay_target(*args, **kwargs)

    def store_transition(
        self,
        obs: np.ndarray,
        region_id: np.ndarray,
        cont_action: np.ndarray,
        relay_target: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_region_id: np.ndarray,
        server_feat: np.ndarray,
        next_server_feat: np.ndarray,
        hop_matrix: np.ndarray,
        next_hop_matrix: np.ndarray,
        alive_mask: np.ndarray,
        next_alive_mask: np.ndarray,
        done: float,
    ) -> None:
        alive_mask_f = alive_mask.astype(np.float32)
        cont_action = cont_action * alive_mask_f[..., None]
        relay_target = relay_target * alive_mask.astype(np.int64)
        self.replay_buffer.add(
            obs=obs,
            region_id=region_id,
            cont_action=cont_action,
            relay_target=relay_target,
            reward=reward,
            next_obs=next_obs,
            next_region_id=next_region_id,
            server_feat=server_feat,
            next_server_feat=next_server_feat,
            hop_matrix=hop_matrix,
            next_hop_matrix=next_hop_matrix,
            alive_mask=alive_mask,
            next_alive_mask=next_alive_mask,
            done=done,
        )

    def train_step(self) -> Dict[str, float]:
        if self.replay_buffer.size < self.cfg.batch_size:
            return {}

        self.total_it += 1
        batch = self.replay_buffer.sample(self.cfg.batch_size, self.device)

        obs = batch["obs"]
        region_id = batch["region_id"]
        cont_action = batch["cont_action"]
        relay_target = batch["relay_target"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        next_region_id = batch["next_region_id"]
        server_feat = batch["server_feat"]
        next_server_feat = batch["next_server_feat"]
        hop_matrix = batch["hop_matrix"]
        next_hop_matrix = batch["next_hop_matrix"]
        alive_mask = batch["alive_mask"]
        next_alive_mask = batch["next_alive_mask"]
        done = batch["done"]

        mask_now = alive_mask.float()

        next_cont_action = self.actor_target.act(
            next_obs,
            next_region_id,
            deterministic=False,
            noise_std=self.cfg.policy_noise,
            noise_clip=self.cfg.noise_clip,
        )
        next_cont_action = self._mask_dead_cont_action(next_cont_action, next_alive_mask)

        online_next_q1_all, online_next_q2_all = self.critic(
            next_obs,
            next_region_id,
            next_cont_action,
            next_server_feat,
            next_hop_matrix,
            next_alive_mask,
        )
        online_next_q_min_all = torch.min(online_next_q1_all, online_next_q2_all)
        next_relay_target = torch.argmax(online_next_q_min_all, dim=-1)
        next_relay_target = self._mask_dead_relay_target(next_relay_target, next_alive_mask)
        next_relay_target_idx = next_relay_target.unsqueeze(-1)

        target_q1_all, target_q2_all = self.critic_target(
            next_obs,
            next_region_id,
            next_cont_action,
            next_server_feat,
            next_hop_matrix,
            next_alive_mask,
        )
        target_q_min_all = torch.min(target_q1_all, target_q2_all)
        target_q = target_q_min_all.gather(dim=-1, index=next_relay_target_idx).squeeze(-1)

        target_q_mean = target_q.mean(dim=1, keepdim=True)
        y = reward + (1.0 - done) * self.cfg.gamma * target_q_mean

        cont_action = self._mask_dead_cont_action(cont_action, alive_mask)
        relay_target = self._mask_dead_relay_target(relay_target, alive_mask)

        current_q1_all, current_q2_all = self.critic(
            obs,
            region_id,
            cont_action,
            server_feat,
            hop_matrix,
            alive_mask,
        )

        relay_target_idx = relay_target.unsqueeze(-1)
        current_q1 = current_q1_all.gather(dim=-1, index=relay_target_idx).squeeze(-1)
        current_q2 = current_q2_all.gather(dim=-1, index=relay_target_idx).squeeze(-1)

        current_q1_mean = current_q1.mean(dim=1, keepdim=True)
        current_q2_mean = current_q2.mean(dim=1, keepdim=True)
        critic_loss = F.mse_loss(current_q1_mean, y) + F.mse_loss(current_q2_mean, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_optimizer.step()

        info = {
            "critic_loss": float(critic_loss.item()),
            "target_q": float(y.mean().item()),
            "current_q1": float(current_q1_mean.mean().item()),
            "current_q2": float(current_q2_mean.mean().item()),
        }

        if self.total_it % self.cfg.policy_delay == 0:
            new_cont_action = self.actor(obs, region_id)
            new_cont_action = self._mask_dead_cont_action(new_cont_action, alive_mask)

            q1_all = self.critic.q1_only(obs, region_id, new_cont_action, server_feat, hop_matrix, alive_mask)
            q2_all = self.critic.q2_only(obs, region_id, new_cont_action, server_feat, hop_matrix, alive_mask)
            q_min_all = torch.min(q1_all, q2_all)

            best_q_per_agent = q_min_all.max(dim=-1)[0]
            alive_denom = mask_now.sum(dim=1).clamp(min=1.0)
            actor_obj = ((best_q_per_agent * mask_now).sum(dim=1) / alive_denom).mean()
            actor_loss = -actor_obj

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
            self.actor_optimizer.step()

            soft_update(self.actor_target, self.actor, self.cfg.tau)
            soft_update(self.critic_target, self.critic, self.cfg.tau)

            info["actor_loss"] = float(actor_loss.item())
            info["actor_obj"] = float(actor_obj.item())

        if not self.cfg.relay_epsilon_decay_per_episode:
            self._decay_relay_epsilon()
        if not self.cfg.explore_noise_decay_per_episode:
            self._decay_explore_noise()
        info["relay_epsilon"] = float(self.relay_epsilon)
        info["explore_noise"] = float(self.explore_noise)
        return info

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
                "relay_epsilon": self.relay_epsilon,
                "explore_noise": self.explore_noise,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.total_it = ckpt["total_it"]
        self.relay_epsilon = float(ckpt.get("relay_epsilon", self.cfg.relay_epsilon_start))
        self.explore_noise = float(ckpt.get("explore_noise", self.cfg.explore_noise_start))


def build_default_agent() -> HATD3D3QNAgent:
    cfg = HATD3D3QNConfig()
    return HATD3D3QNAgent(cfg)
