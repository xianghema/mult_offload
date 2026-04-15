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
    """Version ha_td3_d3qn_v5.py: pairwise home-referenced dueling D3QN."""
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
    buffer_size: int = 400000

    # replay sampling
    replay_strategy: str = "mixed_recent_uniform"  # uniform | mixed_recent_uniform | mixed_recent_per | per
    recent_ratio: float = 0.30
    recent_window: int = 20000
    per_alpha: float = 0.4
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 1
    per_eps: float = 1e-4

    # TD3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    explore_noise: float = 0.1
    policy_delay: int = 2
    grad_clip: float = 1.0

    # relay-target exploration (discrete epsilon-greedy)
    relay_epsilon_start: float = 0.20
    relay_epsilon_min: float = 0.05
    relay_epsilon_decay: float = 0.9995
    relay_epsilon_decay_start_it: int = 50
    relay_epsilon_decay_per_episode: bool = True
    relay_epsilon_decay_start_episode: int = 60

    # continuous-action exploration schedule
    explore_noise_start: float = 0.10
    explore_noise_min: float = 0.03
    explore_noise_decay: float = 0.9996
    explore_noise_decay_per_episode: bool = True
    explore_noise_decay_start_episode: int = 60

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_orthogonal_init: bool = True
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
    方案二：home-referenced dueling 的 pairwise relay critic。

    对每个设备 n 和候选二次转发目标 j，显式建模：
        Q(n, j) = V_home(n) + A(n, j) - A(n, home)

    这样可保证：
        Q(n, home) = V_home(n)

    与论文叙事一致：默认本区域服务器处理是基线，其它服务器表示相对于 home 的
    二次转发收益修正。
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
        adv_raw = self.adv_out(adv_h).squeeze(-1)

        home_adv = adv_raw.gather(dim=-1, index=region_id.unsqueeze(-1)).squeeze(-1)
        adv = adv_raw - home_adv.unsqueeze(-1)

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
        self.last_sample_info: Dict[str, float] = {}

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
        self.priorities = np.ones((self.max_size,), dtype=np.float32)

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
        if self.size > 0:
            max_pri = float(np.max(self.priorities[: self.size]))
            self.priorities[self.ptr] = max(max_pri, 1.0)
        else:
            self.priorities[self.ptr] = 1.0

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _valid_indices(self) -> np.ndarray:
        if self.size < self.max_size:
            return np.arange(self.size, dtype=np.int64)
        return np.arange(self.max_size, dtype=np.int64)

    def _recent_ranges(self, recent_window: int) -> tuple[list[tuple[int, int]], int]:
        if self.size <= 0:
            return [], 0

        recent_size = int(min(max(recent_window, 1), self.size))
        if self.size < self.max_size:
            start = self.size - recent_size
            return [(start, self.size)], recent_size

        start = (self.ptr - recent_size) % self.max_size
        end = self.ptr
        if start < end:
            return [(start, end)], recent_size
        if start > end:
            return [(start, self.max_size), (0, end)], recent_size
        return [(0, self.max_size)], recent_size

    @staticmethod
    def _sample_from_ranges(ranges: list[tuple[int, int]], n: int, replace: bool = True) -> np.ndarray:
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        lengths = np.asarray([max(0, e - s) for s, e in ranges], dtype=np.int64)
        total = int(lengths.sum())
        if total <= 0:
            return np.zeros((0,), dtype=np.int64)
        if replace:
            offset = np.random.randint(0, total, size=n, dtype=np.int64)
        else:
            if n >= total:
                offset = np.arange(total, dtype=np.int64)
                np.random.shuffle(offset)
                if n > total:
                    pad = np.random.randint(0, total, size=n - total, dtype=np.int64)
                    offset = np.concatenate([offset, pad], axis=0)
            else:
                offset = np.random.choice(total, size=n, replace=False).astype(np.int64)
        cum = np.cumsum(lengths)
        seg = np.searchsorted(cum, offset, side="right")
        base = offset - np.where(seg > 0, cum[seg - 1], 0)
        starts = np.asarray([ranges[i][0] for i in seg], dtype=np.int64)
        return starts + base

    @staticmethod
    def _ranges_to_indices(ranges: list[tuple[int, int]]) -> np.ndarray:
        if not ranges:
            return np.zeros((0,), dtype=np.int64)
        chunks = [np.arange(s, e, dtype=np.int64) for s, e in ranges if e > s]
        if not chunks:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _complement_ranges(domain_start: int, domain_end: int, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if domain_end <= domain_start:
            return []
        if not ranges:
            return [(domain_start, domain_end)]
        merged = sorted((max(domain_start, s), min(domain_end, e)) for s, e in ranges if e > s)
        comp: list[tuple[int, int]] = []
        cursor = domain_start
        for s, e in merged:
            if cursor < s:
                comp.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < domain_end:
            comp.append((cursor, domain_end))
        return comp

    def _sample_uniform(
        self,
        batch_size: int,
        recent_window: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if self.size >= batch_size:
            idx = np.random.choice(self.size, size=batch_size, replace=False).astype(np.int64)
        else:
            idx = np.random.randint(0, self.size, size=batch_size, dtype=np.int64)
        recent_ranges, _ = self._recent_ranges(recent_window)
        if not recent_ranges:
            return idx, np.ones((batch_size,), dtype=np.float32), 0.0
        is_recent = np.zeros((batch_size,), dtype=bool)
        for s, e in recent_ranges:
            is_recent |= (idx >= s) & (idx < e)
        return idx, np.ones((batch_size,), dtype=np.float32), float(is_recent.mean())

    def _sample_mixed_recent_uniform(
        self,
        batch_size: int,
        recent_ratio: float,
        recent_window: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))
        recent_ranges, recent_size = self._recent_ranges(recent_window)
        if recent_size <= 0:
            return self._sample_uniform(batch_size=batch_size, recent_window=recent_window)

        n_recent = int(round(batch_size * recent_ratio))
        n_recent = int(np.clip(n_recent, 0, batch_size))
        domain_end = self.size if self.size < self.max_size else self.max_size
        global_ranges = self._complement_ranges(0, domain_end, recent_ranges)
        global_size = int(sum(max(0, e - s) for s, e in global_ranges))

        n_recent = min(n_recent, recent_size)
        n_global = batch_size - n_recent
        if n_global > global_size:
            n_global = global_size
            n_recent = batch_size - n_global
        n_recent = min(n_recent, recent_size)

        recent_idx = self._sample_from_ranges(recent_ranges, n_recent, replace=False)
        global_idx = self._sample_from_ranges(global_ranges, n_global, replace=False)

        idx = np.concatenate([recent_idx, global_idx], axis=0)
        if idx.shape[0] < batch_size:
            need = batch_size - idx.shape[0]
            pool = np.arange(domain_end, dtype=np.int64)
            if idx.shape[0] > 0:
                used = np.zeros((domain_end,), dtype=bool)
                used[idx] = True
                remain = pool[~used]
            else:
                remain = pool
            if remain.shape[0] >= need:
                pad = np.random.choice(remain, size=need, replace=False).astype(np.int64)
            elif remain.shape[0] > 0:
                extra = np.random.randint(0, domain_end, size=need - remain.shape[0], dtype=np.int64)
                pad = np.concatenate([remain, extra], axis=0)
            else:
                pad = np.random.randint(0, domain_end, size=need, dtype=np.int64)
            idx = np.concatenate([idx, pad], axis=0)
        elif idx.shape[0] > batch_size:
            idx = idx[:batch_size]
        np.random.shuffle(idx)
        if recent_ranges:
            is_recent = np.zeros((batch_size,), dtype=bool)
            for s, e in recent_ranges:
                is_recent |= (idx >= s) & (idx < e)
            sample_recent_ratio = float(is_recent.mean())
        else:
            sample_recent_ratio = 0.0
        return idx, np.ones((batch_size,), dtype=np.float32), sample_recent_ratio

    def _sample_per(
        self,
        batch_size: int,
        alpha: float,
        beta: float,
        eps: float,
        recent_window: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        valid_idx = self._valid_indices()
        pri = self.priorities[valid_idx].astype(np.float64)
        scaled = np.power(np.maximum(pri, 0.0) + max(float(eps), 1e-12), max(float(alpha), 0.0))
        z = float(np.sum(scaled))
        if z <= 0.0 or (not np.isfinite(z)):
            probs = np.full_like(scaled, 1.0 / max(len(scaled), 1), dtype=np.float64)
        else:
            probs = scaled / z

        draw = np.random.choice(len(valid_idx), size=batch_size, replace=True, p=probs)
        idx = valid_idx[draw]
        p_i = probs[draw]
        beta = float(np.clip(beta, 0.0, 1.0))
        weights = np.power(max(len(valid_idx), 1) * np.maximum(p_i, 1e-12), -beta)
        weights = weights / max(np.max(weights), 1e-12)
        weights = weights.astype(np.float32)

        recent_ranges, _ = self._recent_ranges(recent_window)
        if recent_ranges:
            is_recent = np.zeros((batch_size,), dtype=bool)
            for s, e in recent_ranges:
                is_recent |= (idx >= s) & (idx < e)
            sample_recent_ratio = float(is_recent.mean())
        else:
            sample_recent_ratio = 0.0

        priority_mean = float(np.mean(pri))
        priority_max = float(np.max(pri))
        return idx, weights, sample_recent_ratio, priority_mean, priority_max

    def _sample_mixed_recent_per(
        self,
        batch_size: int,
        recent_ratio: float,
        recent_window: int,
        alpha: float,
        beta: float,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))
        beta = float(np.clip(beta, 0.0, 1.0))
        eps = float(max(eps, 1e-12))
        alpha = float(max(alpha, 0.0))

        valid_idx = self._valid_indices()
        valid_size = len(valid_idx)
        if valid_size <= 0:
            raise RuntimeError("ReplayBuffer has no valid indices.")

        recent_ranges, recent_size = self._recent_ranges(recent_window)
        if recent_size <= 0:
            return self._sample_per(
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                eps=eps,
                recent_window=recent_window,
            )

        domain_end = self.size if self.size < self.max_size else self.max_size
        global_ranges = self._complement_ranges(0, domain_end, recent_ranges)
        global_pool = self._ranges_to_indices(global_ranges)
        global_size = int(global_pool.shape[0])

        n_recent = int(round(batch_size * recent_ratio))
        n_recent = int(np.clip(n_recent, 0, batch_size))
        n_recent = min(n_recent, recent_size)
        n_global = batch_size - n_recent
        if n_global > global_size:
            n_global = global_size
            n_recent = batch_size - n_global
            n_recent = min(n_recent, recent_size)

        recent_idx = self._sample_from_ranges(recent_ranges, n_recent, replace=False)
        p_recent = np.full((recent_idx.shape[0],), 1.0 / max(recent_size, 1), dtype=np.float64)

        global_idx = np.zeros((0,), dtype=np.int64)
        p_global = np.zeros((0,), dtype=np.float64)
        if n_global > 0 and global_size > 0:
            pri_global = self.priorities[global_pool].astype(np.float64)
            scaled = np.power(np.maximum(pri_global, 0.0) + eps, alpha)
            z = float(np.sum(scaled))
            if z <= 0.0 or (not np.isfinite(z)):
                probs_global = np.full((global_size,), 1.0 / max(global_size, 1), dtype=np.float64)
            else:
                probs_global = scaled / z
            draw = np.random.choice(global_size, size=n_global, replace=False, p=probs_global)
            global_idx = global_pool[draw]
            p_global = probs_global[draw]

        idx = np.concatenate([recent_idx, global_idx], axis=0)
        if idx.shape[0] < batch_size:
            need = batch_size - idx.shape[0]
            pool = np.arange(domain_end, dtype=np.int64)
            used = np.zeros((domain_end,), dtype=bool)
            if idx.shape[0] > 0:
                used[idx] = True
            remain = pool[~used]
            if remain.shape[0] >= need:
                pad_idx = np.random.choice(remain, size=need, replace=False).astype(np.int64)
            elif remain.shape[0] > 0:
                extra = np.random.randint(0, domain_end, size=need - remain.shape[0], dtype=np.int64)
                pad_idx = np.concatenate([remain, extra], axis=0)
            else:
                pad_idx = np.random.randint(0, domain_end, size=need, dtype=np.int64)
            idx = np.concatenate([idx, pad_idx], axis=0)
            p_pad = np.full((pad_idx.shape[0],), 1.0 / max(domain_end, 1), dtype=np.float64)
        else:
            p_pad = np.zeros((0,), dtype=np.float64)

        p_raw = np.concatenate([p_recent, p_global, p_pad], axis=0)
        mix_recent = float(n_recent) / max(float(batch_size), 1.0)
        mix_global = float(n_global) / max(float(batch_size), 1.0)
        mix_pad = max(0.0, 1.0 - mix_recent - mix_global)

        # Approximate mixture sampling probability for IS correction.
        probs = np.zeros_like(p_raw, dtype=np.float64)
        if p_recent.shape[0] > 0:
            probs[: p_recent.shape[0]] = mix_recent * p_recent
        if p_global.shape[0] > 0:
            start = p_recent.shape[0]
            probs[start : start + p_global.shape[0]] = mix_global * p_global
        if p_pad.shape[0] > 0:
            start = p_recent.shape[0] + p_global.shape[0]
            probs[start:] = mix_pad * p_pad + 1e-12

        perm = np.random.permutation(idx.shape[0])
        idx = idx[perm]
        probs = probs[perm]

        if idx.shape[0] > batch_size:
            idx = idx[:batch_size]
            probs = probs[:batch_size]
        elif idx.shape[0] < batch_size:
            need = batch_size - idx.shape[0]
            pad = np.random.randint(0, domain_end, size=need, dtype=np.int64)
            idx = np.concatenate([idx, pad], axis=0)
            probs = np.concatenate([probs, np.full((need,), 1.0 / max(domain_end, 1), dtype=np.float64)], axis=0)

        weights = np.power(max(valid_size, 1) * np.maximum(probs, 1e-12), -beta)
        weights = weights / max(np.max(weights), 1e-12)
        weights = weights.astype(np.float32)

        is_recent = np.zeros((batch_size,), dtype=bool)
        for s, e in recent_ranges:
            is_recent |= (idx >= s) & (idx < e)
        sample_recent_ratio = float(is_recent.mean())

        pri_all = self.priorities[valid_idx].astype(np.float64)
        priority_mean = float(np.mean(pri_all))
        priority_max = float(np.max(pri_all))
        return idx, weights, sample_recent_ratio, priority_mean, priority_max

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, per_eps: float) -> None:
        if indices is None or td_errors is None:
            return
        if len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        td_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        if td_errors.shape[0] != indices.shape[0]:
            n = min(td_errors.shape[0], indices.shape[0])
            indices = indices[:n]
            td_errors = td_errors[:n]
        pri = np.abs(td_errors) + float(max(per_eps, 1e-12))
        unique_idx = np.unique(indices)
        for idx_i in unique_idx:
            self.priorities[int(idx_i)] = float(np.max(pri[indices == idx_i]))

    def sample(
        self,
        batch_size: int,
        device: str,
        strategy: str = "uniform",
        recent_ratio: float = 0.3,
        recent_window: int = 20000,
        per_alpha: float = 0.4,
        per_beta: float = 0.4,
        per_eps: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        if self.size <= 0:
            raise RuntimeError("ReplayBuffer is empty.")

        strategy = strategy.lower().strip()
        sample_recent_ratio = 0.0
        priority_mean = float(np.mean(self.priorities[: max(self.size, 1)]))
        priority_max = float(np.max(self.priorities[: max(self.size, 1)]))

        if strategy == "uniform":
            idx, weights_np, sample_recent_ratio = self._sample_uniform(
                batch_size=batch_size,
                recent_window=recent_window,
            )
        elif strategy == "mixed_recent_uniform":
            idx, weights_np, sample_recent_ratio = self._sample_mixed_recent_uniform(
                batch_size=batch_size,
                recent_ratio=recent_ratio,
                recent_window=recent_window,
            )
        elif strategy == "mixed_recent_per":
            idx, weights_np, sample_recent_ratio, priority_mean, priority_max = self._sample_mixed_recent_per(
                batch_size=batch_size,
                recent_ratio=recent_ratio,
                recent_window=recent_window,
                alpha=per_alpha,
                beta=per_beta,
                eps=per_eps,
            )
        elif strategy == "per":
            idx, weights_np, sample_recent_ratio, priority_mean, priority_max = self._sample_per(
                batch_size=batch_size,
                alpha=per_alpha,
                beta=per_beta,
                eps=per_eps,
                recent_window=recent_window,
            )
        else:
            raise ValueError(f"Unsupported replay strategy: {strategy}")

        self.last_sample_info = {
            "sample_recent_ratio": float(sample_recent_ratio),
            "priority_mean": float(priority_mean),
            "priority_max": float(priority_max),
            "is_weight_mean": float(np.mean(weights_np)),
        }

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
            "indices": torch.tensor(idx, device=device, dtype=torch.long),
            "weights": torch.tensor(weights_np, device=device).unsqueeze(-1),
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
        per_progress = min(1.0, self.total_it / max(float(self.cfg.per_beta_steps), 1.0))
        per_beta = self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start) * per_progress
        batch = self.replay_buffer.sample(
            batch_size=self.cfg.batch_size,
            device=self.device,
            strategy=self.cfg.replay_strategy,
            recent_ratio=self.cfg.recent_ratio,
            recent_window=self.cfg.recent_window,
            per_alpha=self.cfg.per_alpha,
            per_beta=per_beta,
            per_eps=self.cfg.per_eps,
        )

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
        sample_indices = batch["indices"]
        sample_weights = batch["weights"]

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
        td_err1 = current_q1_mean - y
        td_err2 = current_q2_mean - y
        critic_loss = ((td_err1.pow(2) + td_err2.pow(2)) * sample_weights).mean()
        td_abs_mean = 0.5 * (td_err1.abs() + td_err2.abs())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_optimizer.step()

        if self.cfg.replay_strategy in ("per", "mixed_recent_per"):
            self.replay_buffer.update_priorities(
                indices=sample_indices.detach().cpu().numpy(),
                td_errors=td_abs_mean.detach().cpu().numpy(),
                per_eps=self.cfg.per_eps,
            )

        info = {
            "critic_loss": float(critic_loss.item()),
            "target_q": float(y.mean().item()),
            "current_q1": float(current_q1_mean.mean().item()),
            "current_q2": float(current_q2_mean.mean().item()),
            "sample_recent_ratio": float(self.replay_buffer.last_sample_info.get("sample_recent_ratio", 0.0)),
            "priority_mean": float(self.replay_buffer.last_sample_info.get("priority_mean", 0.0)),
            "priority_max": float(self.replay_buffer.last_sample_info.get("priority_max", 0.0)),
            "is_weight_mean": float(self.replay_buffer.last_sample_info.get("is_weight_mean", 1.0)),
            "per_beta": float(per_beta),
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
