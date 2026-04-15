
from __future__ import annotations

"""
第四章最终训练环境（重构版）
--------------------------------
目标：
1. 与 HA-TD3-D3QN 训练脚本接口对齐；
2. 保留第四章系统模型与问题公式化中的关键物理过程；
3. 精简观测与环境职责，避免将网络结构细节和论文分析量混入主状态；
4. 奖励采用“固定 N 平均团队奖励”，每回合累计该标量即可得到“每回合累计平均奖励”。

环境对外接口
------------
reset() -> state, info
state = {
    "obs":        np.ndarray, shape [N, 7]
    "region_id":  np.ndarray, shape [N]
    "server_feat":np.ndarray, shape [M, 4]
    "hop_matrix": np.ndarray, shape [M, M]   # 归一化 hop 矩阵，显式提供给 Critic
    "alive_mask": np.ndarray, shape [N]
}

step(cont_action, relay_target) -> next_state, reward, terminated, truncated, info
- cont_action: [N, 3], 每行 [xi, f_L(GHz), p_R(W)]，均为真实物理量
- relay_target: [N], 每个元素表示区域服务器为该任务选择的二次转发目标服务器编号（0 ~ M-1）

奖励
----
单设备奖励采用第三章分层逻辑的多设备扩展：
1) 已掉电设备：奖励记 0，不再重复惩罚；
2) 当前动作导致不可行 / 掉电：固定惩罚 -k_inv；
3) 可执行但超时：按超时程度惩罚；
4) 可执行且未超时：动态权重下的时延效用 + 能耗效用，并叠加动态能耗约束惩罚。

团队奖励定义：
    r_team(t) = sum_n r_n(t)
训练时返回：
    r_train(t) = r_team(t) / N

额外统计（用于训练诊断与论文指标）：
- success_count(t): 本步中“可执行、未掉电、未超时”的成功任务数
- task_dropped_count(t): 本步中被判定为丢弃/失败的任务数
- executed_count(t): 本步中未被丢弃、真正执行了动作的任务数（含超时任务）
- executed_delay_sum(t): 本步 executed_count 对应的总时延
- executed_energy_sum(t): 本步 executed_count 对应的总能耗

注意：
- avg_delay / avg_energy 若希望与第三章训练代码保持一致，
  应按“未被丢弃的样本”统计，而不是只按 success 样本统计。
- drop_count 与 dead_count 需要分开：drop 表示任务被丢弃，dead 表示设备掉电事件。
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# gym / gymnasium 兼容层：当前训练脚本其实只依赖 reset/step 接口，
# 因此在没有 gymnasium/gym 时，也允许环境独立运行。
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except Exception:  # pragma: no cover
        class _DummyEnv:
            metadata = {"render_modes": []}

            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    self.np_random = np.random.default_rng(seed)
                elif not hasattr(self, "np_random"):
                    self.np_random = np.random.default_rng()
                return None

        class _Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _MultiDiscrete:
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec)

        class _MultiBinary:
            def __init__(self, n):
                self.n = n

        class _DictSpace(dict):
            def __init__(self, mapping):
                super().__init__(mapping)

        class _Spaces:
            Box = _Box
            MultiDiscrete = _MultiDiscrete
            MultiBinary = _MultiBinary
            Dict = _DictSpace

        class _GymModule:
            Env = _DummyEnv

        gym = _GymModule()
        spaces = _Spaces()


def get_default_chapter4_final_config() -> Dict:
    return {
        # ===== 系统规模 =====
        "num_devices": 30,
        "num_servers": 6,  # 默认支持 1~10，使用母图前缀子图
        "region_assignment_mode": "balanced",  # balanced / manual
        "manual_region_assignment": None,

        # ===== 时间与任务 =====
        "T_f": 0.2,
        "max_steps": 100,
        "D_values": [2.3e5, 2.4e5, 2.5e5, 2.6e5, 2.7e5, 2.8e5],  # bit
        "avg_task_bits": None,  # None 时取 D_values 平均值
        "phi": 800.0,           # cycle/bit
        "tau_max": 0.2,

        # ===== 终端侧资源 =====
        "kappa": 1e-27,
        "f_L_min": 0.01,        # GHz
        "f_L_max": 1.0,         # GHz
        "p_R_min": 0.01,        # W
        "p_R_max": 2.0,         # W
        "B_max": 16.0,          # J
        "battery_drop_threshold": 1e-10,

        # ===== 无线链路 =====
        "B": 2.5e6,             # Hz
        "sigma2": 1e-7,         # 噪声功率
        "channel_distance_values": [60.0, 65.0, 70.0, 75.0, 80.0, 85.0],
        "harvest_distance_values": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],

        # ===== 边缘服务器 =====
        "server_freqs": 32e9,   # Hz
        "R_C": 3e8,             # 服务器间有线速率 bps
        "bg_task_count_min": 0,
        "bg_task_count_max": 8,

        # ===== 无线供能 =====
        "eta": 0.8,
        "P_RF": 10.0,
        "path_loss_exp": 2.4,
        "G": 4.11,

        # ===== 奖励与动态机制 =====
        "w0": 0.1,
        "reward_scale": 1.5,
        "infeasible_penalty": 4.0,    # 实际使用时以 -k_inv 进入奖励
        "dead_step_penalty_ratio": 0.50,  # 已掉电设备每步持续惩罚系数：k_dead_step = ratio * k_inv
        "delay_penalty_scale": 1.8,
        "energy_penalty_scale": 1.0,

        # ===== 拓扑 =====
        "server_topology": None,  # None 时使用 10 节点母图前缀子图

        # ===== 距离马尔可夫链 =====
        "distance_transition_self": 0.2,
        "distance_transition_neighbor": 0.4,

        # ===== 输出控制 =====
        "debug_info": False,  # True 时在 info 中返回详细 per-device 诊断信息
    }


class CooperativeWPMECChapter4FinalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        base_cfg = get_default_chapter4_final_config()
        if config:
            base_cfg.update(config)
        self.config = base_cfg

        # ===== 基础参数 =====
        self.N = int(self.config["num_devices"])
        self.M = int(self.config["num_servers"])
        self.T_f = float(self.config["T_f"])
        self.max_steps = int(self.config["max_steps"])

        self.D_values = np.array(self.config["D_values"], dtype=np.float64)
        avg_task_bits_cfg = self.config.get("avg_task_bits")
        self.avg_task_bits = float(np.mean(self.D_values) if avg_task_bits_cfg is None else avg_task_bits_cfg)
        self.phi = float(self.config["phi"])
        self.tau_max = float(self.config["tau_max"])

        self.kappa = float(self.config["kappa"])
        self.f_min = float(self.config["f_L_min"])  # GHz
        self.f_max = float(self.config["f_L_max"])  # GHz
        self.p_min = float(self.config["p_R_min"])
        self.p_max = float(self.config["p_R_max"])
        self.B_max = float(self.config["B_max"])
        self.battery_drop_threshold = float(self.config["battery_drop_threshold"])


        self.B_bandwidth = float(self.config["B"])
        self.sigma2 = float(self.config["sigma2"])
        self.channel_distance_values = np.array(self.config["channel_distance_values"], dtype=np.float64)
        self.harvest_distance_values = np.array(self.config["harvest_distance_values"], dtype=np.float64)

        self.server_freqs = float(self.config["server_freqs"])
        self.R_C = float(self.config["R_C"])

        self.bg_task_count_min = int(self.config["bg_task_count_min"])
        self.bg_task_count_max = int(self.config["bg_task_count_max"])


        self.eta = float(self.config["eta"])
        self.P_RF = float(self.config["P_RF"])
        self.rho_path_loss = float(self.config["path_loss_exp"])
        self.G = float(self.config["G"])

        self.w0 = float(self.config["w0"])
        self.reward_scale = float(self.config["reward_scale"])
        self.k_inv = float(self.config["infeasible_penalty"])
        self.dead_step_penalty_ratio = float(self.config.get("dead_step_penalty_ratio", 0.0))
        self.k_dead_step = max(0.0, self.dead_step_penalty_ratio) * self.k_inv
        self.k_delay = float(self.config["delay_penalty_scale"])
        self.k_energy = float(self.config["energy_penalty_scale"])


        self.debug_info = bool(self.config["debug_info"])

        # ===== 拓扑与区域绑定 =====
        self.topology = self._build_topology(self.config.get("server_topology"))
        self.hop_cache = self._precompute_hops(self.topology)
        self.region_assignment = self._build_region_assignment()

        # ===== 距离马尔可夫链 =====
        self.transition_matrix = self._build_markov_transition_matrix()
        self.channel_dist_idx_map = {float(d): i for i, d in enumerate(self.channel_distance_values)}
        self.harvest_dist_idx_map = {float(d): i for i, d in enumerate(self.harvest_distance_values)}

        # ===== 背景基础等待 / 等效队列归一化尺度 =====
        # 当前版本中，server_queues 表示“由背景任务数通过 M/M/1 基础等待时间
        # 换算得到的等效排队比特”，而非真实跨时隙累计队列。
        self.mu_bg = self.server_freqs / max(self.avg_task_bits * self.phi, 1e-12)  # tasks / s
        bg_wait_cap = self._mm1_bg_wait_from_task_count(self.bg_task_count_max)
        self.queue_norm_cap = max(bg_wait_cap * self.server_freqs / max(self.phi, 1e-12), 1.0)
        self.queue_delta_norm_cap = self.queue_norm_cap
        self.bg_norm_cap = max(float(self.bg_task_count_max), 1.0)
        self.g_min, self.g_max = self._calculate_channel_gain_range()
        self.max_hop = max(int(np.max(self.hop_cache)), 1)
        self.hop_matrix_norm = self._build_normalized_hop_matrix()

        # ===== 运行时状态 =====
        self.current_step = 0
        self.device_alive = np.ones(self.N, dtype=bool)
        self.device_battery = np.full(self.N, self.B_max, dtype=np.float64)
        self.device_task_bits = np.zeros(self.N, dtype=np.float64)
        self.device_tx_distance = np.zeros(self.N, dtype=np.float64)
        self.device_harvest_distance = np.zeros(self.N, dtype=np.float64)
        self.device_channel_gain = np.zeros(self.N, dtype=np.float64)

        self.server_queues = np.zeros(self.M, dtype=np.float64)       # 背景基础等待对应的等效排队比特
        self.server_queue_deltas = np.zeros(self.M, dtype=np.float64)
        self.server_bg_task_counts = np.zeros(self.M, dtype=np.int64)  # 当前时隙背景任务数
        self.region_alive_users = np.zeros(self.M, dtype=np.int64)     # 当前时隙开始时各区域存活设备数

        # ===== 对外观测 =====
        self.obs_dim = 7
        self.server_feat_dim = 4

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0.0, high=1.0, shape=(self.N, self.obs_dim), dtype=np.float32),
            "region_id": spaces.MultiDiscrete(np.full(self.N, self.M, dtype=np.int64)),
            "server_feat": spaces.Box(low=0.0, high=1.0, shape=(self.M, self.server_feat_dim), dtype=np.float32),
            "hop_matrix": spaces.Box(low=0.0, high=1.0, shape=(self.M, self.M), dtype=np.float32),
            "alive_mask": spaces.MultiBinary(self.N),
        })
        cont_low = np.tile(
            np.array([
                self.config["xi_low"] if "xi_low" in self.config else 0.0,
                self.f_min,
                self.p_min,
            ], dtype=np.float32),
            (self.N, 1),
        )
        cont_high = np.tile(
            np.array([1.0, self.f_max, self.p_max], dtype=np.float32),
            (self.N, 1),
        )

        self.action_space = spaces.Dict({
            "continuous": spaces.Box(
                low=cont_low,
                high=cont_high,
                dtype=np.float32,
            ),
            "discrete": spaces.MultiDiscrete(np.full(self.N, self.M, dtype=np.int64)),
        })

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self.current_step = 0
        self.device_alive[:] = True

        self.device_battery[:] = self.B_max

        self.device_task_bits = self.np_random.choice(self.D_values, size=self.N)
        self.device_tx_distance = self.np_random.choice(self.channel_distance_values, size=self.N)
        self.device_harvest_distance = self.np_random.choice(self.harvest_distance_values, size=self.N)
        self.device_channel_gain = np.array(
            [self._calculate_channel_gain(d) for d in self.device_tx_distance],
            dtype=np.float64,
        )

        # 每个 episode 开始时，仅生成“背景任务数快照”。
        # 再通过 M/M/1 基础等待时间换算为当前时隙开始时的等效排队比特。
        self.server_queue_deltas[:] = 0.0

        self.server_bg_task_counts, self.server_queues = self._sample_bg_snapshot()

        self.region_alive_users = self._count_alive_devices_per_region()

        state = self._build_state()
        info = self._build_summary_info(
            team_reward=0.0,
            reward_train=0.0,
            alive_prev=self.device_alive.copy(),
            alive_now=self.device_alive.copy(),
            timeout_count=0,
            infeasible_count=0,
            dead_event_count=0,
            avg_delay=0.0,
            avg_energy=0.0,
            success_count=0,
            success_delay_sum=0.0,
            success_energy_sum=0.0,
            executed_count=0,
            executed_delay_sum=0.0,
            executed_energy_sum=0.0,
            task_dropped_count=0,
            avg_weight=0.0,
            valid_count=0,
            xi_invalid_count=0,
            energy_violation_count=0,
            server_user_load_bits_step=np.zeros(self.M, dtype=np.float64),
            server_bg_equiv_bits_step=np.zeros(self.M, dtype=np.float64),
        )
        return state, info

    def step(self, action):
        if isinstance(action, dict):
            cont_action = action["continuous"]
            relay_target = action.get("relay_target", action.get("discrete"))
        else:
            raise ValueError("action 必须是包含 'continuous' 与 'relay_target'（或兼容字段 'discrete'）的 dict；relay_target 语义为区域服务器选择的二次转发目标")

        cont_action = np.asarray(cont_action, dtype=np.float64)
        relay_target = np.asarray(relay_target, dtype=np.int64)

        if cont_action.shape != (self.N, 3):
            raise ValueError(f"continuous 动作形状必须为 {(self.N, 3)}，当前为 {cont_action.shape}")
        if relay_target.shape != (self.N,):
            raise ValueError(f"relay_target 动作形状必须为 {(self.N,)}, 当前为 {relay_target.shape}")

        self.current_step += 1

        # ===== 当前时隙开始时的状态快照 =====
        alive_prev = self.device_alive.copy()
        battery_prev = self.device_battery.copy()
        queue_prev = self.server_queues.copy()  # 仅用于状态快照 / queue_delta 更新，不直接参与 T_W 主计算
        bg_task_count_prev = self.server_bg_task_counts.copy()
        bg_wait_prev = np.array(
            [self._mm1_bg_wait_from_task_count(c) for c in bg_task_count_prev],
            dtype=np.float64,
        )
        region_alive_prev = self.region_alive_users.copy()

        # ===== 预处理动作 =====
        mapped_actions = self._clip_continuous_action(cont_action)
        relay_target = np.clip(relay_target, 0, self.M - 1)

        # ===== 统计本时隙真正执行上传的 FDMA 用户数（按区域） =====
        scheduled_fdma_users = np.zeros(self.M, dtype=np.int64)
        for n in range(self.N):
            if not alive_prev[n]:
                continue
            xi_n = mapped_actions[n, 0]
            region_server = int(self.region_assignment[n])
            target_server = int(relay_target[n])
            hops = int(self.hop_cache[region_server, target_server])
            if xi_n > 1e-12 and hops >= 0:
                scheduled_fdma_users[region_server] += 1

        # ===== 按目标服务器构造 FCFS 顺序，并预计算与排队无关的物理量 =====
        fcfs_order, phys = self._build_fcfs_order(
            mapped_actions=mapped_actions,
            relay_target=relay_target,
            scheduled_fdma_users=scheduled_fdma_users,
        )

        # ===== 本时隙统计量 =====
        per_device_rewards = np.zeros(self.N, dtype=np.float64)
        per_device_delay = np.zeros(self.N, dtype=np.float64)
        per_device_energy = np.zeros(self.N, dtype=np.float64)

        timeout_count = 0
        infeasible_count = 0
        dead_event_count = 0

        # 统计量：
        # 1) success_*：仅对“成功完成任务（可执行、未掉电、未超时）”累计
        # 2) executed_*：对“未被丢弃、真正执行了动作”的样本累计（与第三章 avg_delay / avg_energy 口径一致）
        # 3) task_dropped_count：任务被丢弃/失败的计数，需与 dead_event_count 分开
        success_count = 0
        success_delay_sum = 0.0
        success_energy_sum = 0.0

        executed_count = 0
        executed_delay_sum = 0.0
        executed_energy_sum = 0.0

        task_dropped_count = 0
        xi_invalid_count = 0
        energy_violation_count = 0
        weight_sum = 0.0
        weight_count = 0

        arrivals_to_server = np.zeros(self.M, dtype=np.float64)
        bg_equiv_bits_step = bg_task_count_prev.astype(np.float64) * self.avg_task_bits

        # ===== 已经掉电的设备：奖励记 0，不再参与任何过程 =====
        for n in range(self.N):
            if not alive_prev[n]:
                # 持续小惩罚：抑制“早死止损”策略，避免后续全 0 奖励导致行为偏激
                per_device_rewards[n] = -self.k_dead_step

        # ===== 按服务器 FCFS 顺序处理活着的设备 =====
        for server_idx in range(self.M):
            cumulative_arrivals_bits = 0.0

            for n in fcfs_order[server_idx]:
                # 已经掉电设备在前面已处理为持续惩罚
                if not alive_prev[n]:
                    per_device_rewards[n] = -self.k_dead_step
                    continue

                battery = float(battery_prev[n])
                b_norm = battery / self.B_max
                D_t = float(self.device_task_bits[n])
                hops = int(phys["hops"][n])
                xi_t = float(mapped_actions[n, 0])

                T_L = float(phys["local_delay"][n])
                E_L = float(phys["local_energy"][n])
                T_R = float(phys["tx_delay"][n])
                E_R = float(phys["tx_energy"][n])
                T_FWD = float(phys["forward_delay"][n])
                T_C = float(phys["exec_delay"][n])
                fdma_users = int(phys["region_fdma_users"][n])

                # 当前等待时间显式分两部分计算：
                # 1) 背景基础等待：由背景任务数通过 M/M/1 给出
                # 2) 用户排序等待：由同时隙先到用户任务累计量给出
                T_W_bg = float(bg_wait_prev[server_idx])
                T_W_usr = self._queue_wait_delay(cumulative_arrivals_bits, self.server_freqs)
                T_W = T_W_bg + T_W_usr

                remote_delay = T_R + T_FWD + T_W + T_C
                T_total = np.inf if (not np.isfinite(T_R) or not np.isfinite(T_W)) else max(T_L, remote_delay)
                E_total = np.inf if not np.isfinite(E_R) else (E_L + E_R)

                per_device_delay[n] = T_total
                per_device_energy[n] = E_total

                E_H = self._harvest_energy(self.device_harvest_distance[n])

                # 目标服务器不可达，或基础物理量异常
                if hops < 0:
                    infeasible_count += 1
                    task_dropped_count += 1
                    per_device_rewards[n] = -self.k_inv
                    # 动作不执行，仅收割能量
                    self.device_battery[n] = min(self.B_max, battery + E_H)
                    continue

                # 动态能耗约束
                E_res, E_max, E_min, T_min, xi_range_valid = self._calculate_dynamic_energy_constraint(
                    D_t=D_t,
                    g_t=float(self.device_channel_gain[n]),
                    T_W=T_W,
                    hops=hops,
                    f_S=self.server_freqs,
                    b_norm=b_norm,
                    fdma_users=fdma_users,
                )

                # 电池处理修正：
                # - 结构性不可行（xi_range 无效 / T_total 非有限） -> 任务丢弃，仅收能
                # - 能耗侧不可行（E_total 非有限 / E_total > battery） -> 立即掉电
                structure_feasible = (
                    xi_range_valid
                    and np.isfinite(T_total)
                )
                energy_exhausted_now = (
                    (not np.isfinite(E_total))
                    or (E_total > battery)
                )

                if not structure_feasible:
                    infeasible_count += 1
                    if not xi_range_valid:
                        xi_invalid_count += 1
                    task_dropped_count += 1
                    per_device_rewards[n] = -self.k_inv
                    self.device_battery[n] = min(self.B_max, battery + E_H)
                    continue

                if energy_exhausted_now:
                    dead_event_count += 1
                    infeasible_count += 1
                    task_dropped_count += 1
                    per_device_rewards[n] = -self.k_inv
                    self.device_battery[n] = 0.0
                    self.device_alive[n] = False
                    # 视为执行过程中立即掉电，任务失败，不计入到达队列
                    continue

                # 动作执行后的暂态电量
                battery_next = battery - E_total + E_H

                # 若当前动作导致执行后掉电：按事件惩罚一次，后续时隙记 0
                if battery_next <= self.battery_drop_threshold:
                    dead_event_count += 1
                    infeasible_count += 1
                    task_dropped_count += 1
                    per_device_rewards[n] = -self.k_inv
                    self.device_battery[n] = 0.0
                    self.device_alive[n] = False
                    # 动作视为失败，不计入到达队列
                    continue

                # 动作成功执行：更新电量
                self.device_battery[n] = np.clip(battery_next, 0.0, self.B_max)

                # 该动作已成功执行（即使后续超时，也应计入第三章口径下的 avg_delay / avg_energy）
                executed_count += 1
                executed_delay_sum += float(T_total)
                executed_energy_sum += float(E_total)

                # 记录动态权重（按 executed 样本统计，和第三章 valid 样本口径一致）
                w_n = self._compute_dynamic_weight(b_norm)
                weight_sum += float(w_n)
                weight_count += 1

                # 能耗约束违反次数（对 executed 样本统计）
                if E_total > E_res:
                    energy_violation_count += 1

                # 若存在卸载任务，则计入同时隙先到用户任务累计量
                arrival_bits = xi_t * D_t
                arrivals_to_server[server_idx] += arrival_bits
                cumulative_arrivals_bits += arrival_bits

                # 超时奖励
                if T_total > self.tau_max:
                    timeout_count += 1
                    violation = min(1.0, (T_total - self.tau_max) / max(self.tau_max, 1e-12))
                    per_device_rewards[n] = -self.k_delay * violation
                    continue

                # 可行且未超时：计算效用奖励
                delay_score = np.clip((self.tau_max - T_total) / max(self.tau_max - T_min, 1e-12), 0.0, 1.0)
                energy_score = np.clip((E_max - E_total) / max(E_max - E_min, 1e-12), 0.0, 1.0)

                penalty_e = 0.0
                if E_total > E_res:
                    violation_e = min(1.0, (E_total - E_res) / max(E_res, 1e-12))
                    penalty_e = -self.k_energy * violation_e

                per_device_rewards[n] = self.reward_scale * (
                    w_n * delay_score + (1.0 - w_n) * energy_score
                ) + penalty_e

                # 成功任务：可执行、未掉电、且未超时
                success_count += 1
                success_delay_sum += float(T_total)
                success_energy_sum += float(E_total)

        # ===== 背景基础负载快照更新 =====
        # 用户任务不跨时隙形成真实队列；下一时隙的基础负载由新的背景任务数快照决定。
        next_bg_task_counts, next_server_queues = self._sample_bg_snapshot()

        self.server_queue_deltas = next_server_queues - queue_prev
        self.server_bg_task_counts = next_bg_task_counts
        self.server_queues = next_server_queues

        # ===== 生成下一时隙的外生随机量 =====
        for n in range(self.N):
            if self.device_alive[n]:
                self.device_task_bits[n] = float(self.np_random.choice(self.D_values))
                self.device_tx_distance[n] = self._update_distance_markov(
                    self.device_tx_distance[n],
                    self.channel_distance_values,
                    self.channel_dist_idx_map,
                )
                self.device_harvest_distance[n] = self._update_distance_markov(
                    self.device_harvest_distance[n],
                    self.harvest_distance_values,
                    self.harvest_dist_idx_map,
                )
                self.device_channel_gain[n] = self._calculate_channel_gain(self.device_tx_distance[n])
            else:
                self.device_task_bits[n] = 0.0
                self.device_channel_gain[n] = 0.0

        # ===== 更新区域存活设备统计 =====
        self.region_alive_users = self._count_alive_devices_per_region()

        # ===== 终止条件 =====
        terminated = bool(np.all(~self.device_alive))
        truncated = bool(self.current_step >= self.max_steps)

        # ===== 奖励 =====
        dead_count_total = int(self.N - np.sum(self.device_alive))

        team_reward = float(np.sum(per_device_rewards))
        reward_train = team_reward / max(self.N, 1)

        # ===== 状态与信息 =====
        state = self._build_state()

        alive_now = self.device_alive.copy()
        alive_count = int(np.sum(alive_prev))
        if alive_count > 0:
            avg_delay = float(np.sum(per_device_delay[alive_prev]) / alive_count)
            avg_energy = float(np.sum(per_device_energy[alive_prev]) / alive_count)
        else:
            avg_delay = 0.0
            avg_energy = 0.0

        info = self._build_summary_info(
            team_reward=team_reward,
            reward_train=reward_train,
            alive_prev=alive_prev,
            alive_now=alive_now,
            timeout_count=timeout_count,
            infeasible_count=infeasible_count,
            dead_event_count=dead_event_count,
            avg_delay=avg_delay,
            avg_energy=avg_energy,
            success_count=success_count,
            success_delay_sum=success_delay_sum,
            success_energy_sum=success_energy_sum,
            executed_count=executed_count,
            executed_delay_sum=executed_delay_sum,
            executed_energy_sum=executed_energy_sum,
            task_dropped_count=task_dropped_count,
            avg_weight=(weight_sum / max(weight_count, 1)),
            valid_count=executed_count,
            xi_invalid_count=xi_invalid_count,
            energy_violation_count=energy_violation_count,
            server_user_load_bits_step=arrivals_to_server,
            server_bg_equiv_bits_step=bg_equiv_bits_step,
        )

        if self.debug_info:
            info["debug"] = {
                "per_device_rewards": per_device_rewards.astype(np.float32),
                "per_device_delay": np.where(np.isfinite(per_device_delay), per_device_delay, 0.0).astype(np.float32),
                "per_device_energy": np.where(np.isfinite(per_device_energy), per_device_energy, 0.0).astype(np.float32),
                "scheduled_fdma_users": scheduled_fdma_users.astype(np.int32),
                "queue_prev": queue_prev.astype(np.float32),
                "queue_now": self.server_queues.astype(np.float32),
                "bg_task_count_prev": bg_task_count_prev.astype(np.int32),
                "region_alive_prev": region_alive_prev.astype(np.int32),
                "mapped_actions": mapped_actions.astype(np.float32),
                "relay_target": relay_target.astype(np.int32),
                "arrivals_to_server": arrivals_to_server.astype(np.float32),
                "bg_equiv_bits_step": bg_equiv_bits_step.astype(np.float32),
                "avg_weight_step": np.float32(weight_sum / max(weight_count, 1)),
            }

        return state, reward_train, terminated, truncated, info

    # ------------------------------------------------------------------
    # 状态构造
    # ------------------------------------------------------------------
    def _build_state(self) -> Dict[str, np.ndarray]:
        queue_norm = np.array([self._normalize_queue(q) for q in self.server_queues], dtype=np.float32)
        queue_delta_norm = np.array([self._normalize_queue_delta(dq) for dq in self.server_queue_deltas], dtype=np.float32)
        bg_norm = np.array([self._normalize_bg_task_count(c) for c in self.server_bg_task_counts], dtype=np.float32)
        region_users_norm = np.array(
            [self._normalize_region_users(u) for u in self.region_alive_users],
            dtype=np.float32,
        )

        obs = np.zeros((self.N, self.obs_dim), dtype=np.float32)
        for n in range(self.N):
            if not self.device_alive[n]:
                continue
            region_server = int(self.region_assignment[n])
            obs[n] = np.array([
                self.device_battery[n] / self.B_max,
                self._normalize_task(self.device_task_bits[n]),
                self._normalize_channel(self.device_channel_gain[n]),
                self._normalize_distance(self.device_harvest_distance[n], self.harvest_distance_values),
                queue_norm[region_server],
                queue_delta_norm[region_server],
                region_users_norm[region_server],
            ], dtype=np.float32)

        server_feat = np.stack(
            [queue_norm, queue_delta_norm, bg_norm, region_users_norm],
            axis=-1,
        ).astype(np.float32)

        return {
            "obs": obs,
            "region_id": self.region_assignment.astype(np.int64).copy(),
            "server_feat": server_feat,
            "hop_matrix": self.hop_matrix_norm.copy(),
            "alive_mask": self.device_alive.astype(np.float32),
        }

    def _build_normalized_hop_matrix(self) -> np.ndarray:
        hop = self.hop_cache.astype(np.float32).copy()
        if self.max_hop > 0:
            hop = hop / float(self.max_hop)
        hop = np.clip(hop, 0.0, 1.0)
        return hop.astype(np.float32)


    def _sample_bg_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样下一时隙的背景任务数快照，并同步换算为等效背景排队比特。
        返回:
            bg_task_counts: [M]
            bg_queue_equiv_bits: [M]
        """
        bg_task_counts = self._sample_task_counts(
            self.bg_task_count_min,
            self.bg_task_count_max,
            size=self.M,
        )
        bg_queue_equiv_bits = self._bg_task_counts_to_equiv_queue_bits(bg_task_counts)
        return bg_task_counts, bg_queue_equiv_bits

    def _build_summary_info(
        self,
        team_reward: float,
        reward_train: float,
        alive_prev: np.ndarray,
        alive_now: np.ndarray,
        timeout_count: int,
        infeasible_count: int,
        dead_event_count: int,
        avg_delay: float,
        avg_energy: float,
        success_count: int,
        success_delay_sum: float,
        success_energy_sum: float,
        executed_count: int,
        executed_delay_sum: float,
        executed_energy_sum: float,
        task_dropped_count: int,
        avg_weight: float,
        valid_count: int,
        xi_invalid_count: int,
        energy_violation_count: int,
        server_user_load_bits_step: np.ndarray,
        server_bg_equiv_bits_step: np.ndarray,
    ) -> Dict:
        success_avg_delay = float(success_delay_sum / max(success_count, 1))
        success_avg_energy = float(success_energy_sum / max(success_count, 1))
        executed_avg_delay = float(executed_delay_sum / max(executed_count, 1))
        executed_avg_energy = float(executed_energy_sum / max(executed_count, 1))

        return {
            "step": int(self.current_step),
            "global_reward": float(team_reward),
            "train_reward": float(reward_train),
            "alive_count_prev": int(np.sum(alive_prev)),
            "alive_count_now": int(np.sum(alive_now)),
            "dead_count_total": int(self.N - np.sum(alive_now)),
            "dead_event_count": int(dead_event_count),
            "timeout_count": int(timeout_count),
            "infeasible_count": int(infeasible_count),
            "task_dropped_count": int(task_dropped_count),
            "valid_count": int(valid_count),
            "avg_weight": float(avg_weight),
            "xi_invalid_count": int(xi_invalid_count),
            "energy_violation_count": int(energy_violation_count),
            "battery_exhausted_count": int(dead_event_count),

            # 保持兼容：仍保留“对 alive_prev 的平均”
            "avg_delay_alive_prev": float(avg_delay),
            "avg_energy_alive_prev": float(avg_energy),

            # 成功样本统计（用于 success rate 等）
            "success_count": int(success_count),
            "success_delay_sum": float(success_delay_sum),
            "success_energy_sum": float(success_energy_sum),
            "success_avg_delay": float(success_avg_delay),
            "success_avg_energy": float(success_avg_energy),

            # 与第三章训练日志保持一致：对未被丢弃/真正执行的样本统计平均时延和平均能耗
            "executed_count": int(executed_count),
            "executed_delay_sum": float(executed_delay_sum),
            "executed_energy_sum": float(executed_energy_sum),
            "executed_avg_delay": float(executed_avg_delay),
            "executed_avg_energy": float(executed_avg_energy),
            "server_user_load_bits_step": np.asarray(server_user_load_bits_step, dtype=np.float32).copy(),
            "server_bg_equiv_bits_step": np.asarray(server_bg_equiv_bits_step, dtype=np.float32).copy(),
            "server_bg_load_bits_step": np.asarray(server_bg_equiv_bits_step, dtype=np.float32).copy(),
            "server_total_load_bits_step": np.asarray(server_user_load_bits_step + server_bg_equiv_bits_step, dtype=np.float32).copy(),

            "device_battery": self.device_battery.astype(np.float32).copy(),
            "server_queues": self.server_queues.astype(np.float32).copy(),
            "server_queue_deltas": self.server_queue_deltas.astype(np.float32).copy(),
            "server_bg_task_counts": self.server_bg_task_counts.astype(np.int32).copy(),
        }

    # ------------------------------------------------------------------
    # 动作与排序
    # ------------------------------------------------------------------
    def _clip_continuous_action(self, cont_action: np.ndarray) -> np.ndarray:
        mapped = np.zeros((self.N, 3), dtype=np.float64)
        mapped[:, 0] = np.clip(cont_action[:, 0], 0.0, 1.0)
        mapped[:, 1] = np.clip(cont_action[:, 1], self.f_min, self.f_max)  # GHz
        mapped[:, 2] = np.clip(cont_action[:, 2], self.p_min, self.p_max)
        return mapped

    def _build_fcfs_order(
        self,
        mapped_actions: np.ndarray,
        relay_target: np.ndarray,
        scheduled_fdma_users: np.ndarray,
    ) -> Tuple[List[List[int]], Dict]:
        phys = {
            "hops": np.full(self.N, -1, dtype=np.int64),
            "local_delay": np.zeros(self.N, dtype=np.float64),
            "local_energy": np.zeros(self.N, dtype=np.float64),
            "tx_delay": np.zeros(self.N, dtype=np.float64),
            "tx_energy": np.zeros(self.N, dtype=np.float64),
            "forward_delay": np.zeros(self.N, dtype=np.float64),
            "exec_delay": np.zeros(self.N, dtype=np.float64),
            "region_fdma_users": np.zeros(self.N, dtype=np.int64),
        }

        server_candidates: List[List[Tuple[float, int]]] = [[] for _ in range(self.M)]

        for n in range(self.N):
            if not self.device_alive[n]:
                continue

            D_t = float(self.device_task_bits[n])
            g_t = float(self.device_channel_gain[n])
            region_server = int(self.region_assignment[n])
            target_server = int(np.clip(relay_target[n], 0, self.M - 1))
            hops = int(self.hop_cache[region_server, target_server])

            xi_t, f_L_t_GHz, p_R_t = mapped_actions[n]
            f_L_t = f_L_t_GHz * 1e9  # 转换为 Hz
            fdma_users = max(1, int(scheduled_fdma_users[region_server]))

            T_L, E_L = self._local_compute(D_t=D_t, xi=xi_t, f_L=f_L_t)
            T_R, E_R, _ = self._uplink_transmission(D_t=D_t, xi=xi_t, p_R=p_R_t, g_t=g_t, fdma_users=fdma_users)
            T_FWD = self._forward_delay(D_t=D_t, xi=xi_t, hops=hops)
            T_C = self._server_exec_delay(D_t=D_t, xi=xi_t, f_S=self.server_freqs)

            arrival_time = T_R + T_FWD if (np.isfinite(T_R) and np.isfinite(T_FWD)) else np.inf

            phys["hops"][n] = hops
            phys["local_delay"][n] = T_L
            phys["local_energy"][n] = E_L
            phys["tx_delay"][n] = T_R
            phys["tx_energy"][n] = E_R
            phys["forward_delay"][n] = T_FWD
            phys["exec_delay"][n] = T_C
            phys["region_fdma_users"][n] = fdma_users

            if hops >= 0:
                server_candidates[target_server].append((arrival_time, n))

        fcfs_order = [
            [n for _, n in sorted(candidates, key=lambda x: (x[0], x[1]))]
            for candidates in server_candidates
        ]
        return fcfs_order, phys

    # ------------------------------------------------------------------
    # 底层物理模型
    # ------------------------------------------------------------------
    def _local_compute(self, D_t: float, xi: float, f_L: float) -> Tuple[float, float]:
        if xi >= 1.0 or D_t <= 0.0:
            return 0.0, 0.0
        T_L = (1.0 - xi) * D_t * self.phi / f_L
        E_L = self.kappa * (f_L ** 3) * T_L
        return float(T_L), float(E_L)

    def _uplink_transmission(
        self,
        D_t: float,
        xi: float,
        p_R: float,
        g_t: float,
        fdma_users: int,
    ) -> Tuple[float, float, float]:
        if xi <= 0.0 or D_t <= 0.0 or g_t <= 0.0:
            return 0.0, 0.0, 0.0

        fdma_users = max(int(fdma_users), 1)
        per_user_bandwidth = self.B_bandwidth / fdma_users
        rate = per_user_bandwidth * np.log2(1.0 + p_R * g_t)
        if rate <= 0.0 or not np.isfinite(rate):
            return np.inf, np.inf, 0.0

        T_R = xi * D_t / rate
        E_R = p_R * T_R
        return float(T_R), float(E_R), float(rate)

    def _forward_delay(self, D_t: float, xi: float, hops: int) -> float:
        if xi <= 0.0 or D_t <= 0.0 or hops <= 0:
            return 0.0
        return float(xi * D_t * hops / self.R_C)

    def _queue_wait_delay(self, queue_bits: float, f_S: float) -> float:
        if queue_bits <= 0.0:
            return 0.0
        return float(queue_bits * self.phi / f_S)

    def _mm1_bg_wait_from_task_count(self, bg_task_count: float) -> float:
        """
        基于背景任务数计算 M/M/1 平均基础等待时间。
        令 λ_bg = N_bg / T_f，μ_bg = f_e / (d * phi)。
        """
        count = float(max(bg_task_count, 0.0))
        if count <= 0.0:
            return 0.0

        lambda_bg = count / max(self.T_f, 1e-12)  # tasks / s
        mu_bg = max(self.mu_bg, 1e-12)

        if lambda_bg >= mu_bg:
            return float(self.tau_max)

        rho_bg = lambda_bg / mu_bg
        wait = rho_bg / max(mu_bg - lambda_bg, 1e-12)
        return float(max(wait, 0.0))

    def _bg_task_counts_to_equiv_queue_bits(self, bg_task_counts: np.ndarray) -> np.ndarray:
        """
        将背景任务数对应的 M/M/1 基础等待时间换算为等效排队比特：
            q_eq * phi / f_e = T_bg
            => q_eq = T_bg * f_e / phi
        """
        bg_task_counts = np.asarray(bg_task_counts, dtype=np.float64)
        equiv = np.zeros_like(bg_task_counts, dtype=np.float64)
        for i, c in enumerate(bg_task_counts):
            t_bg = self._mm1_bg_wait_from_task_count(float(c))
            equiv[i] = t_bg * self.server_freqs / max(self.phi, 1e-12)
        return equiv

    def _server_exec_delay(self, D_t: float, xi: float, f_S: float) -> float:
        if xi <= 0.0 or D_t <= 0.0:
            return 0.0
        return float(xi * D_t * self.phi / f_S)

    def _harvest_energy(self, d_h: float) -> float:
        return float(self.eta * self.P_RF * (d_h ** (-self.rho_path_loss)) * self.G * self.T_f)

    # ------------------------------------------------------------------
    # 动态权重与动态能耗约束
    # ------------------------------------------------------------------
    def _compute_dynamic_weight(self, b_norm: float) -> float:
        b_norm = float(np.clip(b_norm, 0.0, 1.0))
        eta_w = 2.0 * np.log((1.0 - self.w0) / max(self.w0, 1e-12))
        numerator = self.w0 * np.exp(eta_w * b_norm)
        denominator = self.w0 * (np.exp(eta_w * b_norm) - 1.0) + 1.0
        return float(numerator / max(denominator, 1e-12))

    def _calculate_dynamic_energy_constraint(
        self,
        D_t: float,
        g_t: float,
        T_W: float,
        hops: int,
        f_S: float,
        b_norm: float,
        fdma_users: int,
    ) -> Tuple[float, float, float, float, bool]:
        eps = 1e-12
        f_max_b = self.f_max * 1e9
        fdma_users = max(int(fdma_users), 1)

        nu_max = (self.B_bandwidth / fdma_users) * np.log2(1.0 + self.p_max * g_t)
        if nu_max <= eps or not np.isfinite(nu_max):
            return 0.0, 0.0, 0.0, np.inf, False

        A = D_t * self.phi / f_max_b
        C = D_t / nu_max + D_t * hops / self.R_C + D_t * self.phi / f_S

        xi_min = max(0.0, 1.0 - (self.tau_max * f_max_b) / max(D_t * self.phi, eps))
        xi_max_val = (self.tau_max - T_W) / max(C, eps)
        xi_max = min(1.0, max(0.0, xi_max_val))
        if xi_min > xi_max:
            return 0.0, 0.0, 0.0, np.inf, False

        if not np.isfinite(T_W):
            T_min = np.inf
        elif T_W >= A:
            if xi_min <= 0.0 <= xi_max:
                T_min = A
            else:
                T_min = max((1.0 - xi_min) * A, T_W + xi_min * C)
        else:
            xi_bal = (A - T_W) / max(A + C, eps)
            if xi_min <= xi_bal <= xi_max:
                T_min = A * (C + T_W) / max(A + C, eps)
            else:
                T_min = min(
                    max((1.0 - xi_min) * A, T_W + xi_min * C),
                    max((1.0 - xi_max) * A, T_W + xi_max * C),
                )

        if not np.isfinite(T_min):
            return 0.0, 0.0, 0.0, np.inf, False

        E_max_L_bit = self.kappa * (f_max_b ** 2) * self.phi
        E_max_R_bit = self.p_max / nu_max
        E_max_xi_min = (1.0 - xi_min) * D_t * E_max_L_bit + xi_min * D_t * E_max_R_bit
        E_max_xi_max = (1.0 - xi_max) * D_t * E_max_L_bit + xi_max * D_t * E_max_R_bit
        E_max = max(E_max_xi_min, E_max_xi_max)

        E_min, _, _, xi_opt = self._solve_P5_min_energy(
            D_t=D_t,
            g_t=g_t,
            T_W=T_W,
            hops=hops,
            f_S=f_S,
            xi_min=xi_min,
            xi_max=xi_max,
            fdma_users=fdma_users,
        )
        if E_min is None or xi_opt is None:
            return 0.0, 0.0, 0.0, T_min, False

        if E_max > 2.0 * E_min:
            eta_e = 2.0 * np.log((E_max - E_min) / max(E_min, eps))
        else:
            eta_e = 2.0

        numerator = E_min * E_max * np.exp(eta_e * b_norm)
        denominator = E_min * (np.exp(eta_e * b_norm) - 1.0) + E_max
        E_res = numerator / max(denominator, eps)
        return float(E_res), float(E_max), float(E_min), float(T_min), True

    def _solve_P5_min_energy(
        self,
        D_t: float,
        g_t: float,
        T_W: float,
        hops: int,
        f_S: float,
        xi_min: float,
        xi_max: float,
        fdma_users: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        f_max_b = self.f_max * 1e9
        f_min_b = self.f_min * 1e9
        fdma_users = max(int(fdma_users), 1)
        per_user_bandwidth = self.B_bandwidth / fdma_users

        def energy_func(xi: float):
            xi_c = float(np.clip(xi, xi_min, xi_max))
            T_R = self.tau_max - T_W - xi_c * D_t * hops / self.R_C - xi_c * D_t * self.phi / f_S
            if T_R <= 1e-12:
                return np.inf, np.inf, np.inf

            f_required = ((1.0 - xi_c) * D_t * self.phi) / self.tau_max
            if f_required > f_max_b:
                return np.inf, np.inf, np.inf
            f_used = max(f_required, f_min_b)
            T_local = ((1.0 - xi_c) * D_t * self.phi) / f_used if f_used > 0 else np.inf
            E_L = self.kappa * (f_used ** 3) * T_local

            exponent = (xi_c * D_t) / max(per_user_bandwidth * T_R, 1e-12)
            if exponent > 100:
                return np.inf, np.inf, np.inf
            p_required = (2.0 ** exponent - 1.0) / max(g_t, 1e-12)
            if p_required > self.p_max:
                return np.inf, np.inf, np.inf
            p_used = max(p_required, self.p_min)
            E_R = p_used * T_R
            return E_L + E_R, f_used, p_used

        try:
            xi_opt, (E_min, f_re, p_re) = self._golden_section_search_with_inf(energy_func, xi_min, xi_max)
            if not np.isfinite(E_min):
                return None, None, None, None
            return float(E_min), float(f_re), float(p_re), float(xi_opt)
        except Exception:
            return None, None, None, None

    @staticmethod
    def _golden_section_search_with_inf(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        it = 0
        while it < max_iter and abs(b - a) > tol:
            it += 1
            fc_energy = fc[0]
            fd_energy = fd[0]

            if np.isinf(fc_energy) and np.isinf(fd_energy):
                a = (a + c) / 2.0
                b = (b + d) / 2.0
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = f(c)
                fd = f(d)
                continue
            if np.isinf(fc_energy):
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
                continue
            if np.isinf(fd_energy):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
                continue

            if fc_energy < fd_energy:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)

        x_opt = (a + b) / 2.0
        f_opt = f(x_opt)
        return x_opt, f_opt

    # ------------------------------------------------------------------
    # 拓扑与区域绑定
    # ------------------------------------------------------------------
    def _build_topology(self, topology: Optional[List[List[int]]]) -> np.ndarray:
        full_topology_10 = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ], dtype=np.int8)

        if topology is None:
            if not (1 <= self.M <= 10):
                raise ValueError("默认母图前缀规则仅支持 1~10 个服务器；如超出请显式给出 server_topology")
            mat = full_topology_10[:self.M, :self.M].copy()
        else:
            mat = np.asarray(topology, dtype=np.int8)
            if mat.shape != (self.M, self.M):
                raise ValueError(f"server_topology 形状应为 {(self.M, self.M)}")

        np.fill_diagonal(mat, 0)
        if not np.array_equal(mat, mat.T):
            raise ValueError("server_topology 必须是对称无向邻接矩阵")

        if self.M > 0:
            visited = {0}
            q = deque([0])
            while q:
                node = q.popleft()
                for nei in np.where(mat[node] > 0)[0]:
                    nei = int(nei)
                    if nei not in visited:
                        visited.add(nei)
                        q.append(nei)
            if len(visited) != self.M:
                raise ValueError("server_topology 必须是连通图")

        return mat

    def _precompute_hops(self, topology: np.ndarray) -> np.ndarray:
        hop_cache = np.full((self.M, self.M), -1, dtype=np.int64)
        for i in range(self.M):
            for j in range(self.M):
                hop_cache[i, j] = 0 if i == j else self._bfs_hops(topology, i, j)
        return hop_cache

    @staticmethod
    def _bfs_hops(topology: np.ndarray, start: int, end: int) -> int:
        q = deque([(start, 0)])
        visited = {start}
        while q:
            node, dist = q.popleft()
            for nei in np.where(topology[node] > 0)[0]:
                nei = int(nei)
                if nei == end:
                    return dist + 1
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, dist + 1))
        return -1

    def _build_region_assignment(self) -> np.ndarray:
        mode = self.config["region_assignment_mode"]
        if mode == "manual":
            arr = np.asarray(self.config["manual_region_assignment"], dtype=np.int64)
            if arr.shape != (self.N,):
                raise ValueError("manual_region_assignment 长度必须等于 num_devices")
            if np.any(arr < 0) or np.any(arr >= self.M):
                raise ValueError("manual_region_assignment 中存在非法服务器编号")
            return arr

        quotient = self.N // self.M
        remainder = self.N % self.M
        allocation = []
        for device_id in range(self.N):
            if quotient == 0:
                region = device_id
            elif device_id < remainder * (quotient + 1):
                region = device_id // (quotient + 1)
            else:
                adjusted = device_id - remainder * (quotient + 1)
                region = remainder + adjusted // quotient
            allocation.append(region)
        return np.array(allocation, dtype=np.int64)

    # ------------------------------------------------------------------
    # 距离马尔可夫链
    # ------------------------------------------------------------------
    def _build_markov_transition_matrix(self) -> np.ndarray:
        n_states = len(self.channel_distance_values)
        mat = np.zeros((n_states, n_states), dtype=np.float64)
        p_self = float(self.config["distance_transition_self"])
        p_nei = float(self.config["distance_transition_neighbor"])

        for i in range(n_states):
            if i == 0:
                mat[i, i] = 1.0 - p_nei
                mat[i, i + 1] = p_nei
            elif i == n_states - 1:
                mat[i, i] = 1.0 - p_nei
                mat[i, i - 1] = p_nei
            else:
                mat[i, i] = p_self
                mat[i, i - 1] = p_nei
                mat[i, i + 1] = p_nei

        mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-12, None)
        return mat

    def _update_distance_markov(
        self,
        current_distance: float,
        distance_values: np.ndarray,
        index_map: Dict[float, int],
    ) -> float:
        idx = index_map[float(current_distance)]
        next_idx = int(self.np_random.choice(len(distance_values), p=self.transition_matrix[idx]))
        return float(distance_values[next_idx])

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _sample_task_counts(self, min_count: int, max_count: int, size) -> np.ndarray:
        if max_count < min_count:
            raise ValueError("任务数上界不能小于下界")
        return self.np_random.integers(min_count, max_count + 1, size=size, dtype=np.int64)


    def _count_alive_devices_per_region(self) -> np.ndarray:
        counts = np.zeros(self.M, dtype=np.int64)
        for n in range(self.N):
            if self.device_alive[n]:
                counts[int(self.region_assignment[n])] += 1
        return counts

    def _calculate_channel_gain(self, distance: float) -> float:
        c = 3e8
        f_c = 915e6
        denominator = 4.0 * np.pi * f_c * distance
        h = 4.11 * (c / denominator) ** 2
        return float(h / self.sigma2)

    def _calculate_channel_gain_range(self) -> Tuple[float, float]:
        g_values = [self._calculate_channel_gain(d) for d in self.channel_distance_values]
        return 0.9 * min(g_values), 1.1 * max(g_values)

    def _normalize_channel(self, g: float) -> float:
        if g <= 0.0 or self.g_max <= self.g_min:
            return 0.0
        return float(np.clip((g - self.g_min) / (self.g_max - self.g_min), 0.0, 1.0))

    def _normalize_task(self, D_t: float) -> float:
        D_min = float(np.min(self.D_values))
        D_max = float(np.max(self.D_values))
        if D_max <= D_min:
            return 0.0
        return float(np.clip((D_t - D_min) / (D_max - D_min), 0.0, 1.0))

    def _normalize_queue(self, q: float) -> float:
        return float(np.clip(q / self.queue_norm_cap, 0.0, 1.0))

    def _normalize_queue_delta(self, dq: float) -> float:
        cap = max(self.queue_delta_norm_cap, 1e-12)
        dq = float(np.clip(dq, -cap, cap))
        return float((dq + cap) / (2.0 * cap))

    def _normalize_bg_task_count(self, bg_task_count: float) -> float:
        return float(np.clip(float(bg_task_count) / self.bg_norm_cap, 0.0, 1.0))

    def _normalize_region_users(self, users: int) -> float:
        return float(np.clip(float(users) / max(self.N, 1), 0.0, 1.0))

    @staticmethod
    def _normalize_distance(d: float, values: np.ndarray) -> float:
        d_min = float(np.min(values))
        d_max = float(np.max(values))
        if d_max <= d_min:
            return 0.0
        return float(np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0))

    def close(self):
        pass


def make_default_chapter4_final_env(config: Optional[Dict] = None) -> CooperativeWPMECChapter4FinalEnv:
    return CooperativeWPMECChapter4FinalEnv(config=config)


if __name__ == "__main__":
    env = CooperativeWPMECChapter4FinalEnv({
        "num_devices": 30,
        "num_servers": 6,
        "max_steps": 5,
        "debug_info": True,
    })
    state, info = env.reset(seed=42)
    print("reset:",
          state["obs"].shape,
          state["region_id"].shape,
          state["server_feat"].shape,
          state["alive_mask"].shape)

    cont_action = np.zeros((env.N, 3), dtype=np.float32)
    cont_action[:, 0] = 0.3
    cont_action[:, 1] = 0.5
    cont_action[:, 2] = 1.0
    relay_target = env.region_assignment.copy()

    next_state, reward, terminated, truncated, info = env.step({
        "continuous": cont_action,
        "relay_target": relay_target,
    })
    print("step:", reward, terminated, truncated, info["global_reward"])
