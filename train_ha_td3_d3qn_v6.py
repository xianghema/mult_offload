from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from ha_td3_d3qn_v5 import HATD3D3QNAgent, HATD3D3QNConfig
from wpmec_env_chapter4_v15 import (
    CooperativeWPMECChapter4FinalEnv,
    get_default_chapter4_final_config,
)


def moving_average(values: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        out.append(float(np.mean(values[left : i + 1])))
    return out


def save_reward_curve(
    episode_rewards: List[float],
    reward_ma_series: List[float],
    output_path: Path,
    ma_window: int,
) -> None:
    if len(episode_rewards) == 0 or plt is None:
        return

    x = np.arange(1, len(episode_rewards) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, episode_rewards, label="Episode Reward")
    plt.plot(x, reward_ma_series, label=f"Reward MA({ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Average Reward per Episode")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_eval_curve(
    eval_episode_indices: List[int],
    eval_rewards: List[float],
    eval_best_so_far: List[float],
    output_path: Path,
) -> None:
    if len(eval_episode_indices) == 0 or plt is None:
        return

    x = np.asarray(eval_episode_indices, dtype=np.int32)
    y = np.asarray(eval_rewards, dtype=np.float32)
    y_best = np.asarray(eval_best_so_far, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Deterministic Eval Reward")
    plt.plot(x, y_best, label="Best-So-Far Eval Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Average Reward per Episode")
    plt.title("Evaluation Convergence Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_env(args) -> CooperativeWPMECChapter4FinalEnv:
    env_cfg = get_default_chapter4_final_config()
    env_cfg.update(
        {
            "num_devices": args.num_devices,
            "num_servers": args.num_servers,
            "max_steps": args.max_steps,
            "B_max": args.battery_capacity,
            "infeasible_penalty": args.infeasible_penalty,
            "delay_penalty_scale": args.delay_penalty_scale,
            "energy_penalty_scale": args.energy_penalty_scale,
            "dead_step_penalty_ratio": args.dead_step_penalty_ratio,
            "debug_info": False,
        }
    )
    return CooperativeWPMECChapter4FinalEnv(env_cfg)


def make_agent(args) -> HATD3D3QNAgent:
    expected_updates = max((args.episodes - args.warmup_episodes) * args.max_steps, 1)
    cfg = HATD3D3QNConfig(
        num_agents=args.num_devices,
        num_servers=args.num_servers,
        obs_dim=7,
        server_feat_dim=4,
        num_regions=args.num_servers,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        replay_strategy=args.replay_strategy,
        recent_ratio=args.recent_ratio,
        recent_window=args.recent_window,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_beta_steps=expected_updates,
        per_eps=args.per_eps,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        explore_noise=args.explore_noise_start,
        explore_noise_start=args.explore_noise_start,
        explore_noise_min=args.explore_noise_min,
        explore_noise_decay=args.explore_noise_decay,
        explore_noise_decay_per_episode=True,
        explore_noise_decay_start_episode=args.explore_noise_decay_start_episode,
        policy_delay=args.policy_delay,
        grad_clip=args.grad_clip,
        relay_epsilon_start=args.relay_epsilon_start,
        relay_epsilon_min=args.relay_epsilon_min,
        relay_epsilon_decay=args.relay_epsilon_decay,
        relay_epsilon_decay_start_it=args.relay_epsilon_decay_start_it,
        relay_epsilon_decay_per_episode=True,
        relay_epsilon_decay_start_episode=args.relay_epsilon_decay_start_episode,
        use_orthogonal_init=True,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
    )
    return HATD3D3QNAgent(cfg)


def sample_random_actions(env):
    cont = np.zeros((env.N, 3), dtype=np.float32)
    cont[:, 0] = np.random.uniform(0.0, 1.0, size=env.N)
    cont[:, 1] = np.random.uniform(env.f_min, env.f_max, size=env.N)
    cont[:, 2] = np.random.uniform(env.p_min, env.p_max, size=env.N)
    relay_target = np.random.randint(0, env.M, size=(env.N,), dtype=np.int64)
    return cont, relay_target


def evaluate_policy(
    agent: HATD3D3QNAgent,
    args,
    eval_episodes: int,
    eval_seed_base: int,
) -> Tuple[float, float, float]:
    eval_env = make_env(args)
    eval_rewards: List[float] = []
    eval_success_rates: List[float] = []
    eval_timeout_counts: List[float] = []

    for i in range(eval_episodes):
        state, _ = eval_env.reset(seed=eval_seed_base + i)
        obs = state["obs"]
        region_id = state["region_id"]
        server_feat = state["server_feat"]
        hop_matrix = state["hop_matrix"]
        alive_mask = state["alive_mask"]

        ep_reward = 0.0
        success_total = 0
        timeout_total = 0

        for _ in range(args.max_steps):
            cont_action, relay_target = agent.select_relay_target(
                obs=obs,
                region_id=region_id,
                server_feat=server_feat,
                hop_matrix=hop_matrix,
                alive_mask=alive_mask,
                cont_action=None,
                deterministic_cont=True,
                deterministic_relay=True,
            )

            next_state, reward, terminated, truncated, info = eval_env.step(
                {"continuous": cont_action, "relay_target": relay_target}
            )
            ep_reward += float(reward)
            success_total += int(info.get("success_count", 0))
            timeout_total += int(info.get("timeout_count", 0))

            obs = next_state["obs"]
            region_id = next_state["region_id"]
            server_feat = next_state["server_feat"]
            hop_matrix = next_state["hop_matrix"]
            alive_mask = next_state["alive_mask"]

            if terminated or truncated:
                break

        total_slots = args.num_devices * args.max_steps
        eval_rewards.append(float(ep_reward))
        eval_success_rates.append(float(success_total / max(total_slots, 1)))
        eval_timeout_counts.append(float(timeout_total))

    return (
        float(np.mean(eval_rewards)),
        float(np.mean(eval_success_rates)),
        float(np.mean(eval_timeout_counts)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_episodes", type=int, default=50)
    parser.add_argument("--reward_ma_window", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="runs/ha_td3_d3qn_metrics_v6_home_ref_d3qn")
    parser.add_argument("--save_best_by", type=str, default="eval_reward", choices=["reward_ma", "reward", "eval_reward"])

    parser.add_argument("--num_devices", type=int, default=30)
    parser.add_argument("--num_servers", type=int, default=6)
    parser.add_argument("--battery_capacity", type=float, default=16.0)
    parser.add_argument("--infeasible_penalty", type=float, default=4.0)
    parser.add_argument("--delay_penalty_scale", type=float, default=1.5)
    parser.add_argument("--energy_penalty_scale", type=float, default=1.0)
    parser.add_argument("--dead_step_penalty_ratio", type=float, default=0.50)

    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=400000)
    parser.add_argument(
        "--replay_strategy",
        type=str,
        default="mixed_recent_uniform",
        choices=["uniform", "mixed_recent_uniform", "mixed_recent_per", "per"],
    )
    parser.add_argument("--recent_ratio", type=float, default=0.30)
    parser.add_argument("--recent_window", type=int, default=20000)
    parser.add_argument("--per_alpha", type=float, default=0.4)
    parser.add_argument("--per_beta_start", type=float, default=0.4)
    parser.add_argument("--per_beta_end", type=float, default=1.0)
    parser.add_argument("--per_eps", type=float, default=1e-4)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--explore_noise_start", type=float, default=0.10)
    parser.add_argument("--explore_noise_min", type=float, default=0.03)
    parser.add_argument("--explore_noise_decay", type=float, default=0.9996)
    parser.add_argument("--explore_noise_decay_start_episode", type=int, default=60)
    parser.add_argument("--relay_epsilon_start", type=float, default=0.20)
    parser.add_argument("--relay_epsilon_min", type=float, default=0.05)
    parser.add_argument("--relay_epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--relay_epsilon_decay_start_episode", type=int, default=60)
    parser.add_argument("--relay_epsilon_decay_start_it", type=int, default=50)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_seed_base", type=int, default=100000)
    parser.add_argument("--quiet", action="store_true", help="Disable per-episode console logging.")
    parser.add_argument("--print_interval", type=int, default=1, help="Print once every K episodes when not quiet.")
    parser.add_argument("--final_report_name", type=str, default="final_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    agent = make_agent(args)

    episode_rewards: List[float] = []
    episode_steps: List[int] = []

    episode_success_rates: List[float] = []
    episode_drop_counts: List[int] = []
    episode_dead_event_counts: List[int] = []
    episode_dead_rates: List[float] = []
    episode_timeout_counts: List[int] = []
    episode_avg_delay_executed: List[float] = []
    episode_avg_energy_executed: List[float] = []
    episode_valid_counts: List[int] = []
    episode_avg_weights: List[float] = []
    episode_xi_invalid_counts: List[int] = []
    episode_energy_violation_counts: List[int] = []
    episode_avg_device_steps: List[float] = []
    episode_avg_device_steps_ratio: List[float] = []
    episode_server_user_load_bits: List[List[float]] = []
    episode_server_user_load_norm: List[List[float]] = []
    episode_server_bg_load_bits: List[List[float]] = []
    episode_server_bg_load_norm: List[List[float]] = []
    episode_server_total_load_bits: List[List[float]] = []
    episode_server_total_load_norm: List[List[float]] = []
    episode_sample_recent_ratios: List[float] = []
    episode_priority_means: List[float] = []
    episode_priority_maxes: List[float] = []
    episode_is_weight_means: List[float] = []
    eval_episode_indices: List[int] = []
    eval_rewards: List[float] = []
    eval_success_rates: List[float] = []
    eval_timeout_counts: List[float] = []
    eval_best_so_far: List[float] = []

    best_score = -float("inf")
    best_reward = -float("inf")
    best_eval_reward = -float("inf")
    best_ckpt = None
    best_episode = -1

    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + ep)
        obs = state["obs"]
        region_id = state["region_id"]
        server_feat = state["server_feat"]
        hop_matrix = state["hop_matrix"]
        alive_mask = state["alive_mask"]

        ep_reward = 0.0
        success_total = 0
        timeout_total = 0

        executed_total = 0
        executed_delay_sum_total = 0.0
        executed_energy_sum_total = 0.0

        drop_count_total = 0
        dead_event_total = 0
        last_dead_count_total = 0
        steps = 0

        valid_total = 0
        avg_weight_sum_total = 0.0
        avg_weight_count_total = 0
        xi_invalid_total = 0
        energy_violation_total = 0
        alive_count_prev_sum = 0
        server_user_load_bits_ep = np.zeros(args.num_servers, dtype=np.float64)
        server_bg_load_bits_ep = np.zeros(args.num_servers, dtype=np.float64)
        sample_recent_ratio_sum = 0.0
        priority_mean_sum = 0.0
        priority_max_ep = 0.0
        is_weight_mean_sum = 0.0
        replay_stat_count = 0

        for step in range(args.max_steps):
            if ep <= args.warmup_episodes:
                cont_action, relay_target = sample_random_actions(env)
            else:
                cont_action, relay_target = agent.select_relay_target(
                    obs=obs,
                    region_id=region_id,
                    server_feat=server_feat,
                    hop_matrix=hop_matrix,
                    alive_mask=alive_mask,
                    cont_action=None,
                    deterministic_cont=False,
                )

            next_state, reward, terminated, truncated, info = env.step(
                {
                    "continuous": cont_action,
                    "relay_target": relay_target,
                }
            )

            done = float(terminated or truncated)

            agent.store_transition(
                obs=obs,
                region_id=region_id,
                cont_action=cont_action,
                relay_target=relay_target,
                reward=float(reward),
                next_obs=next_state["obs"],
                next_region_id=next_state["region_id"],
                server_feat=server_feat,
                next_server_feat=next_state["server_feat"],
                hop_matrix=hop_matrix,
                next_hop_matrix=next_state["hop_matrix"],
                alive_mask=alive_mask,
                next_alive_mask=next_state["alive_mask"],
                done=done,
            )

            train_info = {}
            if ep > args.warmup_episodes:
                train_info = agent.train_step()
                if len(train_info) > 0:
                    sample_recent_ratio_sum += float(train_info.get("sample_recent_ratio", 0.0))
                    priority_mean_sum += float(train_info.get("priority_mean", 0.0))
                    priority_max_ep = max(priority_max_ep, float(train_info.get("priority_max", 0.0)))
                    is_weight_mean_sum += float(train_info.get("is_weight_mean", 1.0))
                    replay_stat_count += 1

            ep_reward += float(reward)
            steps += 1

            success_total += int(info.get("success_count", 0))
            timeout_total += int(info.get("timeout_count", 0))
            executed_total += int(info.get("executed_count", 0))
            executed_delay_sum_total += float(info.get("executed_delay_sum", 0.0))
            executed_energy_sum_total += float(info.get("executed_energy_sum", 0.0))
            drop_count_total += int(info.get("task_dropped_count", 0))
            dead_event_total += int(info.get("dead_event_count", 0))
            last_dead_count_total = int(info.get("dead_count_total", 0))
            valid_total += int(info.get("valid_count", 0))
            avg_weight_sum_total += float(info.get("avg_weight", 0.0)) * int(info.get("valid_count", 0))
            avg_weight_count_total += int(info.get("valid_count", 0))
            xi_invalid_total += int(info.get("xi_invalid_count", 0))
            energy_violation_total += int(info.get("energy_violation_count", 0))
            alive_count_prev_sum += int(info.get("alive_count_prev", 0))
            server_user_load_bits_ep += np.asarray(info.get("server_user_load_bits_step", np.zeros(args.num_servers)), dtype=np.float64)
            server_bg_load_bits_ep += np.asarray(info.get("server_bg_load_bits_step", np.zeros(args.num_servers)), dtype=np.float64)

            obs = next_state["obs"]
            region_id = next_state["region_id"]
            server_feat = next_state["server_feat"]
            hop_matrix = next_state["hop_matrix"]
            alive_mask = next_state["alive_mask"]

            if terminated or truncated:
                break

        total_slots = args.num_devices * args.max_steps
        success_rate = success_total / max(total_slots, 1)
        dead_rate = last_dead_count_total / max(args.num_devices, 1)
        avg_delay_executed = executed_delay_sum_total / max(executed_total, 1)
        avg_energy_executed = executed_energy_sum_total / max(executed_total, 1)
        avg_weight = avg_weight_sum_total / max(avg_weight_count_total, 1)
        avg_device_steps = alive_count_prev_sum / max(args.num_devices, 1)
        avg_device_steps_ratio = avg_device_steps / max(args.max_steps, 1)

        server_total_load_bits_ep = server_user_load_bits_ep + server_bg_load_bits_ep
        user_max = float(np.max(server_user_load_bits_ep)) if server_user_load_bits_ep.size > 0 else 0.0
        bg_max = float(np.max(server_bg_load_bits_ep)) if server_bg_load_bits_ep.size > 0 else 0.0
        total_max = float(np.max(server_total_load_bits_ep)) if server_total_load_bits_ep.size > 0 else 0.0
        server_user_load_norm = (
            server_user_load_bits_ep / user_max if user_max > 1e-12 else np.zeros_like(server_user_load_bits_ep)
        )
        server_bg_load_norm = (
            server_bg_load_bits_ep / bg_max if bg_max > 1e-12 else np.zeros_like(server_bg_load_bits_ep)
        )
        server_total_load_norm = (
            server_total_load_bits_ep / total_max if total_max > 1e-12 else np.zeros_like(server_total_load_bits_ep)
        )
        ep_sample_recent_ratio = sample_recent_ratio_sum / max(replay_stat_count, 1)
        ep_priority_mean = priority_mean_sum / max(replay_stat_count, 1)
        ep_priority_max = priority_max_ep
        ep_is_weight_mean = is_weight_mean_sum / max(replay_stat_count, 1)

        episode_rewards.append(float(ep_reward))
        episode_steps.append(int(steps))
        episode_success_rates.append(float(success_rate))
        episode_drop_counts.append(int(drop_count_total))
        episode_dead_event_counts.append(int(dead_event_total))
        episode_dead_rates.append(float(dead_rate))
        episode_timeout_counts.append(int(timeout_total))
        episode_avg_delay_executed.append(float(avg_delay_executed))
        episode_avg_energy_executed.append(float(avg_energy_executed))
        episode_valid_counts.append(int(valid_total))
        episode_avg_weights.append(float(avg_weight))
        episode_xi_invalid_counts.append(int(xi_invalid_total))
        episode_energy_violation_counts.append(int(energy_violation_total))
        episode_avg_device_steps.append(float(avg_device_steps))
        episode_avg_device_steps_ratio.append(float(avg_device_steps_ratio))
        episode_server_user_load_bits.append(server_user_load_bits_ep.astype(float).tolist())
        episode_server_user_load_norm.append(server_user_load_norm.astype(float).tolist())
        episode_server_bg_load_bits.append(server_bg_load_bits_ep.astype(float).tolist())
        episode_server_bg_load_norm.append(server_bg_load_norm.astype(float).tolist())
        episode_server_total_load_bits.append(server_total_load_bits_ep.astype(float).tolist())
        episode_server_total_load_norm.append(server_total_load_norm.astype(float).tolist())
        episode_sample_recent_ratios.append(float(ep_sample_recent_ratio))
        episode_priority_means.append(float(ep_priority_mean))
        episode_priority_maxes.append(float(ep_priority_max))
        episode_is_weight_means.append(float(ep_is_weight_mean))

        reward_ma = float(np.mean(episode_rewards[-args.reward_ma_window :]))
        eval_reward: Optional[float] = None
        eval_success: Optional[float] = None
        eval_timeout: Optional[float] = None
        if args.eval_interval > 0 and ep % args.eval_interval == 0:
            eval_reward, eval_success, eval_timeout = evaluate_policy(
                agent=agent,
                args=args,
                eval_episodes=args.eval_episodes,
                eval_seed_base=args.eval_seed_base + ep * 1000,
            )
            eval_episode_indices.append(int(ep))
            eval_rewards.append(float(eval_reward))
            eval_success_rates.append(float(eval_success))
            eval_timeout_counts.append(float(eval_timeout))
            if len(eval_best_so_far) == 0:
                eval_best_so_far.append(float(eval_reward))
            else:
                eval_best_so_far.append(float(max(eval_best_so_far[-1], eval_reward)))
            best_eval_reward = max(best_eval_reward, float(eval_reward))

        if args.save_best_by == "reward_ma":
            score = reward_ma
        elif args.save_best_by == "reward":
            score = ep_reward
        else:
            score = eval_reward if eval_reward is not None else -float("inf")
        if score > best_score:
            best_score = score
            best_reward = max(best_reward, ep_reward)
            best_episode = ep
            best_ckpt = {
                "actor": copy.deepcopy(agent.actor.state_dict()),
                "critic": copy.deepcopy(agent.critic.state_dict()),
                "actor_target": copy.deepcopy(agent.actor_target.state_dict()),
                "critic_target": copy.deepcopy(agent.critic_target.state_dict()),
                "actor_opt": copy.deepcopy(agent.actor_optimizer.state_dict()),
                "critic_opt": copy.deepcopy(agent.critic_optimizer.state_dict()),
                "total_it": agent.total_it,
                "cfg": agent.cfg.__dict__.copy(),
                "best_episode": best_episode,
                "best_score": float(best_score),
                "best_reward": float(best_reward),
                "best_eval_reward": float(best_eval_reward),
            }

        if ep_reward > best_reward:
            best_reward = ep_reward

        srv_bg_str = "[" + ",".join(f"{x:.2f}" for x in server_bg_load_norm.tolist()) + "]"
        srv_user_str = "[" + ",".join(f"{x:.2f}" for x in server_user_load_norm.tolist()) + "]"
        srv_total_str = "[" + ",".join(f"{x:.2f}" for x in server_total_load_norm.tolist()) + "]"

        relay_eps = float(getattr(agent, "relay_epsilon", 0.0))
        explore_noise = float(getattr(agent, "explore_noise", 0.0))
        if (not args.quiet) and (args.print_interval > 0) and (
            ep == 1 or ep == args.episodes or ep % args.print_interval == 0
        ):
            print(
                f"[Episode {ep}/{args.episodes}] "
                f"reward={ep_reward:.4f} | ma={reward_ma:.4f} | steps={steps} | avg_dev_steps={avg_device_steps:.2f} | "
                f"valid={valid_total} | succ_rate={success_rate:.4f} | drop={drop_count_total} | "
                f"dead={dead_event_total} | dead_rate={dead_rate:.4f} | timeout={timeout_total} | "
                f"avg_delay={avg_delay_executed:.6f} | avg_energy={avg_energy_executed:.6f} | avg_weight={avg_weight:.4f} | "
                f"relay_eps={relay_eps:.4f} | explore_noise={explore_noise:.4f} | "
                f"sample_recent={ep_sample_recent_ratio:.3f} | pri_mean={ep_priority_mean:.4f} | pri_max={ep_priority_max:.4f} | is_w={ep_is_weight_mean:.4f} | "
                f"srv_bg={srv_bg_str} | srv_user={srv_user_str} | srv_total={srv_total_str}"
            )
            if eval_reward is not None:
                print(
                    f"  [Eval@{ep}] reward={eval_reward:.4f} | success={float(eval_success):.4f} | timeout={float(eval_timeout):.2f} | "
                    f"best_eval={best_eval_reward:.4f}"
                )

        if ep > args.warmup_episodes:
            agent.on_episode_end(ep)

    training_time = time.time() - t0
    reward_ma_series = moving_average(episode_rewards, args.reward_ma_window)
    tail_w = min(100, len(episode_rewards))
    best_w = min(100, len(episode_rewards))
    best_w_start = max(0, best_episode - best_w // 2 - 1)
    best_w_end = min(len(episode_rewards), best_w_start + best_w)
    around_best_timeout = float(np.mean(episode_timeout_counts[best_w_start:best_w_end])) if best_w_end > best_w_start else 0.0
    around_best_dead_rate = float(np.mean(episode_dead_rates[best_w_start:best_w_end])) if best_w_end > best_w_start else 0.0
    around_best_energy_violation = float(np.mean(episode_energy_violation_counts[best_w_start:best_w_end])) if best_w_end > best_w_start else 0.0
    tail_timeout = float(np.mean(episode_timeout_counts[-tail_w:])) if tail_w > 0 else 0.0
    tail_dead_rate = float(np.mean(episode_dead_rates[-tail_w:])) if tail_w > 0 else 0.0
    tail_energy_violation = float(np.mean(episode_energy_violation_counts[-tail_w:])) if tail_w > 0 else 0.0

    metrics_json = {
        "meta": {
            "algorithm": "HA-TD3-D3QN-v6-relay-eps",
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "reward_ma_window": args.reward_ma_window,
            "save_best_by": args.save_best_by,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "infeasible_penalty": args.infeasible_penalty,
            "delay_penalty_scale": args.delay_penalty_scale,
            "energy_penalty_scale": args.energy_penalty_scale,
            "dead_step_penalty_ratio": args.dead_step_penalty_ratio,
            "battery_capacity": args.battery_capacity,
            "replay_strategy": args.replay_strategy,
            "buffer_size": args.buffer_size,
            "recent_ratio": args.recent_ratio,
            "recent_window": args.recent_window,
            "per_alpha": args.per_alpha,
            "per_beta_start": args.per_beta_start,
            "per_beta_end": args.per_beta_end,
            "per_eps": args.per_eps,
            "reward_definition": "cumulative average reward per episode",
            "success_rate_definition": "successful device-step samples / (N * T_step)",
            "drop_count_definition": "sum of task_dropped_count across all steps in an episode",
            "dead_event_count_definition": "sum of dead_event_count across all steps in an episode",
            "dead_rate_definition": "dead devices at episode end / N",
            "avg_delay_definition": "sum of executed sample delays / executed sample count",
            "avg_energy_definition": "sum of executed sample energies / executed sample count",
            "timeout_definition": "sum of timeout_count across all steps in an episode",
            "valid_definition": "sum of executed_count across all steps in an episode",
            "avg_weight_definition": "weighted mean of step avg_weight over executed samples in an episode",
            "avg_device_steps_definition": "sum of alive_count_prev across all steps divided by N",
            "relay_target_definition": "secondary forwarding target selected by region server after device uplink reaches its home region server",
            "server_user_load_definition": "sum of per-step user accepted load bits for each server in an episode",
            "server_total_load_definition": "server_user_load_bits + background equivalent bits, normalized by episode-wise max server",
            "relay_epsilon_definition": "current epsilon used by epsilon-greedy secondary relay-target selection",
            "explore_noise_definition": "current gaussian noise std used by actor for continuous-action exploration",
            "sample_recent_ratio_definition": "fraction of sampled transitions from recent window in this episode",
            "priority_mean_definition": "mean replay priority observed during this episode",
            "priority_max_definition": "max replay priority observed during this episode",
            "is_weight_mean_definition": "mean importance-sampling weight observed during this episode",
        },
        "episode_rewards": episode_rewards,
        "episode_reward_ma": reward_ma_series,
        "episode_steps": episode_steps,
        "episode_success_rates": episode_success_rates,
        "episode_drop_counts": episode_drop_counts,
        "episode_dead_event_counts": episode_dead_event_counts,
        "episode_dead_rates": episode_dead_rates,
        "episode_timeout_counts": episode_timeout_counts,
        "episode_avg_delay_executed": episode_avg_delay_executed,
        "episode_avg_energy_executed": episode_avg_energy_executed,
        "episode_valid_counts": episode_valid_counts,
        "episode_avg_weights": episode_avg_weights,
        "episode_xi_invalid_counts": episode_xi_invalid_counts,
        "episode_energy_violation_counts": episode_energy_violation_counts,
        "episode_avg_device_steps": episode_avg_device_steps,
        "episode_avg_device_steps_ratio": episode_avg_device_steps_ratio,
        "episode_server_user_load_bits": episode_server_user_load_bits,
        "episode_server_user_load_norm": episode_server_user_load_norm,
        "episode_server_bg_load_bits": episode_server_bg_load_bits,
        "episode_server_bg_load_norm": episode_server_bg_load_norm,
        "episode_server_total_load_bits": episode_server_total_load_bits,
        "episode_server_total_load_norm": episode_server_total_load_norm,
        "episode_sample_recent_ratios": episode_sample_recent_ratios,
        "episode_priority_means": episode_priority_means,
        "episode_priority_maxes": episode_priority_maxes,
        "episode_is_weight_means": episode_is_weight_means,
        "eval_episode_indices": eval_episode_indices,
        "eval_rewards": eval_rewards,
        "eval_success_rates": eval_success_rates,
        "eval_timeout_counts": eval_timeout_counts,
        "eval_best_so_far": eval_best_so_far,
    }
    with open(out_dir / "episode_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    summary = {
        "algorithm": "HA-TD3-D3QN-v6-relay-eps",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "best_reward": float(best_reward),
        "best_eval_reward": float(best_eval_reward if best_eval_reward > -float("inf") else float("nan")),
        "infeasible_penalty": float(args.infeasible_penalty),
        "delay_penalty_scale": float(args.delay_penalty_scale),
        "energy_penalty_scale": float(args.energy_penalty_scale),
        "dead_step_penalty_ratio": float(args.dead_step_penalty_ratio),
        "battery_capacity": float(args.battery_capacity),
        "final_reward": float(episode_rewards[-1]),
        "final_reward_ma": float(reward_ma_series[-1]),
        "final_success_rate": float(episode_success_rates[-1]),
        "final_drop_count": int(episode_drop_counts[-1]),
        "final_dead_event_count": int(episode_dead_event_counts[-1]),
        "final_dead_rate": float(episode_dead_rates[-1]),
        "final_timeout_count": int(episode_timeout_counts[-1]),
        "final_avg_delay_executed": float(episode_avg_delay_executed[-1]),
        "final_avg_energy_executed": float(episode_avg_energy_executed[-1]),
        "final_valid_count": int(episode_valid_counts[-1]),
        "final_avg_weight": float(episode_avg_weights[-1]),
        "final_avg_device_steps": float(episode_avg_device_steps[-1]),
        "final_avg_device_steps_ratio": float(episode_avg_device_steps_ratio[-1]),
        "final_xi_invalid_count": int(episode_xi_invalid_counts[-1]),
        "final_energy_violation_count": int(episode_energy_violation_counts[-1]),
        "final_server_bg_load_norm": episode_server_bg_load_norm[-1],
        "final_server_user_load_norm": episode_server_user_load_norm[-1],
        "final_server_total_load_norm": episode_server_total_load_norm[-1],
        "final_relay_epsilon": float(getattr(agent, "relay_epsilon", 0.0)),
        "final_explore_noise": float(getattr(agent, "explore_noise", 0.0)),
        "final_sample_recent_ratio": float(episode_sample_recent_ratios[-1]),
        "final_priority_mean": float(episode_priority_means[-1]),
        "final_priority_max": float(episode_priority_maxes[-1]),
        "final_is_weight_mean": float(episode_is_weight_means[-1]),
        "best_success_rate": float(np.max(episode_success_rates)),
        "best_drop_count": int(np.min(episode_drop_counts)),
        "best_dead_event_count": int(np.min(episode_dead_event_counts)),
        "best_dead_rate": float(np.min(episode_dead_rates)),
        "best_timeout_count": int(np.min(episode_timeout_counts)),
        "best_avg_delay_executed": float(np.min(episode_avg_delay_executed)),
        "best_avg_energy_executed": float(np.min(episode_avg_energy_executed)),
        "best_avg_weight": float(np.max(episode_avg_weights)),
        "best_avg_device_steps": float(np.max(episode_avg_device_steps)),
        "training_time_sec": float(training_time),
        "best_episode": int(best_episode),
        "audit_around_best_timeout_mean": around_best_timeout,
        "audit_tail100_timeout_mean": tail_timeout,
        "audit_timeout_delta_tail_vs_best": float(tail_timeout - around_best_timeout),
        "audit_around_best_dead_rate_mean": around_best_dead_rate,
        "audit_tail100_dead_rate_mean": tail_dead_rate,
        "audit_dead_rate_delta_tail_vs_best": float(tail_dead_rate - around_best_dead_rate),
        "audit_around_best_energy_violation_mean": around_best_energy_violation,
        "audit_tail100_energy_violation_mean": tail_energy_violation,
        "audit_energy_violation_delta_tail_vs_best": float(tail_energy_violation - around_best_energy_violation),
        "reward_curve_path": str(out_dir / "reward_curve.png"),
        "eval_curve_path": str(out_dir / "eval_convergence_curve.png"),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    final_report = {
        "algorithm": summary["algorithm"],
        "episodes": summary["episodes"],
        "max_steps": summary["max_steps"],
        "best_reward": summary["best_reward"],
        "best_eval_reward": summary["best_eval_reward"],
        "final_reward": summary["final_reward"],
        "final_reward_ma": summary["final_reward_ma"],
        "tail100_reward": float(np.mean(episode_rewards[-tail_w:])) if tail_w > 0 else float("nan"),
        "tail100_timeout": tail_timeout,
        "tail100_dead_rate": tail_dead_rate,
        "tail100_energy_violation": tail_energy_violation,
        "peak_to_tail_drawdown": float(np.max(np.asarray(episode_rewards, dtype=np.float64)) - np.mean(np.asarray(episode_rewards[-tail_w:], dtype=np.float64)))
        if tail_w > 0
        else float("nan"),
        "final_sample_recent_ratio": summary["final_sample_recent_ratio"],
        "final_priority_mean": summary["final_priority_mean"],
        "final_priority_max": summary["final_priority_max"],
        "final_is_weight_mean": summary["final_is_weight_mean"],
        "best_episode": summary["best_episode"],
        "training_time_sec": summary["training_time_sec"],
        "output_dir": summary["output_dir"],
        "reward_curve_path": summary["reward_curve_path"],
        "eval_curve_path": summary["eval_curve_path"],
        "train_summary_path": str(out_dir / "train_summary.json"),
        "episode_metrics_path": str(out_dir / "episode_metrics.json"),
    }
    with open(out_dir / args.final_report_name, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    if best_ckpt is not None:
        torch.save(best_ckpt, out_dir / "best_model.pt")
    agent.save(str(out_dir / "last_model.pt"))

    save_reward_curve(
        episode_rewards=episode_rewards,
        reward_ma_series=reward_ma_series,
        output_path=out_dir / "reward_curve.png",
        ma_window=args.reward_ma_window,
    )
    save_eval_curve(
        eval_episode_indices=eval_episode_indices,
        eval_rewards=eval_rewards,
        eval_best_so_far=eval_best_so_far,
        output_path=out_dir / "eval_convergence_curve.png",
    )

    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Final compact report saved to: {str(out_dir / args.final_report_name)}")


if __name__ == "__main__":
    main()
