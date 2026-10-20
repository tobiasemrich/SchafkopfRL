import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import Collector
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import MarlGroupMapType

from schafkopfrl.environment.torchrl_env import make_torchrl_env
from schafkopfrl.evaluation import NetworkPlayer, evaluate_vs_rulebased
from schafkopfrl.policy.models import build_actor_critic
from schafkopfrl.training import acting_transitions, make_gae, make_ppo_loss


def make_env():
    return make_torchrl_env(group_map=MarlGroupMapType.ALL_IN_ONE_GROUP)


def main() -> None:
    """Self-play PPO with a shared policy for all four seats; evaluated against the rule-based policy."""
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--history-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=1, help="parallel env processes (1 = in-process)")
    p.add_argument("--frames-per-batch", type=int, default=32768, help="env steps (= agent moves) per iteration")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lmbda", type=float, default=1.0)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.2)
    p.add_argument("--reward-scale", type=float, default=0.01, help="payoffs are in cents; scaled for training only")
    p.add_argument("--eval-interval", type=int, default=3)
    p.add_argument("--eval-games", type=int, default=30)
    p.add_argument("--device", default="cpu", help="small nets and small batches usually collect faster on cpu")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ppo"))
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(args.out_dir)

    actor_critic = build_actor_critic(args.encoder, args.hidden, args.history_dim, args.num_layers, args.num_heads).to(device)
    gae = make_gae(actor_critic.get_value_operator(), args.gamma, args.lmbda)
    loss_module = make_ppo_loss(actor_critic, args.clip_epsilon, args.entropy_coeff)
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    env = ParallelEnv(args.num_envs, make_env) if args.num_envs > 1 else make_env()
    collector = Collector(
        env, actor_critic.get_policy_operator(),
        frames_per_batch=args.frames_per_batch, total_frames=args.frames_per_batch * args.iterations,
        policy_device=device, env_device="cpu", storing_device=device,
    )

    for it, data in enumerate(collector):
        data["next", "agents", "reward"] *= args.reward_scale
        with torch.no_grad():
            gae(data)
        batch = acting_transitions(data)

        stats = {}
        for _ in range(args.epochs):
            perm = torch.randperm(batch.batch_size[0], device=batch.device)
            for start in range(0, perm.numel(), args.minibatch_size):
                loss = loss_module(batch[perm[start:start + args.minibatch_size]])
                total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                optim.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.grad_clip)
                optim.step()
                stats = {k: v.item() for k, v in loss.items()}

        for k, v in stats.items():
            writer.add_scalar(f"train/{k}", v, it)
        print(f"iter {it}: " + " ".join(f"{k}={v:.3f}" for k, v in stats.items() if k in ("loss_objective", "loss_critic", "entropy")))

        if (it + 1) % args.eval_interval == 0:
            actor_critic.eval()
            reward = evaluate_vs_rulebased(NetworkPlayer(actor_critic), args.eval_games)
            actor_critic.train()
            writer.add_scalar("eval/avg_reward_against_rulebased_policy", reward, it)
            print(f"iter {it}: avg_reward_against_rulebased_policy={reward:.2f}")
            torch.save(actor_critic.state_dict(), os.path.join(args.out_dir, f"actor_critic_{it + 1}.pt"))

    collector.shutdown()


if __name__ == "__main__":
    main()
