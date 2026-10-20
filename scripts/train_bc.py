import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import glob
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from schafkopfrl.data import iter_batches, load_expert_shard
from schafkopfrl.evaluation import NetworkPlayer, evaluate_vs_rulebased, validation_accuracy
from schafkopfrl.policy.models import build_actor_critic


def main() -> None:
    """Behavioural cloning of the policy from expert shards; the last shard is held out for validation."""
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shards"))
    p.add_argument("--encoder", choices=["lstm", "transformer"], default="transformer")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--history-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=0.2)
    p.add_argument("--eval-interval", type=int, default=20, help="evaluate every N shards")
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "bc"))
    args = p.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shard_dir, "expert_data_shard_*.jsonl")))
    train_shards, validation_shard = shards[:-1], shards[-1]
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(args.out_dir)

    actor_critic = build_actor_critic(args.encoder, args.hidden, args.history_dim, args.num_layers, args.num_heads).to(device)
    policy = actor_critic.get_policy_operator()
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    step = 0
    shards_seen = 0
    for epoch in range(args.epochs):
        random.shuffle(train_shards)
        for shard_path in train_shards:
            data = load_expert_shard(shard_path)
            for td, actions in iter_batches(data, args.batch_size):
                loss = -policy.get_dist(td.to(device)).log_prob(actions.to(device)).mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optim.step()
                writer.add_scalar("train/nll", loss.item(), step)
                step += 1
            shards_seen += 1
            print(f"epoch {epoch} shard {shards_seen}: nll={loss.item():.4f}")

            if shards_seen % args.eval_interval == 0:
                actor_critic.eval()
                accuracy = validation_accuracy(actor_critic, validation_shard)
                reward = evaluate_vs_rulebased(NetworkPlayer(actor_critic), args.eval_games)
                actor_critic.train()
                writer.add_scalar("eval/validation_accuracy", accuracy, step)
                writer.add_scalar("eval/avg_reward_against_rulebased_policy", reward, step)
                print(f"validation_accuracy={accuracy:.4f} avg_reward_against_rulebased_policy={reward:.2f}")
                torch.save(actor_critic.state_dict(), os.path.join(args.out_dir, f"actor_critic_{shards_seen}.pt"))

    torch.save(actor_critic.state_dict(), os.path.join(args.out_dir, "actor_critic_final.pt"))


if __name__ == "__main__":
    main()
