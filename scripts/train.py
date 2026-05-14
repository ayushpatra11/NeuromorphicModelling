"""
Training entry point.

Usage
-----
  # Simple training (no WandB):
  python scripts/train.py

  # WandB hyperparameter sweep:
  WANDB_API_KEY=<key> python scripts/train.py --sweep
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from snntorch import surrogate
from torch.utils.data import DataLoader

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromorphic import utils
from neuromorphic.config import ModelConfig
from neuromorphic.dataset import NavDataset
from neuromorphic.model import SpikingNet
from neuromorphic.trainer import Trainer
from neuromorphic.utils import setup_logging

setup_logging("train")
logger = logging.getLogger(__name__)


def train_simple(cfg: ModelConfig, save_path: str = "model.pth") -> None:
    """Train with fast_sigmoid surrogate gradient and save the best model."""
    torch.manual_seed(42)
    net = SpikingNet(cfg)
    sample = torch.randn(cfg.num_steps, cfg.num_inputs)
    net = utils.init_network(net, sample)

    f0 = 40.0 / 100.0
    train_set = NavDataset(
        cfg.num_steps,
        cfg.num_inputs,
        cfg.recall_duration,
        cfg.p_group,
        f0,
        cfg.n_cues,
        cfg.t_cue,
        cfg.t_cue_spacing,
        4,
        length=100,
    )
    val_set = NavDataset(
        cfg.num_steps,
        cfg.num_inputs,
        cfg.recall_duration,
        cfg.p_group,
        f0,
        cfg.n_cues,
        cfg.t_cue,
        cfg.t_cue_spacing,
        4,
        length=50,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.bs, shuffle=True, num_workers=0)

    trainer = Trainer(
        net,
        train_loader,
        val_loader,
        cfg.target_sparsity,
        cfg.recall_duration,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        num_steps=cfg.num_steps,
    )
    net, _, best_acc, final_acc, mtrcs = trainer.train(cfg.device)
    torch.save(net.state_dict(), save_path)
    logger.info("Saved model → %s  (best_val=%.4f  final=%.4f)", save_path, best_acc, final_acc)
    logger.info(
        "Precision=%.3f  Recall=%.3f  F1=%.3f",
        mtrcs.precision(),
        mtrcs.recall(),
        mtrcs.f1_score(),
    )


def run_sweep() -> None:
    """Launch a WandB hyperparameter sweep."""
    import wandb

    from neuromorphic.sweep import SweepHandler

    cfg = ModelConfig()
    sweep_handler = SweepHandler()

    def sweep_fn(config=None):
        with wandb.init(config=config):
            wc = wandb.config
            grad_map = {
                "atan": surrogate.atan(),
                "sigmoid": surrogate.sigmoid(),
                "fast_sigmoid": surrogate.fast_sigmoid(),
            }
            spike_grad = grad_map.get(wc.surrogate_gradient, surrogate.fast_sigmoid())

            net = SpikingNet(
                cfg,
                spike_grad=spike_grad,
                learn_alpha=wc.get("learn_alpha", False),
                learn_beta=wc.get("learn_beta", False),
                learn_threshold=wc.get("learn_threshold", False),
            )
            sample = torch.randn(cfg.num_steps, cfg.num_inputs)
            net = utils.init_network(net, sample)

            f0 = 40.0 / 100.0
            train_set = NavDataset(
                cfg.num_steps,
                cfg.num_inputs,
                cfg.recall_duration,
                cfg.p_group,
                f0,
                cfg.n_cues,
                cfg.t_cue,
                cfg.t_cue_spacing,
                4,
                length=100,
            )
            val_set = NavDataset(
                cfg.num_steps,
                cfg.num_inputs,
                cfg.recall_duration,
                cfg.p_group,
                f0,
                cfg.n_cues,
                cfg.t_cue,
                cfg.t_cue_spacing,
                4,
                length=50,
            )
            train_loader = DataLoader(
                train_set, batch_size=wc.batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(val_set, batch_size=wc.batch_size, shuffle=True, num_workers=0)
            trainer = Trainer(
                net,
                train_loader,
                val_loader,
                cfg.target_sparsity,
                cfg.recall_duration,
                num_epochs=cfg.num_epochs,
                learning_rate=wc.learning_rate,
                optimizer=wc.optimizer,
                target_frequency=cfg.target_fr,
                num_steps=cfg.num_steps,
                wandb_logging=True,
            )
            trained_net, _, _, _, _ = trainer.train(cfg.device)
            torch.save(trained_net.state_dict(), "best_snn.pth")

    sweep_config = {
        "method": "random",
        "metric": sweep_handler.metric,
        "parameters": sweep_handler.parameters_dict,
    }
    wandb.login(key=cfg.wandb_key)
    sweep_id = wandb.sweep(sweep_config, project="MSC_Thesis_SNN_Training")
    wandb.agent(sweep_id, sweep_fn)
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNN for binary navigation task")
    parser.add_argument("--sweep", action="store_true", help="Run WandB hyperparameter sweep")
    parser.add_argument("--save", default="model.pth", help="Output model checkpoint path")
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="Override number of training epochs"
    )
    args = parser.parse_args()

    cfg = ModelConfig()
    cfg.train = True
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs

    if args.sweep:
        run_sweep()
    else:
        train_simple(cfg, save_path=args.save)


if __name__ == "__main__":
    main()
