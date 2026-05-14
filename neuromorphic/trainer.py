"""SNN training loop with optional WandB logging and structured sparsification."""

from __future__ import annotations

import copy
import logging
from typing import Any

import snntorch.functional as SF
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from neuromorphic import utils
from neuromorphic.model import Metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trains a SpikingNet on a DataLoader-based dataset.

    Parameters
    ----------
    net:               The SNN model.
    train_loader:      DataLoader for training data.
    val_loader:        DataLoader for validation data.
    target_sparsity:   Fraction of long-range connections to retain (1.0 = no pruning).
    recall_duration:   Number of final time steps used for classification loss.
    graph:             Graph object (required when target_sparsity < 1.0).
    num_epochs:        Training epochs.
    learning_rate:     Initial learning rate.
    target_frequency:  Unused; reserved for future rate-regularisation.
    num_steps:         Sequence length.
    optimizer:         "Adam" or "AdamW".
    wandb_logging:     Whether to log metrics to Weights & Biases.
    """

    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_sparsity: float,
        recall_duration: int,
        graph: Any = None,
        num_epochs: int = 150,
        learning_rate: float = 1e-4,
        target_frequency: float = 0.5,
        num_steps: int = 10,
        optimizer: str = "AdamW",
        wandb_logging: bool = False,
    ) -> None:
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.graph = graph
        self.target_sparsity = target_sparsity
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.target_frequency = target_frequency
        self.num_steps = num_steps
        self.recall_duration = recall_duration
        self.wandb_logging = wandb_logging
        self.mtrcs = Metrics()
        self.indices: dict | None = None

        opt_cls = optim.AdamW if optimizer == "AdamW" else optim.Adam
        self.optimizer = opt_cls(self.net.parameters(), lr=learning_rate)
        self.criterion = SF.ce_count_loss()

        logger.info("\n----- TRAINING -----\n")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def eval(
        self,
        device: torch.device,
        val_idx: int,
        external_model: nn.Module | None = None,
        final: bool = False,
    ) -> tuple[float, int]:
        model = external_model if external_model is not None else self.net
        model.eval()
        if external_model is not None:
            model = model.to(device)

        acc_sum, count = 0.0, 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output, *_ = model(data, time_first=False)
                recall = output[-self.recall_duration :]

                if final:
                    preds = self.mtrcs.return_predicted(recall)
                    self.mtrcs.perf_measure(target, preds)

                acc_sum += SF.acc.accuracy_rate(recall, target)
                val_loss = self.criterion(recall, target)

                if self.wandb_logging:
                    import wandb

                    wandb.log({"Val loss": val_loss.item(), "Val index": val_idx})

                count += 1
                val_idx += 1

        return acc_sum / max(count, 1), val_idx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        device: torch.device,
        mapping: Any = None,
        dut: Any = None,
    ) -> tuple[nn.Module, Any, float, float, Metrics]:
        """
        Run the full training loop.

        Returns
        -------
        (best_net, mapping, best_accuracy, final_accuracy, metrics)
        """
        best_acc = 0.0
        best_state: dict | None = None
        self.net = self.net.to(device)

        # Compute per-epoch pruning budget
        conn_reps = 0
        if self.target_sparsity != 1.0 and mapping is not None:
            self.indices = mapping.indices_to_lock
            lr, sr = utils.calculate_lr_sr_conns(mapping, self.graph)
            logger.info("Long-range: %d  Short-range: %d", lr, sr)
            conn_reps = int((lr - lr * self.target_sparsity) / self.num_epochs)
            logger.info("Connections to remove per epoch: %d", conn_reps)

        train_idx = val_idx = 0

        for epoch in range(self.num_epochs):
            self.net.train()
            loss = torch.tensor(0.0)

            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                outputs, *_ = self.net(data, time_first=False)
                loss = self.criterion(outputs[-self.recall_duration :], target)

                if self.wandb_logging:
                    import wandb

                    wandb.log({"loss": loss.item(), "Train index": train_idx})

                self.optimizer.zero_grad()
                loss.backward()

                # Zero frozen connection gradients before and after step
                self._zero_locked_grads()
                self.optimizer.step()
                self._zero_locked_grads()

                train_idx += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                accuracy, val_idx = self.eval(device, val_idx)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_state = copy.deepcopy(self.net.state_dict())

                msg = f"Epoch [{epoch + 1}/{self.num_epochs}]  loss={loss.item():.4f}  val_acc={accuracy:.4f}"
                if dut is not None:
                    dut._log.info(msg)
                else:
                    logger.info(msg)

                if self.wandb_logging:
                    import wandb

                    wandb.log({"Train Accuracy": accuracy})

            if self.target_sparsity != 1.0 and mapping is not None:
                mapping = utils.choose_conn_remove(mapping, reps=conn_reps)
                self.indices = mapping.indices_to_lock

        if self.target_sparsity != 1.0 and mapping is not None:
            lr, sr = utils.calculate_lr_sr_conns(mapping, self.graph)
            logger.info("Final – Long-range: %d  Short-range: %d", lr, sr)

        final_acc, _ = self.eval(device, val_idx, final=True)
        logger.info("Final accuracy: %.4f", final_acc)
        logger.info(
            "Precision: %.3f  Recall: %.3f  F1: %.3f",
            self.mtrcs.precision(),
            self.mtrcs.recall(),
            self.mtrcs.f1_score(),
        )

        if self.wandb_logging:
            import wandb

            wandb.log(
                {
                    "Final Accuracy": final_acc,
                    "Precision": self.mtrcs.precision(),
                    "Recall": self.mtrcs.recall(),
                    "F1-Score": self.mtrcs.f1_score(),
                }
            )

        if best_state is not None:
            self.net.load_state_dict(best_state)

        torch.save(self.net.state_dict(), "model.pth")
        return self.net, mapping, best_acc, final_acc, self.mtrcs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _zero_locked_grads(self) -> None:
        if self.indices is None or self.target_sparsity == 1.0:
            return
        layer = self.net.lif1.recurrent  # type: ignore[attr-defined]
        for idx in self.indices["indices"]:
            layer.weight.data[idx] = 0
            if layer.weight.grad is not None:
                layer.weight.grad[idx] = 0
