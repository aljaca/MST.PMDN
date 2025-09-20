"""Simple training helper for MST-PMDN models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import mst_pmdn_nll


@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[Optional[float]]


def _make_dataset(inputs: Tensor, targets: Tensor, image_inputs: Optional[Tensor]) -> TensorDataset:
    if image_inputs is None:
        return TensorDataset(inputs, targets)
    return TensorDataset(inputs, image_inputs, targets)


def _split_dataset(dataset: TensorDataset, val_split: float) -> tuple[TensorDataset, Optional[TensorDataset]]:
    if val_split <= 0:
        return dataset, None
    if not 0 < val_split < 1:
        raise ValueError("val_split must be in (0, 1)")
    length = len(dataset)
    val_len = int(length * val_split)
    perm = torch.randperm(length)
    val_idx = perm[:val_len]
    train_idx = perm[val_len:]
    train_tensors = [tensor[train_idx] for tensor in dataset.tensors]
    val_tensors = [tensor[val_idx] for tensor in dataset.tensors]
    return TensorDataset(*train_tensors), TensorDataset(*val_tensors)


def train_mst_pmdn(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    image_inputs: Optional[Tensor] = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.0,
    device: Optional[torch.device | str] = None,
    lambda_alpha: float = 0.0,
    lambda_nu_inv: float = 0.0,
    patience: Optional[int] = None,
) -> TrainingHistory:
    """Train an ``MSTPMDN`` model using mini-batch gradient descent."""

    dataset = _make_dataset(inputs, targets, image_inputs)
    train_dataset, val_dataset = _split_dataset(dataset, val_split)

    device = torch.device(device or inputs.device)
    model.to(device)

    def _loader(ds: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(train_dataset, shuffle=True)
    val_loader = _loader(val_dataset, shuffle=False) if val_dataset is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    wait = 0
    history = TrainingHistory(train_loss=[], val_loss=[])

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if image_inputs is None:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                output = model(x)
            else:
                x, im, y = batch
                x = x.to(device)
                im = im.to(device)
                y = y.to(device)
                output = model(x, image_input=im)
            loss = mst_pmdn_nll(output, y, lambda_alpha=lambda_alpha, lambda_nu_inv=lambda_nu_inv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            count += 1
        epoch_loss = total_loss / max(count, 1)
        history.train_loss.append(epoch_loss)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                val_count = 0
                for batch in val_loader:
                    if image_inputs is None:
                        x, y = batch
                        x = x.to(device)
                        y = y.to(device)
                        output = model(x)
                    else:
                        x, im, y = batch
                        x = x.to(device)
                        im = im.to(device)
                        y = y.to(device)
                        output = model(x, image_input=im)
                    loss = mst_pmdn_nll(output, y, lambda_alpha=lambda_alpha, lambda_nu_inv=lambda_nu_inv)
                    val_total += loss.item()
                    val_count += 1
                val_loss = val_total / max(val_count, 1)
            history.val_loss.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    break
        else:
            history.val_loss.append(None)

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
