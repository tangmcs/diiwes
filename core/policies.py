"""Small NumPy policies for ES experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class MLPPolicy:
    """Fully connected policy with flattened parameter vectors."""

    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "tanh",
        output_activation: str | None = "tanh",
    ) -> None:
        self.ob_dim = int(ob_dim)
        self.ac_dim = int(ac_dim)
        self.hidden_dims = [int(x) for x in hidden_dims]
        self.activation = activation
        self.output_activation = output_activation

        dims = [self.ob_dim] + self.hidden_dims + [self.ac_dim]
        self.num_params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

    def _activate(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == "tanh":
            return np.tanh(x)
        if name == "relu":
            return np.maximum(x, 0.0)
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        raise ValueError(f"unknown activation: {name}")

    def act(self, observation: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64).ravel()
        params = np.asarray(params, dtype=np.float64)
        dims = [self.ob_dim] + self.hidden_dims + [self.ac_dim]
        cursor = 0
        for layer in range(len(dims) - 1):
            w_size = dims[layer] * dims[layer + 1]
            b_size = dims[layer + 1]
            weight = params[cursor : cursor + w_size].reshape(dims[layer], dims[layer + 1])
            bias = params[cursor + w_size : cursor + w_size + b_size]
            cursor += w_size + b_size
            x = x @ weight + bias
            if layer < len(dims) - 2:
                x = self._activate(x, self.activation)
            elif self.output_activation is not None:
                x = self._activate(x, self.output_activation)
        return x


class DiscretePolicy:
    """Argmax wrapper for discrete action environments."""

    def __init__(self, mlp: MLPPolicy, n_actions: int) -> None:
        self.continuous_policy = mlp
        self.n_actions = int(n_actions)
        self.ob_dim = mlp.ob_dim
        self.ac_dim = self.n_actions
        self.num_params = mlp.num_params
        if mlp.ac_dim != self.n_actions:
            raise ValueError("MLP output dimension must match n_actions")

    def act(self, observation: np.ndarray, params: np.ndarray) -> int:
        logits = self.continuous_policy.act(observation, params)
        return int(np.argmax(logits))


def make_layer_slices(policy: MLPPolicy | DiscretePolicy) -> list[slice]:
    """Return one parameter slice per MLP layer."""
    mlp = getattr(policy, "continuous_policy", policy)
    dims = [mlp.ob_dim] + list(mlp.hidden_dims) + [mlp.ac_dim]
    slices: list[slice] = []
    cursor = 0
    for layer in range(len(dims) - 1):
        layer_size = dims[layer] * dims[layer + 1] + dims[layer + 1]
        slices.append(slice(cursor, cursor + layer_size))
        cursor += layer_size
    if cursor != policy.num_params:
        raise ValueError(f"layer slices cover {cursor} params, expected {policy.num_params}")
    return slices
