"""Supervised neural filtering and standard digital signal operations."""

import torch
from scipy.signal import butter, sosfilt, sosfiltfilt
from torch import nn
from torch.nn import functional as F

from .PerceptronMain import Optimizers, TorchActivations

__all__ = ["KalmanNet", "butterworth_filter"]


def butterworth_filter(data, cutoff, sample_rate, *, kind="lowpass", order=4,
                       axis=-1, zero_phase=True):
    """Filter regularly sampled data. Cutoffs are in units of sample_rate."""
    sections = butter(order, cutoff, btype=kind, fs=sample_rate, output="sos")
    filter_function = sosfiltfilt if zero_phase else sosfilt
    return filter_function(sections, data, axis=axis)


class _StateModel(nn.Module):
    def __init__(self, sizes, activation, bias):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out, bias=bias) for n_in, n_out in zip(sizes, sizes[1:])
        )
        self.activation = activation

    def forward(self, inputs):
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)


class KalmanNet(nn.Module):
    """Learn transition, observation and recurrent gain maps from labelled states."""

    def __init__(self, layer_sizes_f, layer_sizes_h, activation_function="tanh",
                 optimizer_function=None, f_add_bias=True, h_add_bias=True, *,
                 gain_hidden=16):
        super().__init__()
        if any(len(sizes) < 2 or min(sizes) < 1 for sizes in (layer_sizes_f, layer_sizes_h)):
            raise ValueError("Each model needs positive input and output dimensions.")
        self.state_size = layer_sizes_f[0]
        self.observation_size = layer_sizes_h[-1]
        if layer_sizes_f[-1] != self.state_size or layer_sizes_h[0] != self.state_size:
            raise ValueError("Transition output and observation input must match state size.")
        activation = TorchActivations.activation(activation_function)
        if activation is None:
            raise ValueError(f"Unknown activation {activation_function!r}.")
        self.model_f = _StateModel(layer_sizes_f, activation, f_add_bias)
        self.model_h = _StateModel(layer_sizes_h, activation, h_add_bias)
        self.rnn = nn.GRUCell(self.observation_size, gain_hidden)
        self.gain = nn.Linear(gain_hidden, self.state_size * self.observation_size)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        legacy_optimizers = {
            Optimizers.sgd_optimizer: torch.optim.SGD,
            Optimizers.adagrad_optimizer: torch.optim.Adagrad,
            Optimizers.adam: torch.optim.Adam,
            Optimizers.rmsprop: torch.optim.RMSprop,
            Optimizers.adadelta: torch.optim.Adadelta,
        }
        self.optimizer_function = legacy_optimizers.get(optimizer_function, optimizer_function)
        self.optimizer_function = self.optimizer_function or torch.optim.Adam

    def _tensor(self, data):
        reference = self.gain.weight
        return torch.as_tensor(data, device=reference.device, dtype=reference.dtype)

    def _step(self, state, observation, hidden):
        prior = self.model_f(state)
        residual = observation - self.model_h(prior)
        hidden = self.rnn(residual, hidden)
        gain = self.gain(hidden).reshape(self.state_size, self.observation_size)
        return prior + gain @ residual, hidden

    def forward(self, observations, initial_state):
        """Filter a sequence whose first observation follows the initial state."""
        observations = self._tensor(observations).reshape(-1, self.observation_size)
        state = self._tensor(initial_state).reshape(self.state_size)
        hidden = None
        estimates = []
        for observation in observations:
            state, hidden = self._step(state, observation, hidden)
            estimates.append(state)
        return torch.stack(estimates) if estimates else state.new_empty((0, self.state_size))

    def fit(self, X, y, epochs, batch_size, learning_rate):
        """Fit aligned states X[t] and observations y[t] in contiguous subsequences."""
        states = self._tensor(X).reshape(-1, self.state_size)
        observations = self._tensor(y).reshape(-1, self.observation_size)
        if len(states) != len(observations) or len(states) < 2:
            raise ValueError("Provide at least two aligned states and observations.")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("Training settings must be positive.")
        optimizer = self.optimizer_function(self.parameters(), lr=learning_rate)
        self.train()
        for _ in range(epochs):
            for start in range(1, len(states), batch_size):
                stop = min(start + batch_size, len(states))
                target = states[start:stop]
                predicted = self(observations[start:stop], states[start - 1])
                loss = (
                    F.mse_loss(predicted, target)
                    + F.mse_loss(self.model_f(states[start - 1:stop - 1]), target)
                    + F.mse_loss(self.model_h(target), observations[start:stop])
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
        return self

    def predict_next_state(self, current_state, add_bias=True):
        """Predict one state using the transition network's configured bias."""
        return self.model_f(self._tensor(current_state))

    def update_state(self, next_state_pred, observation, add_bias=True):
        """Apply an independent one-step update. Use forward for recurrent filtering."""
        prior = self._tensor(next_state_pred).reshape(self.state_size)
        residual = self._tensor(observation).reshape(self.observation_size) - self.model_h(prior)
        gain = self.gain(self.rnn(residual)).reshape(self.state_size, self.observation_size)
        return prior + gain @ residual
