import numpy as np
from scipy.stats import norm
import torch

from DeepLearningEconometrics import (
    BlackScholesPINN, DombiKappa, KalmanNet, Optimizers, PerceptronMain,
    TerminalTransport, butterworth_filter, calibrate_esscher, calibrate_esscher_theta,
)


def test_terminal_transport_and_martingale_calibration():
    model = DombiKappa(0.25, 0.6, 1.7)
    u = np.linspace(0.001, 0.999, 50)
    np.testing.assert_allclose(model.inverse(model(u)), u)
    assert np.isclose(model(0.25), 0.6)
    transport = TerminalTransport(norm(), model)
    np.testing.assert_allclose(transport.cdf(transport.ppf(u)), u)
    assert transport.pdf(9) > 0
    values = np.array([-0.3, 0.0, 0.1])
    _, weights = calibrate_esscher(values, [1, 2, 4])
    assert np.isclose(weights.sum(), 1)
    assert np.isclose(weights @ np.exp(values), 1)
    assert np.isclose(calibrate_esscher_theta(lambda t: 0.06*t + 0.02*t*t, (-4, 1)), -2)


def test_pinn_constraints_and_fit():
    torch.manual_seed(3)
    model = BlackScholesPINN(100, 1, 0.2, rate=0.03, dividend=0.01, width=16, depth=2)
    prices = torch.tensor([0., 85., 100., 115., 400.])
    torch.testing.assert_close(model(prices, 0.), (prices - 100).clamp_min(0))
    assert model(0., 0.3) == 0
    history = model.fit(steps=100, batch_size=64, seed=10)
    assert np.mean(history[-20:]) < np.mean(history[:20])
    estimate = model(prices[1:-1], 0.5)
    assert (estimate - model.exact_price(prices[1:-1], 0.5)).abs().max() < 3
    assert torch.isfinite(model.residual(prices[1:-1], 0.5)).all()


def test_signal_processing_and_perceptron_training():
    torch.manual_seed(4)
    time = torch.linspace(0, 3, 32)
    states = torch.stack((time.sin(), time.cos()), dim=1)
    observed = states[:, :1]
    model = KalmanNet([2, 6, 2], [2, 4, 1])
    before = (model(observed[1:], states[0]) - states[1:]).square().mean()
    model.fit(states, observed, epochs=12, batch_size=8, learning_rate=0.01)
    filtered = model(observed[1:], states[0])
    assert filtered.shape == (31, 2)
    assert (filtered - states[1:]).square().mean() < before
    assert butterworth_filter(np.ones(64), 5, 50).shape == (64,)

    X = time[:, None]
    model = PerceptronMain([1, 1], "linear", Optimizers.adam)
    model.fit(X, 2 * X + 1, epochs=500, batch_size=32, learning_rate=0.05)
    torch.testing.assert_close(model.predict(X), 2 * X + 1, rtol=0.03, atol=0.03)
