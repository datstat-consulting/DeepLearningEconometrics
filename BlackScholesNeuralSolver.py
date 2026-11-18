"""Physics-informed approximation to European Black–Scholes prices."""

import math

import torch
from torch import nn


class BlackScholesPINN(nn.Module):
    """Learn the call price on a bounded spot / time-to-expiry rectangle."""

    def __init__(self, strike, maturity, volatility, rate=0.0, dividend=0.0, *,
                 max_spot=None, width=32, depth=3):
        super().__init__()
        if not all(map(math.isfinite, (strike, maturity, volatility, rate, dividend))):
            raise ValueError("Contract parameters must be finite.")
        if min(strike, maturity, volatility, width, depth) <= 0:
            raise ValueError("Strike, maturity, volatility, width and depth must be positive.")
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.volatility = float(volatility)
        self.rate = float(rate)
        self.dividend = float(dividend)
        self.max_spot = float(4 * strike if max_spot is None else max_spot)
        upper_boundary_min = strike * max(1, math.exp((dividend - rate) * maturity))
        if not math.isfinite(self.max_spot) or self.max_spot <= upper_boundary_min:
            raise ValueError("max_spot must exceed the discounted break-even spot.")
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        self.network = nn.Sequential(*layers, nn.Linear(width, 1))

    def _inputs(self, spot, tau):
        reference = next(self.parameters())
        return torch.broadcast_tensors(*[
            torch.as_tensor(x, dtype=reference.dtype, device=reference.device)
            for x in (spot, tau)
        ])

    def forward(self, spot, tau, *, kind="call"):
        spot, tau = self._inputs(spot, tau)
        x = spot / self.max_spot
        t = tau / self.maturity
        upper = self.max_spot * torch.exp(-self.dividend * tau) - self.strike * torch.exp(
            -self.rate * tau
        )
        # Gaussian smoothing removes the payoff kink for every positive tau.
        std = self.strike * self.volatility * tau.sqrt()
        smooth = self._smooth_payoff(spot, std)
        lower_smooth = self._smooth_payoff(spot.new_zeros(()), std)
        upper_smooth = self._smooth_payoff(spot.new_tensor(self.max_spot), std)
        baseline = smooth - (1 - x) * lower_smooth + x * (upper - upper_smooth)
        correction = self.network(torch.stack((x, t.sqrt()), dim=-1)).squeeze(-1)
        call = baseline + self.strike * t.sqrt() * x * (1 - x) * correction
        if kind == "call":
            return call
        if kind == "put":
            return call - spot * torch.exp(-self.dividend * tau) + self.strike * torch.exp(
                -self.rate * tau
            )
        raise ValueError("kind must be 'call' or 'put'.")

    def _smooth_payoff(self, spot, std):
        distance = spot - self.strike
        z = distance / std.clamp_min(torch.finfo(std.dtype).tiny)
        density = (-0.5 * z.square()).exp() / math.sqrt(2 * math.pi)
        return distance * torch.special.ndtr(z) + std * density

    def residual(self, spot, tau):
        """Evaluate the call PDE residual at interior points with positive tau."""
        spot, tau = (x.detach().requires_grad_(True) for x in self._inputs(spot, tau))
        value = self(spot, tau)
        delta, time_derivative = torch.autograd.grad(value.sum(), (spot, tau), create_graph=True)
        gamma, = torch.autograd.grad(delta.sum(), spot, create_graph=True)
        return (
            time_derivative
            - 0.5 * self.volatility**2 * spot.square() * gamma
            - (self.rate - self.dividend) * spot * delta
            + self.rate * value
        )

    def fit(self, *, steps=1000, batch_size=256, learning_rate=1e-3, seed=None):
        """Minimize the PDE residual at random collocation points."""
        if steps < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("Training settings must be positive.")
        ref = next(self.parameters())
        generator = None if seed is None else torch.Generator(device=ref.device).manual_seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = []
        self.train()
        for _ in range(steps):
            draw = torch.rand((2, batch_size), device=ref.device, dtype=ref.dtype,
                              generator=generator)
            spot = self.max_spot * draw[0]
            tau = self.maturity * (1e-3 + (1 - 1e-3) * draw[1])
            loss = (self.residual(spot, tau) * self.maturity / self.strike).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        return history

    def exact_price(self, spot, tau, *, kind="call"):
        """Closed-form benchmark for the same contract."""
        spot, tau = self._inputs(spot, tau)
        safe_spot = spot.clamp_min(torch.finfo(spot.dtype).tiny)
        std = self.volatility * tau.clamp_min(torch.finfo(tau.dtype).tiny).sqrt()
        d1 = ((safe_spot / self.strike).log()
              + (self.rate - self.dividend + self.volatility**2 / 2) * tau) / std
        d2 = d1 - std
        call = spot * torch.exp(-self.dividend * tau) * torch.special.ndtr(d1)
        call -= self.strike * torch.exp(-self.rate * tau) * torch.special.ndtr(d2)
        call = torch.where(tau == 0, (spot - self.strike).clamp_min(0), call)
        if kind == "call":
            return call
        if kind == "put":
            return call - spot * torch.exp(-self.dividend * tau) + self.strike * torch.exp(
                -self.rate * tau
            )
        raise ValueError("kind must be 'call' or 'put'.")
