"""Terminal-law transport and Esscher calibration for discounted exponential models."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit, logsumexp


def _probability(value):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value) & (0 <= value) & (value <= 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return value


@dataclass(frozen=True)
class DombiKappa:
    """Two-anchor Dombi map using the odds generator by default.

    Supply a log-generator, its inverse and optionally its derivative to use
    another admissible generator. Custom generators must be monotone bijections
    from (0, 1) to the real line in their log coordinate.
    """

    source_anchor: float = 0.5
    target_anchor: float = 0.5
    power: float = 1.0
    log_generator: object = None
    inverse_log_generator: object = None
    log_generator_derivative: object = None

    def __post_init__(self):
        if not (0 < self.source_anchor < 1 and 0 < self.target_anchor < 1):
            raise ValueError("Both anchors must lie in (0, 1).")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be finite and positive.")
        if (self.log_generator is None) != (self.inverse_log_generator is None):
            raise ValueError("Supply both custom generator functions.")

    def _transform(self, value, source, target, power):
        value = _probability(value)
        if self.log_generator is None:
            return expit(logit(target) + power * (logit(value) - logit(source)))
        result = np.array(value, copy=True)
        interior = (value > 0) & (value < 1)
        log_g = self.log_generator
        result[interior] = self.inverse_log_generator(
            log_g(target) + power * (log_g(value[interior]) - log_g(source))
        )
        return result

    def __call__(self, value):
        return self._transform(value, self.source_anchor, self.target_anchor, self.power)

    def inverse(self, value):
        return self._transform(value, self.target_anchor, self.source_anchor, 1 / self.power)

    def derivative(self, value):
        value = _probability(value)
        if np.any((value == 0) | (value == 1)):
            raise ValueError("Evaluate the derivative in the open unit interval.")
        if self.log_generator is not None:
            dlog_g = self.log_generator_derivative
            if dlog_g is None:
                raise ValueError("Density evaluation requires a log-generator derivative.")
            return self.power * dlog_g(value) / dlog_g(self(value))
        mapped_logit = logit(self.target_anchor) + self.power * (
            logit(value) - logit(self.source_anchor)
        )
        log_numerator = -np.logaddexp(0, mapped_logit) - np.logaddexp(0, -mapped_logit)
        return self.power * np.exp(log_numerator - np.log(value) - np.log1p(-value))


@dataclass(frozen=True)
class TerminalTransport:
    """The law with CDF Q(x) = transport(P(x)) for a continuous baseline law."""

    physical: object
    transport: DombiKappa

    def cdf(self, value):
        return self.transport(self.physical.cdf(value))

    def ppf(self, probability):
        return self.physical.ppf(self.transport.inverse(probability))

    def _odds_log_ratio(self, value):
        log_u = np.asarray(self.physical.logcdf(value))
        log_complement = np.asarray(self.physical.logsf(value))
        power = self.transport.power
        intercept = logit(self.transport.target_anchor) - power * logit(self.transport.source_anchor)
        mapped_logit = power * (log_u - log_complement) + intercept
        with np.errstate(invalid="ignore"):
            return (np.log(power) - np.logaddexp(0, mapped_logit)
                    - np.logaddexp(0, -mapped_logit) - log_u - log_complement)

    def _can_use_odds_tails(self):
        return self.transport.log_generator is None and all(
            hasattr(self.physical, method) for method in ("logcdf", "logsf", "logpdf")
        )

    def density_ratio(self, value):
        if self._can_use_odds_tails():
            return np.exp(self._odds_log_ratio(value))
        return self.transport.derivative(self.physical.cdf(value))

    def pdf(self, value):
        if self._can_use_odds_tails():
            log_pdf = np.asarray(self.physical.logpdf(value))
            with np.errstate(invalid="ignore"):
                combined = log_pdf + self._odds_log_ratio(value)
            return np.exp(np.where(np.isneginf(log_pdf), -np.inf, combined))
        probabilities = np.asarray(self.physical.cdf(value))
        densities = np.asarray(self.physical.pdf(value))
        interior = (probabilities > 0) & (probabilities < 1)
        result = np.where(densities == 0, 0.0, np.nan)
        result[interior] = densities[interior] * self.transport.derivative(probabilities[interior])
        return result

    def sample(self, size=None, random_state=None):
        return self.ppf(np.random.default_rng(random_state).random(size))


def terminal_cdf_transport(physical, target, probability):
    """Evaluate the exact quantile transport F_target(F_physical^{-1}(u))."""
    return target.cdf(physical.ppf(_probability(probability)))


def esscher_weights(log_returns, theta, physical_weights=None):
    """Return normalized Esscher probabilities on finite terminal scenarios."""
    returns = np.asarray(log_returns, dtype=float)
    if returns.ndim != 1 or not returns.size or not np.all(np.isfinite(returns)):
        raise ValueError("log_returns must be a finite nonempty vector.")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite.")
    weights = np.ones_like(returns) if physical_weights is None else np.asarray(
        physical_weights, dtype=float
    )
    if (weights.shape != returns.shape or np.any(weights < 0)
            or not np.all(np.isfinite(weights)) or not np.any(weights > 0)):
        raise ValueError("Supply aligned nonnegative weights with positive mass.")
    with np.errstate(divide="ignore"):
        logits = np.log(weights) + theta * returns
    if not np.isfinite(logsumexp(logits)):
        raise ValueError("Esscher log-weights overflowed.")
    return np.exp(logits - logsumexp(logits))


def calibrate_esscher(log_returns, physical_weights=None, *, gross_target=1.0):
    """Fit theta so the tilted expected gross return equals gross_target."""
    if not np.isfinite(gross_target) or gross_target <= 0:
        raise ValueError("gross_target must be finite and positive.")
    returns = np.asarray(log_returns, dtype=float)
    weights = esscher_weights(returns, 0, physical_weights)
    active = weights > 0
    support = returns[active]
    target = np.log(gross_target)
    if np.all(support == target):
        return 0.0, weights
    if not support.min() < target < support.max():
        raise ValueError("A finite Esscher tilt requires the target inside the gross-return support.")
    span = support.max() - support.min()
    # Center and scale the likelihood calculation to make bracketing unitless.
    centered = (support - target) / span
    base = np.log(weights[active])

    def gap(scaled_theta):
        tilted = base + scaled_theta * centered
        return logsumexp(tilted + support - target) - logsumexp(tilted)

    bound = 1.
    for _ in range(80):
        if gap(-bound) <= 0 <= gap(bound):
            theta = brentq(gap, -bound, bound) / span
            return theta, esscher_weights(returns, theta, physical_weights)
        bound *= 2
    raise ValueError("Unable to bracket the Esscher tilt for these scenarios.")


def calibrate_esscher_theta(log_mgf, bracket, *, gross_target=1.0):
    """Find an Esscher parameter from a log-MGF valid at theta and theta + 1."""
    if not np.isfinite(gross_target) or gross_target <= 0:
        raise ValueError("gross_target must be finite and positive.")
    left, right = bracket
    if not np.isfinite(left) or not np.isfinite(right) or not left < right:
        raise ValueError("Require a finite increasing bracket.")
    log_target = np.log(gross_target)

    def gap(theta):
        base, shifted = log_mgf(theta), log_mgf(theta + 1)
        if not np.isfinite(base) or not np.isfinite(shifted):
            raise ValueError("The log-MGF must be finite at theta and theta + 1.")
        return shifted - base - log_target

    return brentq(gap, left, right)
