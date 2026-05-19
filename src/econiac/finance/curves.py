"""
Yield curves as temporal connections on the Pacioli manifold.

The forward rate f(t,T) is the connection coefficient on the time bundle.
Zero-coupon bond P(t,T) = exp(-∫_t^T f(t,u) du) is parallel transport.

Temporal flatness: P(t,T)·P(T,T') = P(t,T')  for all t ≤ T ≤ T'.
This is the no-arbitrage condition. The HJM drift condition is its
infinitesimal form: α(t,T) = σ(t,T)·∫_t^T σ(t,u) du.

References:
    Buckley (2026) Term Structure Bundles. doi:10.5281/zenodo.20244445
    Heath, Jarrow, Morton (1992). Bond pricing and the term structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class YieldCurve:
    """
    A yield curve as a temporal connection on the time bundle.

    Stores (maturity, discount_factor) pairs; interpolates log-linearly
    (= linear in zero rates) for continuous evaluation.

    P(0, T) = exp(-r(T)·T)  where r(T) is the continuously compounded zero rate.
    The temporal connection coefficient: f(0,T) = -d/dT log P(0,T).
    """
    maturities: jax.Array        # shape (n,), strictly increasing, > 0
    discount_factors: jax.Array  # shape (n,), in (0, 1]

    def __post_init__(self):
        n = len(self.maturities)
        if self.discount_factors.shape != (n,):
            raise ValueError(
                f"discount_factors shape {self.discount_factors.shape} != ({n},)"
            )
        if not bool(jnp.all(self.discount_factors > 0)):
            raise ValueError("discount_factors must be strictly positive")
        if not bool(jnp.all(self.discount_factors <= 1.0 + 1e-6)):
            raise ValueError("discount_factors must be ≤ 1 (positive interest rates)")

    @property
    def n(self) -> int:
        return len(self.maturities)

    @property
    def zero_rates(self) -> jax.Array:
        """Continuously compounded zero rates r(T) = -log P(0,T) / T."""
        return -jnp.log(self.discount_factors) / self.maturities

    def discount(self, T: float) -> jax.Array:
        """
        Zero-coupon bond price P(0, T): parallel transport from T to 0.

        Interpolates log-linearly in discount factors (= linear in zero rates).
        """
        log_df = np.log(np.array(self.discount_factors))
        mats   = np.array(self.maturities)
        return jnp.exp(jnp.array(float(np.interp(T, mats, log_df))))

    def forward_discount(self, T1: float, T2: float) -> jax.Array:
        """
        Forward discount factor P(T1, T2) = P(0,T2) / P(0,T1).

        Parallel transport from T2 back to T1 along the time axis.
        """
        return self.discount(T2) / self.discount(T1)

    def forward_rate(self, T1: float, T2: float) -> jax.Array:
        """
        Simply-compounded forward rate L(T1, T2).

        L = (P(0,T1)/P(0,T2) - 1) / (T2 - T1).

        The LMM (discrete) connection coefficient on the tenor lattice.
        """
        dt = T2 - T1
        if dt <= 0:
            raise ValueError(f"T2={T2} must be > T1={T1}")
        return (self.discount(T1) / self.discount(T2) - 1.0) / dt

    def instantaneous_forward(self, T: float, dT: float = 1e-4) -> jax.Array:
        """
        Instantaneous forward rate f(0, T) = -d/dT log P(0,T).

        Finite-difference approximation. The HJM connection coefficient.
        """
        return -(jnp.log(self.discount(T + dT / 2)) - jnp.log(self.discount(T - dT / 2))) / dT

    def temporal_curvature(self, T1: float, T2: float) -> jax.Array:
        """
        Temporal curvature F(0, T1, T2).

        F = log P(0,T1) + log P(T1,T2) - log P(0,T2).

        Zero iff the composition law P(0,T1)·P(T1,T2)=P(0,T2) holds (no-arbitrage).
        """
        return (
            jnp.log(self.discount(T1))
            + jnp.log(self.forward_discount(T1, T2))
            - jnp.log(self.discount(T2))
        )

    def is_flat(self, tenors: Optional[list] = None, atol: float = 1e-6) -> bool:
        """True iff temporal curvature is zero on all tenor pairs."""
        if tenors is None:
            tenors = np.array(self.maturities).tolist()
        for i, T1 in enumerate(tenors[:-1]):
            for T2 in tenors[i + 1:]:
                if abs(float(self.temporal_curvature(float(T1), float(T2)))) > atol:
                    return False
        return True

    @staticmethod
    def from_zero_rates(maturities: jax.Array, zero_rates: jax.Array) -> 'YieldCurve':
        """Construct from continuously compounded zero rates."""
        return YieldCurve(
            maturities=jnp.array(maturities),
            discount_factors=jnp.exp(-jnp.array(zero_rates) * jnp.array(maturities)),
        )

    @staticmethod
    def flat(rate: float, maturities: jax.Array) -> 'YieldCurve':
        """Flat yield curve: all tenors at the same zero rate."""
        mats = jnp.array(maturities)
        return YieldCurve(maturities=mats, discount_factors=jnp.exp(-rate * mats))

    def __repr__(self) -> str:
        return (
            f"YieldCurve({self.n} tenors, T_max={float(self.maturities[-1]):.1f}y, "
            f"r_short={float(self.zero_rates[0]):.2%}, r_long={float(self.zero_rates[-1]):.2%})"
        )


def hjm_drift(maturities: jax.Array, sigma: jax.Array) -> jax.Array:
    """
    HJM no-arbitrage drift: α(0,T) = σ(0,T) · ∫_0^T σ(0,u) du.

    The forward-rate model df = α dt + σ dW is arbitrage-free iff this holds
    (HJM Flatness Theorem). Equivalently: the temporal connection is flat under Q.

    Args:
        maturities: shape (n,) — maturity grid
        sigma:      shape (n,) — volatility σ(0,T) at each maturity

    Returns:
        shape (n,) — required arbitrage-free drift α(0,T)
    """
    mats   = jnp.array(maturities)
    sig    = jnp.array(sigma)
    dt     = jnp.diff(mats, prepend=mats[0])
    cumint = jnp.cumsum(sig * dt)    # ∫_0^T σ(0,u) du, trapezoidal
    return sig * cumint


def lmm_forward_rates(curve: YieldCurve, tenors: jax.Array) -> jax.Array:
    """
    LIBOR Market Model forward rates on a tenor lattice.

    L_k = L(T_k, T_{k+1}): simply-compounded forward rate for period k.
    These are the discrete connection coefficients on the tenor lattice.

    Returns shape (n,) for n+1 input tenors.
    """
    tenors_np = np.array(tenors)
    return jnp.array([
        float(curve.forward_rate(float(tenors_np[k]), float(tenors_np[k + 1])))
        for k in range(len(tenors_np) - 1)
    ])


def lmm_discrete_flatness(curve: YieldCurve, tenors: jax.Array) -> jax.Array:
    """
    Discrete flatness residuals of the LMM connection.

    F_k = log P(0,T_k) + log P(T_k,T_{k+1}) - log P(0,T_{k+1})

    All should be ≈ 0 for a self-consistent (flat) lattice connection.
    Returns shape (n,) for n+1 input tenors.
    """
    tenors_np = list(np.array(tenors))
    return jnp.array([
        float(curve.temporal_curvature(tenors_np[k], tenors_np[k + 1]))
        for k in range(len(tenors_np) - 1)
    ])
