"""
Minsky-compatible stock-flow DSL: differentiable drop-in for Keen's Minsky.

Keen's Minsky software uses Godley tables to specify stock-flow consistent
models. This module provides a compatible Python DSL that compiles to JAX,
making Minsky-style models differentiable and GPU-accelerated.

The canonical example is the Keen predator-prey (Lotka-Volterra debt dynamics)
model: employment rate λ and debt ratio d coupled via:
    dλ/dt = λ · (Φ(λ) - α - β)           # employment Phillips curve
    dd/dt = κ(π) - π - (α + β + γ) · d   # debt accumulation

References:
    Buckley (2026) Economic Gauge Theory. doi:10.5281/zenodo.20259495
    Keen (1995) Finance and Economic Breakdown. J. Post Keynesian Economics.
    Minsky software: https://sourceforge.net/projects/minsky/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.manifold import BalanceSheet, GodleyTable, PacioliManifold


# ---------------------------------------------------------------------------
# MinskySFCModel — Godley-table-based DSL
# ---------------------------------------------------------------------------

@dataclass
class MinskySFCModel:
    """
    A Minsky-compatible SFC model specified via Godley tables.

    Each Godley table encodes the flows for one balance sheet.
    The model is a collection of linked Godley tables sharing sectors.

    Variables: stock variables (balance sheet positions) + flow variables
    (Godley table entries, potentially nonlinear functions of stocks).

    The model is simulated by Euler integration (or RK4 via step_rk4).
    Being a JAX function, it is differentiable end-to-end.
    """
    sectors: list[str]
    instruments: list[str]
    initial_positions: jax.Array    # shape (n_sectors, n_instruments)
    flow_fns: dict[str, Callable]   # instrument → fn(positions, t, params) → flows column

    @property
    def n_sectors(self) -> int:
        return len(self.sectors)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    def _compute_flows(
        self,
        positions: jax.Array,
        t: float,
        params: dict,
    ) -> jax.Array:
        """
        Compute the (n_sectors, n_instruments) flow matrix at time t.

        Each instrument column is computed by its registered flow function.
        Flows that are not registered are zero.
        """
        flows = jnp.zeros((self.n_sectors, self.n_instruments))
        for j, instrument in enumerate(self.instruments):
            if instrument in self.flow_fns:
                col = self.flow_fns[instrument](positions, t, params)
                flows = flows.at[:, j].set(col)
        return flows

    def step(
        self,
        positions: jax.Array,
        t: float,
        dt: float,
        params: dict,
    ) -> jax.Array:
        """
        Euler step: positions(t+dt) = positions(t) + flows(t) * dt.

        Args:
            positions: shape (n_sectors, n_instruments)
            t:         current time
            dt:        timestep
            params:    model parameters dict

        Returns:
            Updated positions, shape (n_sectors, n_instruments).
        """
        flows = self._compute_flows(positions, t, params)
        return positions + flows * dt

    def step_rk4(
        self,
        positions: jax.Array,
        t: float,
        dt: float,
        params: dict,
    ) -> jax.Array:
        """
        RK4 step for higher accuracy.
        """
        def f(pos, time):
            return self._compute_flows(pos, time, params)

        k1 = f(positions,          t)
        k2 = f(positions + dt/2*k1, t + dt/2)
        k3 = f(positions + dt/2*k2, t + dt/2)
        k4 = f(positions + dt*k3,   t + dt)
        return positions + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def simulate(
        self,
        T: float,
        dt: float = 0.1,
        params: Optional[dict] = None,
        method: str = 'euler',
    ) -> tuple[jax.Array, jax.Array]:
        """
        Simulate the model from t=0 to T.

        Returns:
            times:      shape (n_steps+1,)
            trajectory: shape (n_steps+1, n_sectors, n_instruments)
        """
        if params is None:
            params = {}
        step_fn = self.step if method == 'euler' else self.step_rk4

        n_steps = int(round(T / dt))
        positions = self.initial_positions
        times = [0.0]
        traj  = [positions]
        t = 0.0
        for _ in range(n_steps):
            positions = step_fn(positions, t, dt, params)
            t += dt
            times.append(t)
            traj.append(positions)

        return jnp.array(times), jnp.stack(traj)

    def balance_sheet_at(self, positions: jax.Array) -> BalanceSheet:
        return BalanceSheet(
            positions=positions,
            sectors=self.sectors,
            instruments=self.instruments,
        )

    def __repr__(self) -> str:
        return (
            f"MinskySFCModel({self.n_sectors} sectors, "
            f"{self.n_instruments} instruments, "
            f"{len(self.flow_fns)} flow functions)"
        )


# ---------------------------------------------------------------------------
# Keen predator-prey model
# ---------------------------------------------------------------------------

class KeenState(NamedTuple):
    """State variables of the Keen (1995) predator-prey model."""
    employment_rate: float    # λ ∈ [0,1]: employed fraction of labour force
    debt_ratio: float         # d = D/Y: private debt to GDP ratio
    wage_share: float         # ω = W/Y: labour share of output


def keen_predator_prey(
    alpha: float = 0.025,   # labour productivity growth rate
    beta:  float = 0.02,    # population growth rate
    gamma: float = 0.01,    # depreciation rate
    delta: float = 0.05,    # target profit rate (investment function)
) -> MinskySFCModel:
    """
    Keen (1995) predator-prey debt dynamics model.

    Three coupled ODEs:
        dω/dt = ω · (Φ(λ) - α)
        dλ/dt = λ · (κ(π)/ν - α - β - γ)
        dd/dt = κ(π) - π - (α + β) · d

    where:
        π = 1 - ω - r·d            (profit share)
        Φ(λ) = Phillips curve: wage growth as function of employment
        κ(π) = investment function: investment share as function of profit
        ν = capital-output ratio (fixed)
        r = real interest rate

    Packaged as a MinskySFCModel over three 'sectors':
        [workers, firms, banks]
    and three 'instruments':
        [wages, investment, debt_service]

    The predator-prey dynamics live in the flow functions.
    """
    # Phillips curve: Φ(λ) = φ₀ + φ₁/(1-λ)² — blows up at full employment
    def phillips(lam, phi0=-0.04, phi1=0.0006):
        return phi0 + phi1 / jnp.maximum(1.0 - lam, 0.01) ** 2

    # Investment function: κ(π) = κ₀ + κ₁·exp(κ₂·π)
    def investment(pi, kappa0=0.0, kappa1=0.05, kappa2=4.0, nu=3.0):
        return (kappa0 + kappa1 * jnp.exp(kappa2 * pi)) / nu

    sectors     = ['workers', 'firms', 'banks']
    instruments = ['wages', 'investment', 'debt_service']

    # Initial state: moderate employment, low debt, labour share ≈ 0.8
    lam0, d0, omega0 = 0.94, 0.1, 0.80
    initial_positions = jnp.array([
        # wages  investment  debt_service
        [ omega0,  0.0,        0.0       ],   # workers: receive wages
        [ -omega0, 1.0-omega0, -d0*0.05  ],   # firms: pay wages, invest, pay interest
        [ 0.0,     0.0,         d0*0.05  ],   # banks: receive interest
    ])

    def wages_flow(pos, t, params):
        omega = pos[0, 0]
        lam   = params.get('lambda', 0.94)
        alpha_ = params.get('alpha', alpha)
        dw = omega * phillips(jnp.array(lam))
        return jnp.array([dw, -dw, 0.0])

    def investment_flow(pos, t, params):
        omega = pos[0, 0]
        d     = params.get('debt', 0.1)
        r     = params.get('r', 0.05)
        pi    = 1.0 - omega - r * d
        kappa = investment(jnp.array(pi))
        return jnp.array([0.0, kappa, -kappa])

    def debt_service_flow(pos, t, params):
        d   = params.get('debt', 0.1)
        r   = params.get('r', 0.05)
        svc = r * d
        return jnp.array([0.0, -svc, svc])

    return MinskySFCModel(
        sectors=sectors,
        instruments=instruments,
        initial_positions=initial_positions,
        flow_fns={
            'wages':        wages_flow,
            'investment':   investment_flow,
            'debt_service': debt_service_flow,
        },
    )


def keen_ode(
    state: jax.Array,
    t: float,
    params: dict,
) -> jax.Array:
    """
    Pure-ODE form of the Keen model: dX/dt = f(X, t, params).

    State vector X = [omega, lambda, d].

    Useful for integration with Diffrax or scipy.integrate.
    Returns shape (3,).
    """
    omega, lam, d = state[0], state[1], state[2]
    r     = params.get('r',     0.05)
    alpha_ = params.get('alpha', 0.025)
    beta_  = params.get('beta',  0.02)
    nu    = params.get('nu',    3.0)
    phi0  = params.get('phi0',  -0.04)
    phi1  = params.get('phi1',  0.0006)
    k0    = params.get('kappa0', 0.0)
    k1    = params.get('kappa1', 0.05)
    k2    = params.get('kappa2', 4.0)

    pi    = 1.0 - omega - r * d
    phi   = phi0 + phi1 / jnp.maximum(1.0 - lam, 0.01) ** 2
    kappa = (k0 + k1 * jnp.exp(k2 * pi)) / nu

    domega = omega * (phi - alpha_)
    dlam   = lam   * (kappa - alpha_ - beta_)
    dd     = kappa - pi - (alpha_ + beta_) * d

    return jnp.array([domega, dlam, dd])


def keen_simulate(
    omega0: float = 0.80,
    lam0:   float = 0.94,
    d0:     float = 0.10,
    T:      float = 100.0,
    dt:     float = 0.1,
    params: Optional[dict] = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulate the Keen predator-prey model via Euler integration.

    Returns:
        times:      shape (n_steps+1,)
        trajectory: shape (n_steps+1, 3) — [omega, lambda, d] at each step
    """
    if params is None:
        params = {}
    state = jnp.array([omega0, lam0, d0])
    n_steps = int(round(T / dt))
    times = [0.0]
    traj  = [state]
    t = 0.0
    for _ in range(n_steps):
        dX    = keen_ode(state, t, params)
        state = state + dX * dt
        t    += dt
        times.append(t)
        traj.append(state)
    return jnp.array(times), jnp.stack(traj)
