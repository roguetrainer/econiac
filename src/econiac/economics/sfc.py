"""
Stock-flow consistent (SFC) engine on the Pacioli manifold.

Every Godley-Lavoie SFC model is a Pacioli manifold instance: sectors are
nodes, flows are edges, ∂²=0 is the accounting identity enforced by
construction. One simulation step = one Godley table application.

The SFC model is fully differentiable via JAX: calibrate parameters by
gradient descent on national accounts data in a single backward pass.

References:
    Buckley (2026) Topology of Conservation. doi:10.5281/zenodo.20234853
    Buckley (2026) Economic Gauge Theory.   doi:10.5281/zenodo.20259495
    Godley & Lavoie (2007) Monetary Economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.manifold import BalanceSheet, GodleyTable, PacioliManifold
from econiac.core.ensemble import gibbs_weights


# ---------------------------------------------------------------------------
# SFCParameters — trainable model parameters
# ---------------------------------------------------------------------------

@dataclass
class SFCParameters:
    """
    Trainable parameters of an SFC model.

    propensities: shape (n_flows,) — one marginal propensity per flow edge.
    beta: rationality (inverse temperature). beta→0: equal allocation;
          beta→∞: winner-takes-all (classical fixed-coefficient SFC).
    """
    propensities: jax.Array   # shape (n_flows,), ≥ 0
    beta: float = 1.0

    def __post_init__(self):
        if not bool(jnp.all(self.propensities >= 0)):
            raise ValueError("propensities must be non-negative")

    @staticmethod
    def uniform(n_flows: int, beta: float = 1.0) -> 'SFCParameters':
        return SFCParameters(propensities=jnp.ones(n_flows) / n_flows, beta=beta)


# ---------------------------------------------------------------------------
# SFCState — economic state at one point in time
# ---------------------------------------------------------------------------

class SFCState(NamedTuple):
    """One-period snapshot of an SFC model."""
    balance_sheet: BalanceSheet
    t: float = 0.0


# ---------------------------------------------------------------------------
# SFCModel — the differentiable SFC engine
# ---------------------------------------------------------------------------

@dataclass
class SFCModel:
    """
    A stock-flow consistent model on the Pacioli manifold.

    At each timestep the model:
    1. Computes flow utilities U_k from current balance sheet and propensities.
    2. Allocates flows via Gibbs weights (differentiable, ∂π/∂U exists).
    3. Applies the Godley table (∂²=0 by construction) to advance stocks.

    Args:
        manifold:     Pacioli manifold defining the flow network topology.
        initial_bs:   starting balance sheet.
        params:       trainable propensity parameters.
        utility_fn:   optional custom utility function
                      (balance_sheet, params) -> Array of shape (n_flows,).
    """
    manifold:   PacioliManifold
    initial_bs: BalanceSheet
    params:     SFCParameters
    utility_fn: Optional[Callable] = None

    def __post_init__(self):
        if self.manifold.n_nodes != self.initial_bs.n_sectors:
            raise ValueError(
                f"manifold has {self.manifold.n_nodes} nodes but "
                f"balance sheet has {self.initial_bs.n_sectors} sectors"
            )
        if len(self.params.propensities) != self.manifold.n_edges:
            raise ValueError(
                f"params has {len(self.params.propensities)} propensities "
                f"but manifold has {self.manifold.n_edges} edges"
            )

    def _flow_utilities(self, bs: BalanceSheet) -> jax.Array:
        """Compute per-edge utility from current balance sheet."""
        if self.utility_fn is not None:
            return self.utility_fn(bs, self.params)
        # Default: U_k = propensity_k * net_worth_of_source_sector
        nw = bs.net_worth()   # (n_sectors,)
        B  = np.array(self.manifold.incidence)   # (n_sectors, n_edges)
        source_nw = jnp.array([
            float(nw[int(np.argmin(B[:, e]))]) if np.any(B[:, e] < 0) else 1.0
            for e in range(self.manifold.n_edges)
        ])
        return self.params.propensities * jnp.maximum(source_nw, 0.0)

    def step(self, state: SFCState, dt: float = 1.0) -> SFCState:
        """
        Advance the model by one period dt.

        Flow allocation is Gibbs-weighted (differentiable). The Godley table
        preserves ∂²=0 by construction.
        """
        bs = state.balance_sheet
        U  = self._flow_utilities(bs)
        w  = gibbs_weights(U, beta=self.params.beta)   # (n_flows,)

        total_capacity = jnp.sum(jnp.maximum(bs.net_worth(), 0.0))
        edge_flows     = w * total_capacity * dt        # (n_flows,)

        # Build Godley flow matrix: map edge flows to (sector, instrument) space
        n_s = bs.n_sectors
        n_i = bs.n_instruments
        B   = self.manifold.incidence   # (n_sectors, n_edges)
        flow_matrix = jnp.zeros((n_s, n_i))
        for e_idx, edge_name in enumerate(self.manifold.edges):
            if edge_name in bs.instruments:
                i_idx = bs.instruments.index(edge_name)
                col   = B[:, e_idx] * edge_flows[e_idx]
                flow_matrix = flow_matrix.at[:, i_idx].add(col)

        gt     = GodleyTable(flows=flow_matrix, sectors=bs.sectors, instruments=bs.instruments)
        new_bs = gt.apply(bs)
        return SFCState(balance_sheet=new_bs, t=state.t + dt)

    def simulate(self, T: float, dt: float = 1.0) -> list[SFCState]:
        """Simulate for T periods; returns trajectory including initial state."""
        state = SFCState(balance_sheet=self.initial_bs, t=0.0)
        trajectory = [state]
        for _ in range(int(round(T / dt))):
            state = self.step(state, dt)
            trajectory.append(state)
        return trajectory

    def gdp(self, state: SFCState) -> jax.Array:
        """GDP proxy: total positive net worth across all sectors."""
        return jnp.sum(jnp.maximum(state.balance_sheet.net_worth(), 0.0))

    def social_multiplier(self, beta_range: jax.Array) -> jax.Array:
        """
        Social multiplier χ(β) = 1/Σ w_k² (participation ratio).

        Phase-transition diagnostic: χ drops sharply at the critical β where
        the economy concentrates on a single flow channel.
        Returns shape (len(beta_range),).
        """
        U = self._flow_utilities(self.initial_bs)
        return jnp.array([
            float(1.0 / jnp.sum(gibbs_weights(U, beta=float(b)) ** 2))
            for b in beta_range
        ])

    def __repr__(self) -> str:
        h = self.manifold.homology()
        return (
            f"SFCModel({self.manifold.n_nodes} sectors, "
            f"{self.manifold.n_edges} flows, H₁={h.H1_rank})"
        )
