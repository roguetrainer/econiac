"""
Thermodynamic Information Routing (TIR): universal Gibbs routing primitive.

Four axioms: candidates, admissibility geometry, β, Gibbs output.
The Gibbs weights are the unique routing primitive preserving:
  (i)  conformal invariance — scale-free in utility units
  (ii) symplectic structure — Hamiltonian flow, no information dissipation
  (iii) adiabatic invariance — β-schedule tracks free energy minimum

Eight independent rediscoveries: McFadden, Sims, McKelvey-Palfrey, Jaynes,
Gibbs, Maslov, Goel, Friston — all derived the same theorem.

Reference: Buckley (2026) TIR, doi:10.5281/zenodo.20237288
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.ensemble import gibbs_weights
from econiac.core.geometry import Geometry, AbelianGeometry, GeometryType, NEG_INF


# ---------------------------------------------------------------------------
# TIRInstance — the universal routing primitive
# ---------------------------------------------------------------------------

@dataclass
class TIRInstance:
    """
    One TIR routing problem: candidates × utilities × geometry × beta.

    candidates: list of n candidate labels (strings or ints)
    utilities:  shape (n,) — utility of each candidate
    geometry:   admissibility filter (Abelian, Fano, G2, Catalan)
    beta:       inverse temperature; β→0 is uniform, β→∞ is argmax
    geometry_kwargs: extra keyword args passed to geometry.mask()
    """
    candidates:       list
    utilities:        jax.Array
    geometry:         Geometry
    beta:             float = 1.0
    geometry_kwargs:  dict  = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.candidates)
        if self.utilities.shape != (n,):
            raise ValueError(
                f"utilities shape {self.utilities.shape} != ({n},) = len(candidates)"
            )
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def __repr__(self) -> str:
        return (
            f"TIRInstance({self.n_candidates} candidates, "
            f"geometry={self.geometry}, β={self.beta:.3f})"
        )


# ---------------------------------------------------------------------------
# route — main entry point
# ---------------------------------------------------------------------------

def route(tir: TIRInstance) -> jax.Array:
    """
    Compute Gibbs routing weights for a TIRInstance.

    Applies the geometry mask (setting inadmissible candidates to -∞),
    then computes Gibbs weights: w_i = softmax(β · U_i).

    Returns:
        shape (n,) — routing probabilities summing to 1.
        Inadmissible candidates receive weight 0.
    """
    U_masked = tir.geometry.mask(tir.utilities, **tir.geometry_kwargs)
    return gibbs_weights(U_masked, beta=tir.beta)


# ---------------------------------------------------------------------------
# free_energy — thermodynamic diagnostic
# ---------------------------------------------------------------------------

def free_energy(utilities: jax.Array, beta: float) -> jax.Array:
    """
    Gibbs free energy: F(β) = -β⁻¹ · log Σ exp(β · U_i).

    At β→0: F → -β⁻¹ · log(n)  (entropy-dominated, uniform routing).
    At β→∞: F → -max(U)         (utility-dominated, greedy routing).

    The free energy is the generating function for all moments of the
    routing distribution: ∂F/∂U_i = -w_i (negative routing weight).

    Args:
        utilities: shape (n,) — may include NEG_INF for masked candidates
        beta:      inverse temperature (≥ 0)

    Returns:
        scalar free energy F(β).
    """
    if beta == 0:
        n_admissible = jnp.sum(utilities > NEG_INF / 2).astype(jnp.float32)
        return -jnp.log(jnp.maximum(n_admissible, 1.0))
    log_Z = jax.nn.logsumexp(beta * utilities)
    return -log_Z / beta


# ---------------------------------------------------------------------------
# escape_arrow — does this instance escape Arrow's Impossibility Theorem?
# ---------------------------------------------------------------------------

def escape_arrow(tir: TIRInstance) -> bool:
    """
    Does this TIRInstance escape Arrow's Impossibility Theorem?

    Arrow's theorem applies to deterministic, transitive, rank-based voting.
    TIR escapes because:
      1. Stochastic (Gibbs weights ≠ deterministic rank aggregation)
      2. Cardinal (utilities, not ordinal rankings)
      3. Non-transitive admissibility (Fano, G2, Catalan geometries)

    Returns True if the instance has any non-Abelian geometry feature that
    breaks Arrow's transitivity assumption, OR if beta < ∞ (stochastic).

    Formally:
      - Any β < ∞: stochastic → escapes (Independence of Irrelevant Alternatives fails)
      - Fano/G2/Catalan geometry: non-transitive admissibility → escapes
      - Abelian + β→∞: deterministic, transitive → Arrow applies (returns False)
    """
    if tir.beta < 1e8:
        return True
    return tir.geometry.geometry_type != GeometryType.ABELIAN


# ---------------------------------------------------------------------------
# admissible_count — how many candidates are admissible?
# ---------------------------------------------------------------------------

def admissible_count(tir: TIRInstance) -> int:
    """Number of admissible candidates under the geometry."""
    U_masked = tir.geometry.mask(tir.utilities, **tir.geometry_kwargs)
    return int(jnp.sum(U_masked > NEG_INF / 2))


# ---------------------------------------------------------------------------
# entropy — routing entropy H(w) = -Σ w_i log w_i
# ---------------------------------------------------------------------------

def routing_entropy(tir: TIRInstance) -> jax.Array:
    """
    Shannon entropy of the routing distribution.

    H = 0 at β→∞ (deterministic routing to argmax).
    H = log(n_admissible) at β=0 (uniform routing).

    Returns scalar entropy in nats.
    """
    w = route(tir)
    safe_w = jnp.where(w > 1e-30, w, 1e-30)
    return -jnp.sum(w * jnp.log(safe_w))


# ---------------------------------------------------------------------------
# social_multiplier — participation ratio χ(β) = 1/Σw²
# ---------------------------------------------------------------------------

def social_multiplier(tir: TIRInstance) -> jax.Array:
    """
    Social multiplier (participation ratio) χ(β) = 1 / Σ_i w_i².

    χ = 1 at β→∞ (winner-take-all: one agent gets all weight).
    χ = n at β=0 (equal participation: all weights equal 1/n).

    Phase transition: χ drops sharply as β crosses a critical β*.
    This is the Gibbs equivalent of the replica symmetry breaking transition.

    Returns scalar participation ratio ∈ [1, n_admissible].
    """
    w = route(tir)
    return 1.0 / jnp.sum(w ** 2)


# ---------------------------------------------------------------------------
# TIR constructor helpers
# ---------------------------------------------------------------------------

def tir_from_scores(
    candidates: list,
    scores: list[float],
    beta: float = 1.0,
    n_nodes: Optional[int] = None,
) -> TIRInstance:
    """
    Convenience: build an Abelian TIRInstance from a list of scalar scores.

    Uses a complete graph (all candidates reachable from all sources).
    """
    n = len(candidates)
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    geom = AbelianGeometry(adjacency=adj)
    return TIRInstance(
        candidates=candidates,
        utilities=jnp.array(scores, dtype=jnp.float32),
        geometry=geom,
        beta=beta,
        geometry_kwargs={'source': 0},
    )
