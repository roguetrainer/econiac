"""
Statistical ensemble — Maslov-Gibbs partition function and routing weights.

The central object is the Gibbs distribution over a finite set of candidates
with utility scores U_i and inverse temperature β:

    Z(β)    = Σ_i exp(β · U_i)          partition function
    w_i(β)  = exp(β · U_i) / Z(β)       routing weights  (sum to 1)
    F(β)    = -β⁻¹ ln Z(β)              free energy

Limits:
    β → 0   uniform weights (maximum entropy, pure exploration)
    β → ∞   argmax weights  (tropical limit, pure exploitation)

The routing weights are the unique probability measure that minimises free
energy F(β) for given β, and simultaneously preserves conformal, symplectic,
and adiabatic invariance (Paper 201, Theorem 3.1).

References:
    Buckley (2026) MGE.  doi:10.5281/zenodo.17981393
    Buckley (2026) TIR.  doi:10.5281/zenodo.20237288
    Smith (2015) Information equilibrium.  arXiv:1510.02435
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import NamedTuple, Callable


# ---------------------------------------------------------------------------
# Core functions — pure, JIT-compilable
# ---------------------------------------------------------------------------

@jit
def partition_function(utilities: jax.Array, beta: float) -> jax.Array:
    """Z(β) = Σ exp(β · U_i), computed in log-space for numerical stability."""
    return jnp.sum(jnp.exp(beta * utilities))


@jit
def log_partition(utilities: jax.Array, beta: float) -> jax.Array:
    """ln Z(β) via log-sum-exp for numerical stability."""
    return jax.nn.logsumexp(beta * utilities)


@jit
def gibbs_weights(utilities: jax.Array, beta: float) -> jax.Array:
    """
    Gibbs routing weights w_i(β) = exp(β·U_i) / Z(β).

    Uses softmax for numerical stability. At β=0 returns uniform weights.
    At β→∞ concentrates on the argmax.

    Args:
        utilities: shape (n,) array of utility scores U_i
        beta:      inverse temperature β ≥ 0

    Returns:
        weights: shape (n,) array summing to 1
    """
    return jax.nn.softmax(beta * utilities)


@jit
def free_energy(utilities: jax.Array, beta: float) -> jax.Array:
    """
    TIR free energy F(β) = -β⁻¹ ln Z(β).

    At β=0, F = -ln(n)  (maximum entropy baseline).
    As β→∞, F → -max(U_i)  (ground state energy).

    Args:
        utilities: shape (n,) array of utility scores
        beta:      inverse temperature β > 0

    Returns:
        scalar free energy
    """
    return -log_partition(utilities, beta) / beta


@jit
def entropy(utilities: jax.Array, beta: float) -> jax.Array:
    """
    Shannon entropy of the Gibbs distribution H = -Σ w_i ln w_i.

    Equals ln(n) at β=0, approaches 0 as β→∞.
    """
    w = gibbs_weights(utilities, beta)
    return -jnp.sum(jnp.where(w > 0, w * jnp.log(w), 0.0))


@jit
def mean_utility(utilities: jax.Array, beta: float) -> jax.Array:
    """⟨U⟩ = Σ w_i U_i — the thermodynamic internal energy."""
    w = gibbs_weights(utilities, beta)
    return jnp.dot(w, utilities)


def choose(beta: float, candidates: jax.Array, utilities: jax.Array) -> jax.Array:
    """
    MGE-weighted combination: returns Σ w_i(β) · candidates_i.

    For scalar candidates this is the Gibbs-weighted average.
    For vector candidates (e.g. policy vectors, portfolio weights) this
    is the Gibbs mixture — a convex combination of the candidate vectors.

    Args:
        beta:        inverse temperature β ≥ 0
        candidates:  shape (n,) or (n, d) — the objects to mix
        utilities:   shape (n,) — utility score for each candidate

    Returns:
        shape () or (d,) — the Gibbs-weighted mixture
    """
    w = gibbs_weights(utilities, beta)
    if candidates.ndim == 1:
        return jnp.dot(w, candidates)
    return jnp.einsum('i,i...->...', w, candidates)


# ---------------------------------------------------------------------------
# β-schedules
# ---------------------------------------------------------------------------

def beta_schedule(
    beta_0: float,
    beta_final: float,
    n_steps: int,
    kind: str = 'geometric',
) -> jax.Array:
    """
    Generate a β-schedule from β_0 (exploration) to β_final (exploitation).

    Args:
        beta_0:      starting inverse temperature (low, e.g. 0.1)
        beta_final:  final inverse temperature (high, e.g. 100.0)
        n_steps:     number of steps
        kind:        'linear' | 'geometric' | 'cosine'
                     geometric is preferred — equal ratio steps preserve
                     adiabatic invariance better than equal additive steps.

    Returns:
        shape (n_steps,) array of β values
    """
    if kind == 'linear':
        return jnp.linspace(beta_0, beta_final, n_steps)
    elif kind == 'geometric':
        return jnp.exp(jnp.linspace(jnp.log(beta_0), jnp.log(beta_final), n_steps))
    elif kind == 'cosine':
        # cosine annealing: slow start and end, fast middle
        t = jnp.linspace(0.0, jnp.pi, n_steps)
        frac = 0.5 * (1 - jnp.cos(t))
        return beta_0 + frac * (beta_final - beta_0)
    else:
        raise ValueError(f"Unknown schedule kind: {kind!r}. Use 'linear', 'geometric', or 'cosine'.")


# ---------------------------------------------------------------------------
# Ensemble sweep — apply gibbs_weights across a β-schedule
# ---------------------------------------------------------------------------

def ensemble_sweep(
    utilities: jax.Array,
    schedule: jax.Array,
) -> jax.Array:
    """
    Compute Gibbs weights at every β in a schedule.

    Args:
        utilities: shape (n,)
        schedule:  shape (T,) β values

    Returns:
        weights: shape (T, n)  — weights[t] = gibbs_weights(utilities, schedule[t])
    """
    return vmap(lambda beta: gibbs_weights(utilities, beta))(schedule)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class EnsembleSummary(NamedTuple):
    """Summary statistics of a Gibbs ensemble at a given β."""
    beta: float
    weights: jax.Array      # shape (n,)
    free_energy: float
    entropy: float
    mean_utility: float
    effective_n: float      # exp(entropy) — effective number of active candidates


def summarise(utilities: jax.Array, beta: float) -> EnsembleSummary:
    """Compute all summary statistics for a Gibbs ensemble at β."""
    w  = gibbs_weights(utilities, beta)
    F  = free_energy(utilities, beta)
    H  = entropy(utilities, beta)
    U  = mean_utility(utilities, beta)
    return EnsembleSummary(
        beta=beta,
        weights=w,
        free_energy=float(F),
        entropy=float(H),
        mean_utility=float(U),
        effective_n=float(jnp.exp(H)),
    )
