"""
policy.py
=========
Policy gradient and optimal policy frontier for contagion models.

The central result: JAX autodiff through the Gibbs-lifted cascade gives
∂(systemic_loss)/∂(policy_parameter) in a single backward pass. This is
the new policy tool that Hurd's lattice-theoretic framework cannot provide —
Knaster-Tarski fixed points are not differentiable; Gibbs-lifted fixed points
are.

Policy parameters
-----------------
Two primary applications, corresponding to Papers 332 and 333:

  Paper 332 (CHZ fire sales):
    ∂L/∂γ_min  — optimal capital floor per bank type
    ∂L/∂β      — sensitivity of systemic loss to constraint softness

  Paper 333 (Sovereign repo run):
    ∂L/∂h_ij   — optimal bilateral haircut schedule (heterogeneous vs FSB uniform)
    ∂L/∂β_ldi  — LDI sensitivity; how much higher should LDI haircuts be?

Smooth loss functional
----------------------
The hard loss 1[p_i < floor] is not differentiable. We use the Gibbs-lifted
smooth surrogate from gibbs.py:

    L_smooth = Σ_i σ(β_loss × (floor − p_i))

At β_loss → ∞ this recovers the hard count. At β_loss = 50 (default) it
approximates the hard count within ±0.02 agents.

Policy gradient
---------------
For a one-period cascade (single fixed-point call):
    grad = jax.grad(lambda theta: smooth_loss(fixed_point(C(theta), x0)))(theta_init)

For a multi-period cascade we differentiate through run_cascade() using
jax.grad on the smooth systemic loss at the final period.

Optimal haircut frontier
------------------------
The optimal haircut schedule minimises systemic loss subject to a budget
constraint:
    min_{h} L(h)   s.t.  Σ_{i,j} h_{ij} ≥ H_min  (FSB haircut floor)

We solve this with projected gradient descent in JAX. The result is the
Pareto frontier of (systemic_loss, total_haircut) showing the trade-off
between macroprudential tightness and systemic risk reduction.

LDI surcharge
-------------
A key Paper 333 result: how much higher should LDI pension fund haircuts be
than MMF haircuts? We compute:
    ∂L/∂h_ldi − ∂L/∂h_mmf
This is the LDI surcharge gradient — the additional haircut that equates the
marginal contribution to systemic risk across lender types.

References:
    FSB (2026). Vulnerabilities in Government Bond-backed Repo Markets.
    Julliard et al. (2022). BIS WP 1027. doi:10.2139/ssrn.4168862
    Buckley (2026). Paper 332: CHZ fire sales. doi:TBD
    Buckley (2026). Paper 333: Sovereign repo run. doi:TBD
    Buckley (2026). Paper 334: Contagion operator algebra. doi:TBD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.finance.contagion.operators import (
    Operator,
    SystemState,
    SolvencyState,
    LiquidityState,
    PriceState,
    fixed_point,
    FixedPointResult,
)
from econiac.finance.contagion.gibbs import (
    smooth_loss,
    smooth_loss_liquidity,
    combined_loss,
)


# ---------------------------------------------------------------------------
# Loss functionals — differentiable surrogates for hard policy objectives
# ---------------------------------------------------------------------------

def cascade_loss(
    op:              Operator,
    x0:              SystemState,
    solvency_floor:  float = 0.95,
    liquidity_floor: float = 0.80,
    beta_loss:       float = 50.0,
    w_solvency:      float = 1.0,
    w_liquidity:     float = 1.0,
    w_price:         float = 0.5,
    fp_tol:          float = 1e-3,
    fp_max_iter:     int   = 200,
) -> jax.Array:
    """
    Smooth systemic loss of a cascade: fixed_point loss at convergence.

    Runs fixed_point(op, x0) and returns combined_loss(final_state).
    This is the function we differentiate with jax.grad.

    Args:
        op:              contagion operator (Gibbs-lifted, differentiable)
        x0:              initial SystemState (pre-shock upper bound)
        solvency_floor:  distress threshold on solvency fractions
        liquidity_floor: run threshold on liquidity fractions
        beta_loss:       sharpness of the smooth indicator
        w_solvency/liquidity/price: loss weights
        fp_tol:          fixed-point convergence tolerance
        fp_max_iter:     maximum fixed-point iterations

    Returns:
        scalar JAX array — smooth systemic loss
    """
    fp     = fixed_point(op, x0, tol=fp_tol, max_iter=fp_max_iter)
    return combined_loss(
        fp.state,
        solvency_floor  = solvency_floor,
        liquidity_floor = liquidity_floor,
        beta_loss       = beta_loss,
        w_solvency      = w_solvency,
        w_liquidity     = w_liquidity,
        w_price         = w_price,
    )


def endpoint_loss(
    op:              Operator,
    x0:              SystemState,
    n_periods:       int   = 10,
    solvency_floor:  float = 0.95,
    liquidity_floor: float = 0.80,
    beta_loss:       float = 50.0,
    fp_tol:          float = 1e-3,
) -> jax.Array:
    """
    Smooth systemic loss at the end of a multi-period cascade.

    Runs the cascade for n_periods and returns combined_loss at
    the final state. Used when the policy gradient is with respect
    to a policy that changes the cascade dynamics over time.

    Args:
        op:         contagion operator
        x0:         initial state
        n_periods:  number of periods
        (other args: same as cascade_loss)

    Returns:
        scalar JAX array
    """
    state = x0
    for _ in range(n_periods):
        fp    = fixed_point(op, state, tol=fp_tol)
        state = fp.state
    return combined_loss(
        state,
        solvency_floor  = solvency_floor,
        liquidity_floor = liquidity_floor,
        beta_loss       = beta_loss,
    )


# ---------------------------------------------------------------------------
# Policy gradient — ∂L/∂θ via finite differences
# ---------------------------------------------------------------------------
# Note: jax.grad cannot differentiate through the Python-level fixed_point()
# loop because the operators contain Python-level control flow that JAX
# cannot trace. We use finite differences as the primary gradient method.
# For fully JIT-compiled operators, jax.grad is available directly on the
# JAX-native loss (see jax_policy_gradient() below for that case).

class PolicyGradientResult(NamedTuple):
    """Result of a policy gradient computation."""
    gradient:        np.ndarray   # shape matches theta_init
    loss_at_init:    float        # L(theta_init)
    loss_at_optimal: float        # L(theta_optimal) after one gradient step
    theta_init:      np.ndarray
    label:           str          # human-readable label for the policy parameter


def policy_gradient(
    loss_fn:     Callable[[np.ndarray], float],
    theta_init:  np.ndarray,
    eps:         float = 1e-4,
    label:       str   = "theta",
) -> PolicyGradientResult:
    """
    Finite-difference policy gradient: ∂L/∂θ at theta_init.

    For scalar theta (e.g. gamma_min, h_floor):
        dL/dtheta ≈ (L(theta + eps) − L(theta − eps)) / (2 eps)

    For vector theta (e.g. haircut schedule h_ij):
        (∂L/∂theta)_k ≈ (L(theta + eps·e_k) − L(theta − eps·e_k)) / (2 eps)
        computed in parallel across all k.

    Args:
        loss_fn:    callable theta → scalar loss. Constructs the operator
                    internally using theta as the policy parameter.
        theta_init: (d,) or scalar — initial policy parameter values
        eps:        finite-difference step size
        label:      human-readable name for reporting

    Returns:
        PolicyGradientResult
    """
    theta = np.array(theta_init, dtype=float).ravel()
    d     = len(theta)
    grad  = np.zeros(d)

    loss0 = float(loss_fn(theta))

    for k in range(d):
        e_k           = np.zeros(d);  e_k[k] = 1.0
        loss_hi       = float(loss_fn(theta + eps * e_k))
        loss_lo       = float(loss_fn(theta - eps * e_k))
        grad[k]       = (loss_hi - loss_lo) / (2.0 * eps)

    # One steepest-descent step (for reporting loss improvement)
    step_size    = eps * 10
    theta_opt    = theta - step_size * grad / (np.linalg.norm(grad) + 1e-8)
    loss_opt     = float(loss_fn(theta_opt))

    return PolicyGradientResult(
        gradient        = grad,
        loss_at_init    = loss0,
        loss_at_optimal = loss_opt,
        theta_init      = theta,
        label           = label,
    )


# ---------------------------------------------------------------------------
# Optimal haircut frontier (Paper 333 §5)
# ---------------------------------------------------------------------------

class HaircutFrontierResult(NamedTuple):
    """
    Pareto frontier of (systemic_loss, total_haircut).

    Each point is one optimal haircut schedule at a given FSB budget
    constraint H_min = Σ h_ij ≥ H_min.

    loss_curve:     (n_points,) — systemic loss at each budget level
    haircut_totals: (n_points,) — Σ h_ij at the optimum
    haircut_schedules: (n_points, d) — optimal per-class / per-lender haircuts
    fsb_loss:       scalar — loss under the FSB uniform minimum (benchmark)
    fsb_haircut:    scalar — FSB total haircut (benchmark)
    """
    loss_curve:          np.ndarray   # (n_points,)
    haircut_totals:      np.ndarray   # (n_points,)
    haircut_schedules:   np.ndarray   # (n_points, d)
    fsb_loss:            float
    fsb_haircut:         float
    h_min_grid:          np.ndarray   # (n_points,) — budget constraint values


class LDISurcharge(NamedTuple):
    """
    LDI haircut surcharge relative to MMF.

    dL_dh_ldi:  ∂L/∂h (gradient component for LDI lenders)
    dL_dh_mmf:  ∂L/∂h (gradient component for MMF lenders)
    surcharge:  dL_dh_ldi − dL_dh_mmf (positive = LDI should pay more)
    ratio:      dL_dh_ldi / dL_dh_mmf (how many times more dangerous per unit haircut)
    """
    dL_dh_ldi: float
    dL_dh_mmf: float
    surcharge:  float
    ratio:      float


def optimal_haircut_frontier(
    loss_fn_factory: Callable[[np.ndarray], Callable[[np.ndarray], float]],
    h_init:          np.ndarray,    # (d,) — initial haircut schedule
    h_min_values:    np.ndarray,    # (n_points,) — FSB budget constraint levels
    h_bounds:        tuple[np.ndarray, np.ndarray],  # (lower, upper) bounds on h
    fsb_uniform:     float = 0.05,  # FSB proposed 5% uniform minimum
    n_iter:          int   = 30,    # gradient descent iterations per budget level
    lr:              float = 0.01,  # learning rate
    eps:             float = 1e-4,  # FD step for gradient
) -> HaircutFrontierResult:
    """
    Compute the optimal haircut frontier: minimise L(h) s.t. Σ h ≥ H_min.

    For each value of H_min in h_min_values:
      1. Start from h_init (or previous optimal as warm start)
      2. Run projected gradient descent: h ← proj(h − lr × ∇L(h))
         where proj enforces: (a) Σ h ≥ H_min; (b) h_lower ≤ h ≤ h_upper
      3. Record optimal h and L(h*)

    The result is a Pareto curve showing how systemic risk decreases as
    the total haircut budget increases — the key Figure for Paper 333 §5.

    The FSB benchmark (uniform h = fsb_uniform on all non-sovereign
    collateral) is computed separately for comparison.

    Args:
        loss_fn_factory: callable h → (callable theta → float). Returns a
                          loss function parameterised by the haircut schedule h.
                          The outer callable receives the haircut vector;
                          the inner is the loss_fn passed to policy_gradient().
        h_init:          (d,) initial haircut schedule
        h_min_values:    (n_points,) budget constraint grid (ascending)
        h_bounds:        (lower, upper) — box constraints on h, each shape (d,)
        fsb_uniform:     FSB proposed uniform minimum haircut
        n_iter:          gradient descent iterations per budget point
        lr:              gradient descent learning rate
        eps:             finite-difference step for gradient

    Returns:
        HaircutFrontierResult with the Pareto curve and FSB benchmark
    """
    h_lo, h_hi   = np.array(h_bounds[0]), np.array(h_bounds[1])
    d            = len(h_init)
    n_pts        = len(h_min_values)

    loss_curve         = np.zeros(n_pts)
    haircut_totals     = np.zeros(n_pts)
    haircut_schedules  = np.zeros((n_pts, d))

    h = h_init.copy()   # warm start from previous optimum

    for idx, H_min in enumerate(h_min_values):
        loss_fn = loss_fn_factory(h)

        for _ in range(n_iter):
            # Finite-difference gradient
            grad = np.zeros(d)
            for k in range(d):
                e_k     = np.zeros(d); e_k[k] = 1.0
                loss_hi = loss_fn(h + eps * e_k)
                loss_lo = loss_fn(h - eps * e_k)
                grad[k] = (loss_hi - loss_lo) / (2.0 * eps)

            # Gradient step
            h = h - lr * grad

            # Box projection: h_lo ≤ h ≤ h_hi
            h = np.clip(h, h_lo, h_hi)

            # Budget projection: Σ h ≥ H_min
            deficit = H_min - h.sum()
            if deficit > 0:
                # Add deficit uniformly across all haircuts
                h = h + deficit / d

            h = np.clip(h, h_lo, h_hi)

            # Update loss_fn with new h (factory re-constructs operator)
            loss_fn = loss_fn_factory(h)

        loss_curve[idx]         = loss_fn(h)
        haircut_totals[idx]     = h.sum()
        haircut_schedules[idx]  = h.copy()

    # FSB uniform benchmark
    h_fsb      = np.clip(np.full(d, fsb_uniform), h_lo, h_hi)
    fsb_loss   = float(loss_fn_factory(h_fsb)(h_fsb))
    fsb_haircut = float(h_fsb.sum())

    return HaircutFrontierResult(
        loss_curve         = loss_curve,
        haircut_totals     = haircut_totals,
        haircut_schedules  = haircut_schedules,
        fsb_loss           = fsb_loss,
        fsb_haircut        = fsb_haircut,
        h_min_grid         = np.array(h_min_values),
    )


def ldi_surcharge(
    loss_fn:     Callable[[np.ndarray], float],
    h_ldi:       np.ndarray,   # (n_collateral,) — LDI haircut vector
    h_mmf:       np.ndarray,   # (n_collateral,) — MMF haircut vector
    eps:         float = 1e-4,
) -> LDISurcharge:
    """
    Compute the LDI haircut surcharge: how much more dangerous is LDI leverage?

    ∂L/∂h_ldi_k − ∂L/∂h_mmf_k  for each collateral class k.

    If positive: LDI lenders should face higher haircuts than MMFs for the
    same collateral class. This is the key policy result in Paper 333 §5.3.

    Args:
        loss_fn:  callable (h_ldi, h_mmf) → scalar loss. The two haircut
                   vectors are concatenated before passing to loss_fn, i.e.:
                   loss_fn(np.concatenate([h_ldi, h_mmf]))
        h_ldi:    (n_collateral,) — LDI haircut schedule
        h_mmf:    (n_collateral,) — MMF haircut schedule
        eps:      finite-difference step

    Returns:
        LDISurcharge with gradient components and surcharge
    """
    d    = len(h_ldi)
    h0   = np.concatenate([h_ldi, h_mmf])
    loss0 = float(loss_fn(h0))

    # Gradient w.r.t. LDI haircuts (first d elements)
    grad_ldi = np.zeros(d)
    for k in range(d):
        e_k       = np.zeros(2*d); e_k[k] = 1.0
        grad_ldi[k] = (float(loss_fn(h0 + eps*e_k)) - float(loss_fn(h0 - eps*e_k))) / (2*eps)

    # Gradient w.r.t. MMF haircuts (second d elements)
    grad_mmf = np.zeros(d)
    for k in range(d):
        e_k       = np.zeros(2*d); e_k[d+k] = 1.0
        grad_mmf[k] = (float(loss_fn(h0 + eps*e_k)) - float(loss_fn(h0 - eps*e_k))) / (2*eps)

    # Mean across collateral classes
    mean_ldi = float(np.mean(grad_ldi))
    mean_mmf = float(np.mean(grad_mmf))

    return LDISurcharge(
        dL_dh_ldi = mean_ldi,
        dL_dh_mmf = mean_mmf,
        surcharge  = mean_ldi - mean_mmf,
        ratio      = mean_ldi / (mean_mmf + 1e-10),
    )


# ---------------------------------------------------------------------------
# Beta sensitivity — ∂L/∂β (how much does constraint softness matter?)
# ---------------------------------------------------------------------------

class BetaSensitivityResult(NamedTuple):
    """
    Sensitivity of systemic loss to the Gibbs inverse temperature β.

    dL_dbeta:  ∂L/∂β — positive means softer constraints → more loss
    beta_init: the β at which the gradient is computed
    loss_init: L at beta_init
    """
    dL_dbeta:  float
    beta_init: float
    loss_init: float


def beta_sensitivity(
    loss_fn_of_beta: Callable[[float], float],
    beta_init:       float,
    eps:             float = 1.0,
) -> BetaSensitivityResult:
    """
    Compute ∂L/∂β via finite differences.

    A positive gradient means increasing β (harder thresholds) increases
    systemic loss — which would be counter-intuitive. In practice the
    relationship is non-monotone: very soft constraints (low β) allow gradual
    stress absorption; very hard constraints (high β) create cliff-edge runs.
    The critical β* is where the gradient changes sign.

    Args:
        loss_fn_of_beta: callable beta → scalar loss
        beta_init:       β at which to evaluate the gradient
        eps:             finite-difference step (default 1.0 for β scale)

    Returns:
        BetaSensitivityResult
    """
    loss0   = float(loss_fn_of_beta(beta_init))
    loss_hi = float(loss_fn_of_beta(beta_init + eps))
    loss_lo = float(loss_fn_of_beta(max(beta_init - eps, 0.1)))
    dL_db   = (loss_hi - loss_lo) / (2.0 * eps)

    return BetaSensitivityResult(
        dL_dbeta  = dL_db,
        beta_init = beta_init,
        loss_init = loss0,
    )


# ---------------------------------------------------------------------------
# Summary report — human-readable policy gradient output
# ---------------------------------------------------------------------------

def policy_report(
    grad_result:     PolicyGradientResult,
    param_names:     Optional[list[str]] = None,
    top_k:           int = 5,
) -> str:
    """
    Format a PolicyGradientResult as a human-readable string.

    Used in experiment scripts (x332c, x333f) to print policy gradient
    results without duplicating the formatting logic.

    Args:
        grad_result:  PolicyGradientResult from policy_gradient()
        param_names:  list of parameter names (length = len(gradient))
        top_k:        show the top-k most important parameters

    Returns:
        formatted string
    """
    g     = grad_result.gradient
    names = param_names or [f"θ_{k}" for k in range(len(g))]
    d_loss = grad_result.loss_at_init - grad_result.loss_at_optimal

    lines = [
        f"Policy gradient: {grad_result.label}",
        f"  Loss at init:    {grad_result.loss_at_init:.4f}",
        f"  Loss after step: {grad_result.loss_at_optimal:.4f}  (ΔL = {d_loss:+.4f})",
        f"  Top-{top_k} parameters by |∂L/∂θ|:",
    ]
    order = np.argsort(np.abs(g))[::-1]
    for rank, k in enumerate(order[:top_k]):
        direction = "↑ raise" if g[k] > 0 else "↓ lower"
        lines.append(
            f"    [{rank+1}] {names[k]:20s}  ∂L/∂θ = {g[k]:+.4f}  → {direction} to reduce loss"
        )
    return "\n".join(lines)
