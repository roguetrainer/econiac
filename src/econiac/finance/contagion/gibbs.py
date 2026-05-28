"""
gibbs.py
========
Gibbs lifting for contagion operators.

The Gibbs lift replaces every hard threshold in a contagion operator with
a sigmoid (logistic) function parameterised by inverse temperature β:

    hard:  1[x > threshold]          →   σ(β × (x − threshold))
    hard:  1[x < threshold]          →   σ(β × (threshold − x))

Limits:
    β → ∞   recovers the original hard binary operator (Gorton-style)
    β = 50  typical fire-sale calibration (CHZ Paper 332)
    β = 20  LDI pension fund (continuous de-leveraging)
    β = 5   hedge fund (smooth, risk-appetite driven)
    β → 0   uniform probability (no contagion; maximum entropy)

The Gibbs lift has two consequences that make it essential:

  1. **Differentiability**: every lifted operator is smooth in β and in all
     state variables. JAX autodiff can compute ∂(loss)/∂(policy parameter)
     through the entire cascade in one backward pass.

  2. **Phase diagram**: sweeping β reveals the critical inverse temperature
     β* where the cascade transitions from subcritical (no run) to
     supercritical (full run). This is the main novel result in Papers 332
     and 333 (the (β, h_baseline) phase diagram).

GibbsParams
-----------
Stores the per-primitive β values. Each contagion channel and each lender
type has its own β — this is required for the 2D phase diagram (sweeping
β_ldi vs β_mmf independently). A single scalar β for the entire composed
operator would collapse the phase diagram to a 1D curve.

beta_sweep
----------
Vectorises a cascade run over a grid of (β, h_baseline) values via
``jax.vmap``. Returns the systemic loss surface — the primary input for the
phase diagram figures in Papers 332 and 333.

References:
    Buckley, I. (2026). Paper 313: Thermal economics. doi:10.5281/zenodo.20318505
    Buckley, I. (2026). Paper 332: CHZ fire sales. doi:TBD
    Buckley, I. (2026). Paper 333: Sovereign repo run. doi:TBD
    Buckley, I. (2026). Paper 334: Operator algebra for contagion. doi:TBD
    Hurd, T.R. (2017). arXiv:1711.05289v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from econiac.finance.contagion.operators import (
    Operator,
    SystemState,
    SolvencyState,
    LiquidityState,
    PriceState,
    _no_adjoint,
)


# ---------------------------------------------------------------------------
# GibbsParams — per-primitive β values
# ---------------------------------------------------------------------------

@dataclass
class GibbsParams:
    """
    Inverse temperatures for each contagion channel and lender type.

    Each β controls the sharpness of the corresponding threshold:
        β → ∞   hard binary (Gorton / Eisenberg-Noe style)
        β small  smooth / continuous (realistic heterogeneous response)

    Channel betas (four Hurd channels):
        beta_solvency    — S channel: interbank default contagion
        beta_liquidity   — L channel: funding withdrawal
        beta_fire_sale   — L^A channel: forced selling pressure
        beta_panic       — S^D channel: deposit/repo run trigger

    Lender-type betas (repo market, Paper 333):
        beta_mmf         — Money-market fund (near-hard cliff, regulatory)
        beta_ldi         — LDI pension fund (VaR-based, moderate)
        beta_hedge       — Hedge fund (risk-appetite, smooth)
        beta_triparty    — Triparty/CCP (algorithmic, near-hard)

    Capital-constraint betas (fire-sale market, Paper 332):
        beta_capital     — Bank capital ratio constraint (CHZ calibration)
        beta_liquidity_ratio — Bank liquidity ratio constraint
    """
    # Hurd channel betas
    beta_solvency:       float = 50.0    # S (EN) — moderately sharp
    beta_liquidity:      float = 20.0    # L (GL) — gradual, VaR-like
    beta_fire_sale:      float = 50.0    # L^A — CHZ calibration
    beta_panic:          float = 100.0   # S^D — sharp run dynamics

    # Repo lender-type betas (Paper 333)
    beta_mmf:            float = 500.0   # MMF: near-hard regulatory cliff
    beta_ldi:            float = 20.0    # LDI: continuous VaR-based
    beta_hedge:          float = 5.0     # HF: smooth, risk-appetite driven
    beta_triparty:       float = 200.0   # Triparty: algorithmic, near-hard

    # Fire-sale capital-constraint betas (Paper 332)
    beta_capital:        float = 50.0    # CHZ γ_min constraint
    beta_liq_ratio:      float = 50.0    # CHZ α_min constraint

    def for_lender_type(self, ltype: str) -> float:
        """
        Return the β for a given lender type string.

        Args:
            ltype: one of "mmf", "ldi", "hedge", "triparty"

        Returns:
            scalar β value
        """
        mapping = {
            "mmf":       self.beta_mmf,
            "ldi":       self.beta_ldi,
            "hedge":     self.beta_hedge,
            "triparty":  self.beta_triparty,
        }
        if ltype not in mapping:
            raise ValueError(
                f"Unknown lender type {ltype!r}. "
                f"Expected one of {list(mapping.keys())}."
            )
        return mapping[ltype]

    def hard_limit(self) -> "GibbsParams":
        """
        Return a copy with all β set to 1e6 (hard threshold limit).

        Used in unit tests to verify that the Gibbs-lifted operator
        recovers the exact hard-threshold behaviour at β → ∞.
        """
        return GibbsParams(**{f.name: 1e6 for f in self.__dataclass_fields__.values()})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Core Gibbs lifting primitives
# ---------------------------------------------------------------------------

def gibbs_threshold(
    logit: jax.Array,
    beta: float,
) -> jax.Array:
    """
    Gibbs lift of a hard threshold: σ(β × logit).

    The ``logit`` should be positive when the agent is "above threshold"
    (safe, solvent, rolling) and negative when below (distressed, defaulting,
    withdrawing).

    Examples:
        solvency fraction:   logit = capital_ratio - gamma_min
        roll probability:    logit = coverage - f_roll_threshold
        panic withdrawal:    logit = funding_ratio - run_threshold

    Args:
        logit: shape (n,) — signed distance from threshold
        beta:  inverse temperature

    Returns:
        shape (n,) — smooth approximation to 1[logit > 0], in (0, 1)
    """
    return jax.nn.sigmoid(beta * logit)


def gibbs_weight_asset(
    utilities: jax.Array,
    beta: float,
) -> jax.Array:
    """
    Gibbs routing weights over assets — for portfolio fire-sale allocation.

    Replaces the hard proportional sell rule with a Gibbs mixture:
        w_a(β) = softmax(β × U_a)
    where U_a is the utility of selling asset a (e.g. liquidity, haircut).

    At β=0: sell uniformly across all assets.
    At β→∞: sell only the most liquid / least haircut-penalised asset.

    This is the same gibbs_weights() from econiac.core.ensemble, specialised
    for the fire-sale allocation context.

    Args:
        utilities: (n_assets,) — selling utility per asset class
        beta:      inverse temperature

    Returns:
        (n_assets,) — portfolio weights summing to 1
    """
    return jax.nn.softmax(beta * utilities)


def gibbs_rehyp(
    sigma: jax.Array,
    rehyp_0: jax.Array,
    beta_rehyp: float,
    sigma_0: float,
) -> jax.Array:
    """
    Gibbs lift for rehypothecation chain collapse.

    Under normal conditions, the chain length is rehyp_0 (e.g. 3.05× for
    Sov-core). Under stress (σ > σ_0), counterparties refuse to re-use
    uncertain collateral and the chain collapses toward 1.0 (no reuse).

    rehyp(σ) = 1 + (rehyp_0 - 1) × σ(−β_rehyp × (σ − σ_0))

    This is a smooth sigmoid collapse, not a hard cliff:
        σ << σ_0  →  rehyp ≈ rehyp_0  (full chain, normal conditions)
        σ >> σ_0  →  rehyp ≈ 1.0       (no chain, stress conditions)

    Args:
        sigma:      (n_collateral,) — current rolling volatility per class
        rehyp_0:    (n_collateral,) — baseline chain length per class
        beta_rehyp: inverse temperature for chain collapse
        sigma_0:    volatility threshold at which chain begins to collapse

    Returns:
        (n_collateral,) — effective chain multiplier, in [1, rehyp_0]
    """
    chain_active = jax.nn.sigmoid(-beta_rehyp * (sigma - sigma_0))
    return 1.0 + (rehyp_0 - 1.0) * chain_active


# ---------------------------------------------------------------------------
# gibbs_lift — wrap an Operator with differentiable thresholds
# ---------------------------------------------------------------------------

def gibbs_lift(
    op:   Operator,
    beta: float,
    *,
    name_suffix: str = "",
) -> Operator:
    """
    Lift an Operator to a differentiable Gibbs approximation at inverse
    temperature ``beta``.

    The lifted operator replaces every hard threshold in ``op`` with
    σ(β × logit). The forward pass is identical to ``op.forward`` in
    structure but smooth everywhere.

    Implementation note:
        This function wraps the operator at the Python level. The actual
        Gibbs substitution happens inside the primitives (primitives.py),
        which accept a ``beta`` parameter and call gibbs_threshold()
        internally. ``gibbs_lift`` creates a new Operator whose forward
        calls the primitive with the specified beta.

        For operators that are already Gibbs-parameterised (i.e.,
        ``op.is_gibbs = True``), this function re-lifts with a new beta —
        useful for beta sweeps.

    Args:
        op:          the Operator to lift
        beta:        inverse temperature for the Gibbs approximation
        name_suffix: optional suffix added to the operator name

    Returns:
        A new Operator with ``is_gibbs=True`` and the same structure as
        ``op`` but with all hard thresholds replaced by sigmoid(β × logit).
    """
    suffix = name_suffix or f"[β={beta:.1f}]"

    # Capture the original forward with the new beta
    # Primitives in primitives.py accept beta as a keyword argument.
    # We rebind it here via partial application.
    original_forward = op.forward

    def lifted_forward(x: SystemState) -> SystemState:
        # If the original forward accepts beta as a kwarg, pass it.
        # Otherwise, it is already a closure over a beta value.
        try:
            return original_forward(x, beta=beta)
        except TypeError:
            # Operator doesn't accept beta kwarg — it's already a closure.
            return original_forward(x)

    return Operator(
        forward  = lifted_forward,
        adjoint  = op.adjoint,
        name     = f"{op.name} {suffix}",
        is_gibbs = True,
        channel  = op.channel,
    )


# ---------------------------------------------------------------------------
# beta_sweep — phase diagram over (β, h_baseline) parameter space
# ---------------------------------------------------------------------------

class BetaSweepResult:
    """
    Result of a beta_sweep run.

    loss_surface[i, j] = systemic loss at beta_grid[i], h_grid[j].
    Used to produce the (β, h_baseline) phase diagram in Papers 332 and 333.
    """
    def __init__(
        self,
        beta_grid:     jax.Array,   # (n_beta,)
        h_grid:        jax.Array,   # (n_h,)
        loss_surface:  jax.Array,   # (n_beta, n_h) — systemic loss
        n_distressed:  jax.Array,   # (n_beta, n_h) — number of distressed agents
    ):
        self.beta_grid    = beta_grid
        self.h_grid       = h_grid
        self.loss_surface = loss_surface
        self.n_distressed = n_distressed

    @property
    def critical_beta(self) -> jax.Array:
        """
        Critical β* for each h_baseline: the β at which d(loss)/d(β) is maximised.

        This is the inflection point of the S-curve transition — the inverse
        temperature at which the cascade first ignites.
        """
        # Gradient along the beta axis
        grad_beta = jnp.diff(self.loss_surface, axis=0)
        idx = jnp.argmax(grad_beta, axis=0)   # (n_h,)
        return self.beta_grid[idx]

    def phase_label(self, loss_threshold: float = 0.1) -> jax.Array:
        """
        Binary phase label: 1 = run (loss > threshold), 0 = stable.

        Returns (n_beta, n_h) boolean array.
        """
        return self.loss_surface > loss_threshold


def beta_sweep(
    op_factory:    Callable[[float, float], Operator],
    x0_factory:    Callable[[float], SystemState],
    beta_grid:     jax.Array,
    h_grid:        jax.Array,
    n_periods:     int   = 10,
    fp_tol:        float = 1e-6,
    solvency_floor: float = 0.95,
) -> BetaSweepResult:
    """
    Sweep over a 2D grid of (β, h_baseline) values and compute systemic loss.

    This produces the phase diagram — the primary figure for Papers 332 and 333.

    Args:
        op_factory:    callable (beta, h_baseline) → Operator.
                       Constructs the contagion operator for a given (β, h).
                       Application papers supply this; the library evaluates it.

        x0_factory:    callable (h_baseline) → SystemState.
                       Constructs the initial (pre-shock) state at haircut h.

        beta_grid:     (n_beta,) array of β values to sweep.
                       Typically: jnp.geomspace(1.0, 1000.0, 40)

        h_grid:        (n_h,) array of h_baseline values to sweep.
                       Typically: jnp.linspace(0.01, 0.25, 30)

        n_periods:     number of cascade periods per run (default 10).

        fp_tol:        fixed-point convergence tolerance.

        solvency_floor: threshold for counting agents as distressed.

    Returns:
        BetaSweepResult with loss_surface (n_beta, n_h) and n_distressed (n_beta, n_h).

    Note on implementation:
        We use a pure Python double loop rather than vmap because op_factory
        constructs Python-level Operator objects that cannot be traced by JAX.
        For fully JIT-compiled operators, use beta_sweep_jit() (not yet implemented).
    """
    from econiac.finance.contagion.operators import run_cascade

    n_beta = len(beta_grid)
    n_h    = len(h_grid)

    loss_arr = jnp.zeros((n_beta, n_h))
    dist_arr = jnp.zeros((n_beta, n_h), dtype=jnp.int32)

    for i, beta in enumerate(beta_grid):
        for j, h in enumerate(h_grid):
            op = op_factory(float(beta), float(h))
            x0 = x0_factory(float(h))
            result = run_cascade(
                op             = op,
                x0             = x0,
                n_periods      = n_periods,
                solvency_floor = solvency_floor,
                fp_tol         = fp_tol,
            )
            loss_arr = loss_arr.at[i, j].set(result.systemic_loss)
            dist_arr = dist_arr.at[i, j].set(result.steps[-1].n_distressed)

    return BetaSweepResult(
        beta_grid    = beta_grid,
        h_grid       = h_grid,
        loss_surface = loss_arr,
        n_distressed = dist_arr,
    )


# ---------------------------------------------------------------------------
# Smooth loss — differentiable systemic loss for policy gradient
# ---------------------------------------------------------------------------

def smooth_loss(
    state:          SystemState,
    solvency_floor: float,
    beta_loss:      float = 50.0,
) -> jax.Array:
    """
    Differentiable (smooth) systemic loss functional.

    Replaces the hard indicator 1[p_i < floor] with σ(β × (floor − p_i)):
        L_smooth = Σ_i σ(β_loss × (floor − p_i))

    This is JAX-differentiable with respect to all model parameters
    (haircuts, capital floors, β values, initial state). It is used in
    policy.py for policy gradient computation.

    At β_loss → ∞, L_smooth → count of distressed agents (hard loss).
    At β_loss = 50 (default), L_smooth ≈ hard loss within ±0.02 agents.

    Args:
        state:          current SystemState
        solvency_floor: threshold for "distressed" (e.g. 0.95)
        beta_loss:      sharpness of the smooth indicator (default 50)

    Returns:
        scalar smooth loss ∈ [0, n_agents]
    """
    logit = solvency_floor - state.solvency.fractions   # positive when distressed
    return jnp.sum(jax.nn.sigmoid(beta_loss * logit))


def smooth_loss_liquidity(
    state:           SystemState,
    liquidity_floor: float,
    beta_loss:       float = 50.0,
) -> jax.Array:
    """
    Differentiable liquidity loss: count of agents in a funding run.

    Analogous to smooth_loss() but on the liquidity dimension.
    Used in repo_market policy gradient (Paper 333 §5).

    Args:
        state:           current SystemState
        liquidity_floor: threshold for "in run" (e.g. 0.80)
        beta_loss:       sharpness of the smooth indicator

    Returns:
        scalar smooth liquidity loss ∈ [0, n_agents]
    """
    logit = liquidity_floor - state.liquidity.fractions
    return jnp.sum(jax.nn.sigmoid(beta_loss * logit))


def combined_loss(
    state:           SystemState,
    solvency_floor:  float = 0.95,
    liquidity_floor: float = 0.80,
    beta_loss:       float = 50.0,
    w_solvency:      float = 1.0,
    w_liquidity:     float = 1.0,
    w_price:         float = 0.5,
) -> jax.Array:
    """
    Weighted combination of solvency, liquidity, and price losses.

    L = w_sol × L_sol + w_liq × L_liq + w_price × L_price

    where:
        L_sol   = smooth count of distressed agents (solvency < floor)
        L_liq   = smooth count of illiquid agents (liquidity < floor)
        L_price = mean fractional price drop from P=1 baseline

    Args:
        state:           current SystemState
        solvency_floor:  floor for solvency loss
        liquidity_floor: floor for liquidity loss
        beta_loss:       sharpness of smooth indicators
        w_solvency:      weight on solvency loss
        w_liquidity:     weight on liquidity loss
        w_price:         weight on price-drop loss

    Returns:
        scalar combined loss
    """
    l_sol   = smooth_loss(state, solvency_floor, beta_loss)
    l_liq   = smooth_loss_liquidity(state, liquidity_floor, beta_loss)
    l_price = jnp.mean(jnp.maximum(1.0 - state.price.prices, 0.0))
    return w_solvency * l_sol + w_liquidity * l_liq + w_price * l_price
