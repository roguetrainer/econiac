"""
operators.py
============
Core operator algebra for financial contagion models.

The central abstraction is the **Operator** — a typed monotone endomorphism
on SystemState, carrying both a forward pass (contagion) and an adjoint
(the AL-symmetric dual, see Hurd 2017 Proposition 1).

    SystemState = (SolvencyState, LiquidityState, PriceState)

Four primitive operators correspond to Hurd's four contagion channels:

    S  — direct solvency cascade (Eisenberg-Noe heritage)
    L  — direct liquidity cascade (Gai-Kapadia heritage); adjoint of S
    L^A — fire sale / indirect contagion (price-mediated)
    S^D — bank panic / repo run (deposit/funding withdrawal)

These are composed via ``compose()`` and iterated to a greatest fixed point
via ``fixed_point()``. The Gibbs lift (see gibbs.py) makes every hard
threshold differentiable.

AL symmetry (Hurd 2017 Proposition 1)
--------------------------------------
Every solvency operator has a dual liquidity operator: S† = L, L† = S.
This is enforced structurally: primitive Operators are required to carry
an ``adjoint`` callable. Pacioli ∂²=0 is the conserved quantity.

Fixed-point semantics (Knaster-Tarski)
---------------------------------------
Every primitive operator is monotone non-increasing on SystemState
(contagion weakly worsens each component). Starting from the upper bound
(all fractions = 1, prices = initial), descending iteration converges to
the greatest fixed point — the maximum-recovery equilibrium.

    ``fixed_point(op, x0=upper_bound)``   →   greatest fixed point

Design decisions (from CONTAGION_ROADMAP.md):
    1. State type = product SolvencyState × LiquidityState × PriceState
    2. Adjoint required on primitive Operators (AL symmetry structural)
    3. Greatest fixed point via while_loop from upper bound (Knaster-Tarski)
    4. Gibbs lift per-primitive (see gibbs.py)
    5. Application paper interface: provide ApplicationParams; get back
       run_cascade / sheaf_h1 / policy_gradient

References:
    Hurd, T.R. (2017). Bank Panics and Fire Sales, Insolvency and Illiquidity.
        arXiv:1711.05289v1.
    Eisenberg, L. & Noe, T.H. (2001). Systemic Risk in Financial Systems.
        Management Science 47(2), 236-249.
    Gai, P. & Kapadia, S. (2010). Contagion in Financial Networks.
        Proc. R. Soc. A 466, 2401-2423.
    Buckley, I. (2026). Paper 291: Pacioli manifold. doi:10.5281/zenodo.20257596
    Buckley, I. (2026). Paper 313: Thermal economics. doi:10.5281/zenodo.20318505
    Buckley, I. (2026). Paper 332: CHZ fire sales. doi:TBD
    Buckley, I. (2026). Paper 333: Sovereign repo run. doi:TBD
    Buckley, I. (2026). Paper 334: Systemic risk operator algebra. doi:TBD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, NamedTuple, Optional, TypeVar

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# State types — the lattice on which operators act
# ---------------------------------------------------------------------------

class SolvencyState(NamedTuple):
    """
    p_i ∈ [0, 1]^n — fraction of obligations met by each agent.

    p_i = 1  : fully solvent (all interbank obligations paid in full)
    p_i = 0  : complete default (nothing paid)
    p_i ∈ (0,1): partial recovery (Eisenberg-Noe proportional clearing)

    The lattice order is componentwise: p ≤ q iff p_i ≤ q_i for all i.
    Contagion operators are monotone non-increasing on this lattice.
    """
    fractions: jax.Array    # shape (n_agents,), dtype float32, values in [0, 1]


class LiquidityState(NamedTuple):
    """
    p̃_i ∈ [0, 1]^n — fraction of short-term funding obligations met.

    p̃_i = 1  : fully liquid (all repo / deposit obligations rolled)
    p̃_i = 0  : complete run (all funding withdrawn)
    p̃_i ∈ (0,1): partial roll (Gibbs soft withdrawal)

    AL symmetry: LiquidityState is the dual of SolvencyState.
    S† = L and L† = S under the Pacioli inner product.
    """
    fractions: jax.Array    # shape (n_agents,), dtype float32, values in [0, 1]


class PriceState(NamedTuple):
    """
    P_a ∈ R_{>0}^m — prices of m asset classes.

    Lives on the Pacioli manifold (R_{>0}, ×).
    Fire sales are the mechanism by which solvency/liquidity stress
    propagates into price deterioration, which then feeds back into
    solvency via mark-to-market accounting.

    P_a = 1.0  : initial (undistressed) normalised price
    P_a < 1.0  : fire-sale discount
    """
    prices: jax.Array       # shape (n_assets,), dtype float32, values in (0, ∞)


class SystemState(NamedTuple):
    """
    Full system state: product of solvency, liquidity, and price.

    All contagion operators map SystemState → SystemState.
    This is the object on which fixed_point() and compose() operate.

    Lattice order: (p, p̃, P) ≤ (q, q̃, Q)  iff  p ≤ q, p̃ ≤ q̃, P ≤ Q
    componentwise. Contagion is monotone non-increasing on this lattice.
    """
    solvency:   SolvencyState
    liquidity:  LiquidityState
    price:      PriceState

    @classmethod
    def upper_bound(cls, n_agents: int, n_assets: int,
                    initial_prices: Optional[jax.Array] = None) -> "SystemState":
        """
        Upper bound of the lattice: all fractions = 1, prices = initial.

        This is the correct starting point for greatest-fixed-point iteration
        (Knaster-Tarski). Starting here and descending guarantees convergence
        to the maximum-recovery equilibrium.

        Args:
            n_agents:       number of banks / dealers / agents
            n_assets:       number of asset / collateral classes
            initial_prices: (n_assets,) undistressed prices; default = ones
        """
        if initial_prices is None:
            initial_prices = jnp.ones(n_assets)
        return cls(
            solvency  = SolvencyState(fractions=jnp.ones(n_agents)),
            liquidity = LiquidityState(fractions=jnp.ones(n_agents)),
            price     = PriceState(prices=initial_prices),
        )

    def pacioli_norm(self) -> jax.Array:
        """
        Pacioli inner product norm: ||x||² = ||p||² + ||p̃||² + ||P||².

        Used in the fixed-point convergence check and in sheaf.py for the
        H¹ early-warning signal.
        """
        return (
            jnp.sum(self.solvency.fractions ** 2)
            + jnp.sum(self.liquidity.fractions ** 2)
            + jnp.sum(self.price.prices ** 2)
        )

    def distance(self, other: "SystemState") -> jax.Array:
        """
        L∞ distance between two SystemStates.

        Used as the convergence criterion in fixed_point():
            ||x_{n+1} - x_n||_∞ < tol  →  converged.
        """
        dp = jnp.max(jnp.abs(self.solvency.fractions - other.solvency.fractions))
        dq = jnp.max(jnp.abs(self.liquidity.fractions - other.liquidity.fractions))
        dP = jnp.max(jnp.abs(self.price.prices - other.price.prices))
        return jnp.maximum(jnp.maximum(dp, dq), dP)

    def clip(self) -> "SystemState":
        """
        Enforce physical constraints: fractions in [0,1], prices in (0, ∞).

        Called after each fixed-point step to prevent numerical drift.
        """
        return SystemState(
            solvency  = SolvencyState(
                fractions=jnp.clip(self.solvency.fractions, 0.0, 1.0)
            ),
            liquidity = LiquidityState(
                fractions=jnp.clip(self.liquidity.fractions, 0.0, 1.0)
            ),
            price     = PriceState(
                prices=jnp.maximum(self.price.prices, 1e-4)
            ),
        )


# ---------------------------------------------------------------------------
# Operator — the core abstraction
# ---------------------------------------------------------------------------

@dataclass
class Operator:
    """
    A typed monotone endomorphism on SystemState.

    Every contagion channel is an Operator. The library provides four
    primitive operators (S, L, L^A, S^D); application papers construct
    application-specific operators and compose them.

    Fields
    ------
    forward : SystemState → SystemState
        The contagion step. Must be monotone non-increasing on SystemState
        (more distress in → more distress out, componentwise).

    adjoint : SystemState → SystemState
        The AL-symmetric dual. For primitive operators:
            S.adjoint = L  (Hurd Proposition 1)
            L.adjoint = S
        For composed operators, adjoint is the reverse composition:
            compose(f, g).adjoint = compose(g.adjoint, f.adjoint)
        Required for all primitive Operators; may be set to ``_no_adjoint``
        for composed operators where the dual has no economic interpretation.

    name : str
        Human-readable label (e.g. "L^A", "S^D", "Rehyp ∘ L^A ∘ S^D").

    is_gibbs : bool
        True if this operator was produced by gibbs_lift(). Used by the
        test suite to select the appropriate monotonicity check.

    channel : str | None
        Hurd channel label: "S", "L", "LA", "SD", "Rehyp", or None for
        composed / application-specific operators.
    """
    forward:  Callable[[SystemState], SystemState]
    adjoint:  Callable[[SystemState], SystemState]
    name:     str
    is_gibbs: bool = False
    channel:  Optional[str] = None

    def __call__(self, x: SystemState) -> SystemState:
        """Apply the operator forward pass. Allows op(state) syntax."""
        return self.forward(x)

    def __repr__(self) -> str:
        gibbs_tag = " [Gibbs]" if self.is_gibbs else ""
        return f"Operator({self.name!r}{gibbs_tag})"


def _no_adjoint(x: SystemState) -> SystemState:
    """
    Placeholder adjoint for operators where the dual has no clear meaning.

    Raises NotImplementedError if called — makes misuse explicit rather
    than silently returning a wrong result.
    """
    raise NotImplementedError(
        "This operator does not have an implemented adjoint. "
        "Use a primitive operator (S, L, L^A, S^D) for AL-symmetric operations."
    )


# ---------------------------------------------------------------------------
# compose — categorical composition
# ---------------------------------------------------------------------------

def compose(*ops: Operator) -> Operator:
    """
    Compose operators: compose(f, g, h) = f ∘ g ∘ h.

    Applied right-to-left (standard mathematical convention):
        compose(f, g)(x) = f(g(x))

    The adjoint of the composition is the reverse:
        compose(f, g).adjoint = compose(g.adjoint, f.adjoint)

    For the ESL cascade:
        C_ESL = compose(L_A, S_D)      # L^A ∘ S^D
        C_repo = compose(Rehyp, L_A, S_D)  # Rehyp ∘ L^A ∘ S^D

    Args:
        *ops: two or more Operators, applied right-to-left.

    Returns:
        A new Operator whose forward is the composition of all forwards,
        and whose adjoint is the reverse composition of all adjoints.

    Raises:
        ValueError: if fewer than two operators are provided.
    """
    if len(ops) < 2:
        raise ValueError(f"compose() requires at least 2 operators; got {len(ops)}.")

    # Build composed name
    name = " ∘ ".join(op.name for op in ops)

    # Composed forward: right-to-left application
    def composed_forward(x: SystemState) -> SystemState:
        result = x
        for op in reversed(ops):
            result = op.forward(result)
        return result

    # Composed adjoint: reverse order of adjoints
    def composed_adjoint(x: SystemState) -> SystemState:
        result = x
        for op in ops:
            result = op.adjoint(result)
        return result

    return Operator(
        forward  = composed_forward,
        adjoint  = composed_adjoint,
        name     = name,
        is_gibbs = any(op.is_gibbs for op in ops),
        channel  = None,
    )


def delay(op: Operator, n_periods: int = 1) -> Operator:
    """
    Delayed operator: applies op.forward only after n_periods have elapsed.

    Used for the triparty clearing bank channel: intraday credit absorbs
    same-day shocks; the run trigger is next-day rollover.

    The delayed operator stores the previous-period state internally by
    returning the identity transformation for the first n_periods steps.
    Callers must pass ``period`` to the simulation loop; this combinator
    wraps the operator so that the simulation loop does not need to know
    which operators are delayed.

    Args:
        op:       the Operator to delay
        n_periods: number of periods of intraday credit (default 1)

    Returns:
        A new Operator that is the identity for t < n_periods, and
        delegates to op.forward for t ≥ n_periods.

    Note:
        The returned operator carries ``_delay_periods`` and ``_inner``
        as attributes for introspection by the simulation loop.
    """
    def delayed_forward(x: SystemState, period: int = 0) -> SystemState:
        if period < n_periods:
            return x   # identity: intraday credit absorbs the shock
        return op.forward(x)

    delayed = Operator(
        forward  = delayed_forward,
        adjoint  = op.adjoint,
        name     = f"delay({op.name}, {n_periods})",
        is_gibbs = op.is_gibbs,
        channel  = op.channel,
    )
    # Attach metadata for introspection
    delayed._delay_periods = n_periods  # type: ignore[attr-defined]
    delayed._inner = op                 # type: ignore[attr-defined]
    return delayed


# ---------------------------------------------------------------------------
# fixed_point — Knaster-Tarski greatest fixed point
# ---------------------------------------------------------------------------

class FixedPointResult(NamedTuple):
    """Result of a fixed_point() call."""
    state:     SystemState    # greatest fixed point (maximum-recovery equilibrium)
    n_iter:    int            # number of iterations to convergence
    converged: bool           # True if ||x_{n+1} - x_n||_∞ < tol
    residual:  float          # final ||x_{n+1} - x_n||_∞


def fixed_point(
    op:       Operator,
    x0:       SystemState,
    tol:      float = 1e-6,
    max_iter: int   = 200,
) -> FixedPointResult:
    """
    Greatest fixed point of ``op`` via descending iteration from ``x0``.

    Correctness conditions (caller's responsibility):
        1. x0 must be the upper bound of the lattice (use SystemState.upper_bound()).
        2. op.forward must be monotone non-increasing on SystemState.

    Under these conditions, Knaster-Tarski guarantees:
        - The sequence x0 ≥ op(x0) ≥ op²(x0) ≥ ... is monotone decreasing.
        - It converges to the greatest fixed point (maximum-recovery equilibrium).

    This is a pure Python loop (not jax.lax.while_loop) to support arbitrary
    Python-level operator logic (e.g., triparty delay, per-type haircut updates).
    For fully JIT-compiled operators, use fixed_point_jit() below.

    Args:
        op:       the Operator to iterate (must be monotone non-increasing)
        x0:       starting state (MUST be SystemState.upper_bound())
        tol:      convergence threshold on L∞ distance between iterates
        max_iter: maximum number of iterations

    Returns:
        FixedPointResult with the greatest fixed point, iteration count,
        convergence flag, and final residual.
    """
    x = x0
    converged = False
    residual  = float("inf")

    for n in range(max_iter):
        x_new = op.forward(x).clip()
        dist  = float(x.distance(x_new))

        if dist < tol:
            converged = True
            residual  = dist
            x = x_new
            return FixedPointResult(
                state=x, n_iter=n + 1, converged=True, residual=residual
            )

        x        = x_new
        residual = dist

    return FixedPointResult(
        state=x, n_iter=max_iter, converged=False, residual=residual
    )


# ---------------------------------------------------------------------------
# Pacioli consistency — ∂²=0 on the balance sheet complex
# ---------------------------------------------------------------------------

class PacioliReport(NamedTuple):
    """Result of a Pacioli consistency check."""
    consistent:   bool      # True if |assets - liabilities| < atol for all agents
    max_imbalance: float    # max |Σ assets_i - Σ liabilities_i| across agents
    mean_imbalance: float   # mean imbalance


def pacioli_check(
    assets:      jax.Array,   # (n_agents,)  — total assets per agent
    liabilities: jax.Array,   # (n_agents,)  — total liabilities per agent
    atol:        float = 0.5,
) -> PacioliReport:
    """
    Check the Pacioli identity (∂²=0): assets = liabilities + equity.

    In the contagion context this is the stock-flow consistency constraint
    (Hurd Assumption A2). Every bilateral exposure is simultaneously:
        - an asset on the lender's balance sheet
        - a liability on the borrower's balance sheet

    Failure indicates a model calibration error (lenders and borrowers
    are not using consistent notional values).

    Args:
        assets:      total assets per agent (n_agents,)
        liabilities: total liabilities per agent (n_agents,)
        atol:        absolute tolerance (generous because haircut structure
                     creates an approximate rather than exact identity)

    Returns:
        PacioliReport
    """
    imbalance     = jnp.abs(assets - liabilities)
    max_imb       = float(jnp.max(imbalance))
    mean_imb      = float(jnp.mean(imbalance))
    consistent    = max_imb < atol
    return PacioliReport(
        consistent     = consistent,
        max_imbalance  = max_imb,
        mean_imbalance = mean_imb,
    )


def pacioli_check_bilateral(
    exposure_matrix: jax.Array,   # (n_agents, n_agents) — A_ij = exposure of i to j
    atol: float = 1e-6,
) -> PacioliReport:
    """
    Check Pacioli for a bilateral exposure matrix.

    For a clean bilateral matrix: Σ_j A_ij = Σ_j A_ji for all i
    (each agent's total claims equal total obligations, up to equity).
    In practice equity is not zero, so we check that the *net* matrix
    A - A.T has small off-diagonal norm.

    This is the discrete version of ∂²=0 on the directed graph of exposures.

    Args:
        exposure_matrix: (n, n) matrix; A_ij = amount i is owed by j
        atol: tolerance on ||A - A.T||_∞

    Returns:
        PacioliReport (max_imbalance = ||A - A.T||_∞)
    """
    net   = exposure_matrix - exposure_matrix.T
    imb   = jnp.abs(net)
    max_i = float(jnp.max(imb))
    mean_i= float(jnp.mean(imb))
    return PacioliReport(
        consistent     = max_i < atol,
        max_imbalance  = max_i,
        mean_imbalance = mean_i,
    )


# ---------------------------------------------------------------------------
# Cascade simulation — time-series wrapper around fixed_point
# ---------------------------------------------------------------------------

class CascadeStep(NamedTuple):
    """State and diagnostics at one period of the cascade simulation."""
    t:                  int
    state:              SystemState
    n_distressed:       int       # agents with solvency fraction < threshold
    n_illiquid:         int       # agents with liquidity fraction < threshold
    total_funding_gap:  float     # Σ_i (1 - p̃_i) × funding_outstanding_i
    price_drop:         jax.Array # (n_assets,) fractional price fall from t=0
    fp_iters:           int       # fixed-point inner loop iterations
    fp_converged:       bool


class CascadeResult(NamedTuple):
    """Full cascade simulation result across T periods."""
    steps:          list[CascadeStep]
    final_state:    SystemState
    systemic_loss:  float   # scalar loss metric (see cascade_loss in policy.py)
    pacioli:        PacioliReport


def run_cascade(
    op:              Operator,
    x0:              SystemState,
    n_periods:       int,
    solvency_floor:  float = 0.95,   # p_i < floor → agent counted as distressed
    liquidity_floor: float = 0.95,   # p̃_i < floor → agent counted as illiquid
    fp_tol:          float = 1e-6,
    fp_max_iter:     int   = 200,
    assets:          Optional[jax.Array] = None,   # for Pacioli check
    liabilities:     Optional[jax.Array] = None,
) -> CascadeResult:
    """
    Run the cascade for n_periods, applying op at each period.

    Each period:
      1. Apply op to current state (inner fixed-point iteration)
      2. Record diagnostics
      3. Advance to next period

    This is the *outer* time-series loop (one period = one macro shock
    propagation round). The inner loop is fixed_point().

    Args:
        op:              the composed contagion operator
        x0:              initial (pre-shock) system state
        n_periods:       number of simulation periods
        solvency_floor:  threshold below which an agent is "distressed"
        liquidity_floor: threshold below which an agent is "illiquid"
        fp_tol:          tolerance for inner fixed-point loop
        fp_max_iter:     max iterations for inner fixed-point loop
        assets:          (n_agents,) for optional Pacioli check
        liabilities:     (n_agents,) for optional Pacioli check

    Returns:
        CascadeResult with per-period steps and summary statistics
    """
    prices_0 = x0.price.prices
    state    = x0
    steps    = []

    for t in range(n_periods):
        fp = fixed_point(op, state, tol=fp_tol, max_iter=fp_max_iter)
        s  = fp.state

        n_dist = int(jnp.sum(s.solvency.fractions < solvency_floor))
        n_ill  = int(jnp.sum(s.liquidity.fractions < liquidity_floor))
        gap    = float(jnp.sum(1.0 - s.liquidity.fractions))
        drop   = (prices_0 - s.price.prices) / jnp.maximum(prices_0, 1e-8)

        steps.append(CascadeStep(
            t                 = t,
            state             = s,
            n_distressed      = n_dist,
            n_illiquid        = n_ill,
            total_funding_gap = gap,
            price_drop        = drop,
            fp_iters          = fp.n_iter,
            fp_converged      = fp.converged,
        ))
        state = s

    # Pacioli check at final state
    if assets is not None and liabilities is not None:
        pac = pacioli_check(assets, liabilities)
    else:
        pac = PacioliReport(consistent=True, max_imbalance=0.0, mean_imbalance=0.0)

    # Systemic loss: sum of solvency deficits (will be replaced by smooth version in policy.py)
    final_sol = state.solvency.fractions
    loss = float(jnp.sum(jnp.maximum(solvency_floor - final_sol, 0.0)))

    return CascadeResult(
        steps         = steps,
        final_state   = state,
        systemic_loss = loss,
        pacioli       = pac,
    )
