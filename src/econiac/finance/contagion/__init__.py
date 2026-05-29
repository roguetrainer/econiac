"""
econiac.finance.contagion
=========================
Operator algebra for financial contagion and systemic risk models.

This subpackage provides:

  Operator algebra (operators.py):
    SystemState, SolvencyState, LiquidityState, PriceState
    Operator, compose(), delay(), fixed_point()
    pacioli_check(), pacioli_check_bilateral()
    run_cascade(), CascadeResult, CascadeStep

  Gibbs lifting (gibbs.py):
    GibbsParams
    gibbs_lift(), gibbs_threshold(), gibbs_rehyp()
    beta_sweep(), BetaSweepResult
    smooth_loss(), smooth_loss_liquidity(), combined_loss()

  Primitive operators (primitives.py) — Step 3 in build order:
    S_direct (EN solvency), L_direct (GL liquidity), wire_al_symmetry()
    L_A (fire sale), S_D (bank panic / repo run), Rehyp (rehypothecation)
    esl_operator(), repo_esl_operator()
    SolvencyParams, LiquidityParams, FireSaleParams, PanicParams, RehypParams

  Sheaf early-warning (sheaf.py) — Step 4 in build order:
    FinancialGraph, WeightedEdge
    sheaf_laplacian(), SheafLaplacianResult
    h1_signal(), h1_signal_normalised()
    harmonic_decomposition(), HarmonicDecomposition
    sheaf_h1_signal(), SheafTimeSeries, SheafPeriod
    compare_h1_series(), IsomorphismResult
    gravity_network()

  Policy gradient (policy.py) — Step 5 in build order:
    cascade_loss(), endpoint_loss()
    policy_gradient(), PolicyGradientResult, policy_report()
    optimal_haircut_frontier(), HaircutFrontierResult
    ldi_surcharge(), LDISurcharge
    beta_sensitivity(), BetaSensitivityResult

Application papers import from here:

    from econiac.finance.contagion import (
        SystemState, Operator, compose, fixed_point,
        GibbsParams, gibbs_lift, beta_sweep,
        smooth_loss, combined_loss, run_cascade,
    )

Build order (from CONTAGION_ROADMAP.md):
    Step 1: operators.py  ✅
    Step 2: gibbs.py      ✅
    Step 3: primitives.py (pending)
    Step 4: sheaf.py      (pending)
    Step 5: policy.py     (pending)
    Step 6: refactor fire_sales.py   (pending)
    Step 7: refactor repo_market.py  (pending)

References:
    Buckley (2026) Paper 334: Systemic Risk Operator Algebra. doi:TBD
    CONTAGION_ROADMAP.md (papers/334_contagion_framework/)
"""

# ── Step 1: operator algebra ────────────────────────────────────────────────
from econiac.finance.contagion.operators import (
    # State types
    SolvencyState,
    LiquidityState,
    PriceState,
    SystemState,
    # Operator type and core combinators
    Operator,
    compose,
    delay,
    # Fixed-point iteration
    fixed_point,
    FixedPointResult,
    # Pacioli consistency
    pacioli_check,
    pacioli_check_bilateral,
    PacioliReport,
    # Cascade simulation
    run_cascade,
    CascadeResult,
    CascadeStep,
)

# ── Step 5: policy gradient ─────────────────────────────────────────────────
from econiac.finance.contagion.policy import (
    # Loss functionals
    cascade_loss,
    endpoint_loss,
    # Policy gradient
    policy_gradient,
    PolicyGradientResult,
    policy_report,
    # Optimal haircut frontier
    optimal_haircut_frontier,
    HaircutFrontierResult,
    # LDI surcharge
    ldi_surcharge,
    LDISurcharge,
    # Beta sensitivity
    beta_sensitivity,
    BetaSensitivityResult,
)

# ── Step 4: sheaf early-warning ─────────────────────────────────────────────
from econiac.finance.contagion.sheaf import (
    # Graph types
    WeightedEdge,
    FinancialGraph,
    # Laplacian
    sheaf_laplacian,
    SheafLaplacianResult,
    # Scalar signals
    h1_signal,
    h1_signal_normalised,
    h1_obstruction_signal,
    # Harmonic decomposition
    harmonic_decomposition,
    HarmonicDecomposition,
    # Time-series tracking
    sheaf_h1_signal,
    SheafTimeSeries,
    SheafPeriod,
    # Cross-model comparison
    compare_h1_series,
    IsomorphismResult,
    # Convenience
    gravity_network,
)

# ── Step 3: primitive operators ─────────────────────────────────────────────
from econiac.finance.contagion.primitives import (
    # Parameter types
    SolvencyParams,
    LiquidityParams,
    FireSaleParams,
    PanicParams,
    RehypParams,
    # Primitive operators
    S_direct,
    L_direct,
    wire_al_symmetry,
    L_A,
    S_D,
    Rehyp,
    # Pre-built ESL compositions
    esl_operator,
    repo_esl_operator,
    # Shared tâtonnement primitive
    _tatonnement_price_step,
)

# ── Step 2: Gibbs lifting ────────────────────────────────────────────────────
from econiac.finance.contagion.gibbs import (
    # Parameter type
    GibbsParams,
    # Core Gibbs primitives
    gibbs_lift,
    gibbs_threshold,
    gibbs_rehyp,
    gibbs_weight_asset,
    # Phase diagram
    beta_sweep,
    BetaSweepResult,
    # Smooth loss functionals
    smooth_loss,
    smooth_loss_liquidity,
    combined_loss,
)

__all__ = [
    # State types
    "SolvencyState", "LiquidityState", "PriceState", "SystemState",
    # Operator algebra
    "Operator", "compose", "delay",
    "fixed_point", "FixedPointResult",
    # Pacioli
    "pacioli_check", "pacioli_check_bilateral", "PacioliReport",
    # Cascade
    "run_cascade", "CascadeResult", "CascadeStep",
    # Gibbs
    "GibbsParams",
    "gibbs_lift", "gibbs_threshold", "gibbs_rehyp", "gibbs_weight_asset",
    "beta_sweep", "BetaSweepResult",
    "smooth_loss", "smooth_loss_liquidity", "combined_loss",
    # Primitives — parameter types
    "SolvencyParams", "LiquidityParams", "FireSaleParams", "PanicParams", "RehypParams",
    # Primitives — operators
    "S_direct", "L_direct", "wire_al_symmetry",
    "L_A", "S_D", "Rehyp",
    # Pre-built compositions
    "esl_operator", "repo_esl_operator",
    # Shared utility
    "_tatonnement_price_step",
    # Policy
    "cascade_loss", "endpoint_loss",
    "policy_gradient", "PolicyGradientResult", "policy_report",
    "optimal_haircut_frontier", "HaircutFrontierResult",
    "ldi_surcharge", "LDISurcharge",
    "beta_sensitivity", "BetaSensitivityResult",
    # Sheaf
    "WeightedEdge", "FinancialGraph",
    "sheaf_laplacian", "SheafLaplacianResult",
    "h1_signal", "h1_signal_normalised", "h1_obstruction_signal",
    "harmonic_decomposition", "HarmonicDecomposition",
    "sheaf_h1_signal", "SheafTimeSeries", "SheafPeriod",
    "compare_h1_series", "IsomorphismResult",
    "gravity_network",
]
