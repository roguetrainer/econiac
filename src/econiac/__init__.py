"""
econiac — Differentiable Economics on the Pacioli Manifold
==========================================================

Named after MONIAC (1949), Bill Phillips's hydraulic computer.
EconIAC is MONIAC for the 21st century: the economy as a conserved
flow system on a geometric manifold, with automatic differentiation.

Three mathematical foundations
-------------------------------
Gauge theory       The Pacioli manifold (M, ∇): double-entry accounting
                   as a discrete gauge theory; arbitrage as curvature;
                   FX/yield/credit as parallel transport.
                   doi:10.5281/zenodo.20234853  (Paper 291)

Thermodynamics     Gibbs ensemble p(x) ∝ exp(−β·cost): bounded rationality,
                   quantal response equilibrium, differentiable Shapley values.
                   doi:10.5281/zenodo.20234841  (Paper 289)

Sheaf cohomology   H⁰/H¹/H² hierarchy: bilateral / triangular / systemic risk.
                   The 2008 crisis was an H² event.
                   doi:10.5281/zenodo.20642908  (Paper 397)
                   doi:10.5281/zenodo.20642983  (Paper 398 — primer)

Quick start
-----------
    pip install econiac

    from econiac.finance.contagion import (
        FinancialGraph, cohomology_report,   # H⁰/H¹/H² risk report
        gibbs_lift, beta_sweep,              # differentiable thresholds
        run_cascade, policy_gradient,        # cascade simulation + gradients
    )

    graph  = FinancialGraph.from_matrix(A_interbank, total_assets)
    report = cohomology_report(graph, capital_ratios)
    print(report.summary)
    # → "SYSTEMIC: H² event — unhedgeable cascade (H²=0.18, H¹=0.43)"

Overview paper
--------------
    doi:10.5281/zenodo.20679006  (Paper 409 — overview & reading guide)

All papers
----------
    https://zenodo.org/communities/econiac/

Source & docs
-------------
    https://github.com/roguetrainer/econiac
    https://roguetrainer.github.io/econiac/
"""

__version__ = "0.1.0"
__author__ = "Ian R. C. Buckley"

# ── Cohomological risk (H⁰/H¹/H²) ──────────────────────────────────────────
from econiac.finance.contagion.h2 import (
    cohomology_report,
    CohomologyReport,
    h0_section,
    h2_obstruction,
    H2Result,
)
from econiac.finance.contagion.sheaf import (
    FinancialGraph,
    WeightedEdge,
    sheaf_laplacian,
    SheafLaplacianResult,
    h1_obstruction_signal,
    harmonic_decomposition,
    HarmonicDecomposition,
    sheaf_h1_signal,
    SheafTimeSeries,
    SheafPeriod,
    gravity_network,
)

# ── Cascade simulation ───────────────────────────────────────────────────────
from econiac.finance.contagion.operators import (
    SystemState,
    SolvencyState,
    LiquidityState,
    PriceState,
    Operator,
    compose,
    fixed_point,
    FixedPointResult,
    run_cascade,
    CascadeResult,
    CascadeStep,
    pacioli_check,
    PacioliReport,
)

# ── Gibbs / thermodynamic lifting ────────────────────────────────────────────
from econiac.finance.contagion.gibbs import (
    GibbsParams,
    gibbs_lift,
    gibbs_threshold,
    beta_sweep,
    BetaSweepResult,
    smooth_loss,
    combined_loss,
)

# ── Policy gradient ──────────────────────────────────────────────────────────
from econiac.finance.contagion.policy import (
    policy_gradient,
    PolicyGradientResult,
    policy_report,
    optimal_haircut_frontier,
    HaircutFrontierResult,
)

__all__ = [
    # Cohomological risk
    "cohomology_report", "CohomologyReport",
    "h0_section",
    "h1_obstruction_signal",
    "h2_obstruction", "H2Result",
    "FinancialGraph", "WeightedEdge",
    "sheaf_laplacian", "SheafLaplacianResult",
    "harmonic_decomposition", "HarmonicDecomposition",
    "sheaf_h1_signal", "SheafTimeSeries", "SheafPeriod",
    "gravity_network",
    # Cascade
    "SystemState", "SolvencyState", "LiquidityState", "PriceState",
    "Operator", "compose", "fixed_point", "FixedPointResult",
    "run_cascade", "CascadeResult", "CascadeStep",
    "pacioli_check", "PacioliReport",
    # Gibbs
    "GibbsParams", "gibbs_lift", "gibbs_threshold",
    "beta_sweep", "BetaSweepResult",
    "smooth_loss", "combined_loss",
    # Policy
    "policy_gradient", "PolicyGradientResult", "policy_report",
    "optimal_haircut_frontier", "HaircutFrontierResult",
]
