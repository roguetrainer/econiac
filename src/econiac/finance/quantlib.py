"""
QuantLib adapter: wrap QuantLib curves and instruments as Pacioli connections.

QuantLib is the upstream curve engine. econiac wraps its outputs as connections
on the Pacioli manifold, enabling XVA as curvature integrals and
gradient-based calibration through QuantLib outputs.

QuantLib is an optional dependency. This module degrades gracefully when
QuantLib is not installed — all functions raise ImportError with a clear message.

Reference: Buckley (2026) XVA as Curvature. doi:10.5281/zenodo.20257724
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from econiac.finance.curves import YieldCurve
from econiac.finance.credit import HazardRateConnection


def _require_quantlib():
    try:
        import QuantLib as ql
        return ql
    except ImportError:
        raise ImportError(
            "QuantLib is required for this function. "
            "Install it with: pip install QuantLib"
        )


# ---------------------------------------------------------------------------
# YieldCurve ← QuantLib YieldTermStructure
# ---------------------------------------------------------------------------

def ql_curve_to_yield_curve(
    ql_curve,
    maturities_years: list[float],
    day_count=None,
    calendar=None,
) -> YieldCurve:
    """
    Convert a QuantLib YieldTermStructure to an econiac YieldCurve.

    Args:
        ql_curve:         QuantLib YieldTermStructure (e.g. FlatForward,
                          PiecewiseYieldCurve, ZeroCurve, DiscountCurve)
        maturities_years: list of maturities in years to sample
        day_count:        QuantLib DayCounter (default: Actual365Fixed)
        calendar:         QuantLib Calendar (default: NullCalendar)

    Returns:
        YieldCurve with the sampled discount factors.

    Example:
        import QuantLib as ql
        today = ql.Date(1, 1, 2026)
        ql.Settings.instance().evaluationDate = today
        ql_curve = ql.FlatForward(today, 0.05, ql.Actual365Fixed())
        curve = ql_curve_to_yield_curve(ql_curve, [0.5, 1, 2, 5, 10])
    """
    ql = _require_quantlib()

    if day_count is None:
        day_count = ql.Actual365Fixed()
    if calendar is None:
        calendar = ql.NullCalendar()

    ref_date = ql_curve.referenceDate()
    discount_factors = []

    for T in maturities_years:
        # Convert years to QuantLib Date
        maturity_date = ref_date + ql.Period(int(T * 365), ql.Days)
        df = ql_curve.discount(maturity_date)
        discount_factors.append(df)

    return YieldCurve(
        maturities=jnp.array(maturities_years),
        discount_factors=jnp.array(discount_factors),
    )


# ---------------------------------------------------------------------------
# HazardRateConnection ← QuantLib hazard rate / CDS helpers
# ---------------------------------------------------------------------------

def ql_hazard_to_credit_connection(
    ql_hazard_curve,
    maturities_years: list[float],
    name: str = "counterparty",
) -> HazardRateConnection:
    """
    Convert a QuantLib HazardRateCurve (or SurvivalProbabilityCurve) to
    an econiac HazardRateConnection.

    Args:
        ql_hazard_curve:  QuantLib SurvivalProbabilityCurve or HazardRateCurve
        maturities_years: list of maturities in years
        name:             label for the obligor

    Returns:
        HazardRateConnection with forward hazard rates inferred from survival
        probabilities sampled from the QuantLib curve.
    """
    ql = _require_quantlib()

    ref_date = ql_hazard_curve.referenceDate()
    survival_probs = []

    for T in maturities_years:
        maturity_date = ref_date + ql.Period(int(T * 365), ql.Days)
        sp = ql_hazard_curve.survivalProbability(maturity_date)
        survival_probs.append(max(sp, 1e-10))   # enforce positivity

    return HazardRateConnection.from_survival_probabilities(
        maturities=jnp.array(maturities_years),
        survival_probs=jnp.array(survival_probs),
        name=name,
    )


def ql_cds_to_credit_connection(
    cds_spreads_bps: dict[float, float],
    recovery_rate: float = 0.4,
    name: str = "counterparty",
) -> HazardRateConnection:
    """
    Bootstrap a HazardRateConnection from CDS spreads.

    Uses a simple flat-hazard bootstrap: h(T) ≈ s(T) / (1 - R)
    where s(T) is the CDS spread and R is the recovery rate.

    For a more accurate bootstrap using QuantLib's CDS helpers, use
    ql_hazard_to_credit_connection() on a bootstrapped QuantLib curve.

    Args:
        cds_spreads_bps: dict mapping maturity (years) → CDS spread in bps
        recovery_rate:   recovery fraction R ∈ [0,1)
        name:            obligor label

    Returns:
        HazardRateConnection with flat-bootstrapped forward hazard rates.
    """
    lgd = 1.0 - recovery_rate
    mats = sorted(cds_spreads_bps.keys())
    hazards = [cds_spreads_bps[T] * 1e-4 / lgd for T in mats]   # bps → decimal, /LGD

    return HazardRateConnection(
        maturities=jnp.array(mats),
        hazard_rates=jnp.array(hazards),
        name=name,
    )


# ---------------------------------------------------------------------------
# Vanilla without QuantLib: curve construction from market quotes
# ---------------------------------------------------------------------------

def curve_from_swap_rates(
    maturities_years: list[float],
    swap_rates: list[float],
    frequency: int = 2,
) -> YieldCurve:
    """
    Bootstrap a YieldCurve from par swap rates (no QuantLib needed).

    Uses iterative bootstrapping: solve for discount factors such that
    the par swap rate equals the ratio of annuity to par bond.

    Args:
        maturities_years: swap tenors in years [1, 2, 5, 10, ...]
        swap_rates:       par swap rates as decimals [0.04, 0.045, ...]
        frequency:        coupon frequency per year (2 = semi-annual)

    Returns:
        YieldCurve bootstrapped from swap rates.
    """
    dt       = 1.0 / frequency
    dfs      = {}
    all_mats = []

    for T, s in zip(maturities_years, swap_rates):
        # Coupon dates for this swap
        coupon_dates = np.arange(dt, T + 1e-9, dt)
        # Sum of discount factors for intermediate coupon dates (already solved)
        annuity_known = sum(dfs.get(round(t, 6), 0.0) * dt * s
                            for t in coupon_dates[:-1])
        # Solve for the terminal discount factor
        # 1 = s * dt * Σ P(0,tᵢ) + (1 + s*dt) * P(0,T)
        # → P(0,T) = (1 - annuity_known) / (1 + s*dt)
        df_T = (1.0 - annuity_known) / (1.0 + s * dt)
        dfs[round(T, 6)] = df_T
        all_mats.append(T)

    mats_arr = jnp.array(all_mats)
    dfs_arr  = jnp.array([dfs[round(T, 6)] for T in all_mats])
    return YieldCurve(maturities=mats_arr, discount_factors=dfs_arr)
