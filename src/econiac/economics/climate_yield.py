"""
Climate Hazard Yield Surface — generalised yield curve for climate mitigation.

Standard finance: a yield curve r(T) maps maturity T to return on investment.
Climate action: the payoff from mitigation depends on BOTH when you invest
and when the benefit is realised — a 2D surface, not a 1D curve.

The Climate Yield Surface Φ(t_invest, t_payoff) gives the damage-avoided per
dollar of mitigation investment made at t_invest, realised at t_payoff:

    Φ(t_invest, t_payoff) = ΔDamage(t_payoff) / Cost(t_invest)

where ΔDamage is the GDP loss avoided relative to BAU, and Cost is the green
investment premium over BAU.

Three key properties distinguish this from a standard yield curve:

1. TRIANGULAR DOMAIN: t_payoff > t_invest always (you can't benefit before investing).
   This makes the surface a triangular region in the (t_invest, t_payoff) plane,
   analogous to a forward-rate surface in fixed income.

2. NONLINEARITY (tipping points): The damage function is convex with exponent γ>1,
   calibrated to the Carbon Tracker / Univ. Exeter (2026) expert elicitation.
   Near the 3°C threshold, each additional avoided GtCO2 is worth much more than
   the previous one — the surface has positive curvature (convexity increases near
   the tipping point), exactly as a callable bond does near its strike.

3. DECLINING EFFECTIVENESS: Early mitigation investments displace more emissions per
   dollar (learning curve, low-hanging fruit). Later investments face higher costs
   and smaller remaining carbon budget. This tilts the surface so early investment
   strictly dominates late investment at any fixed payoff horizon.

The 'point of no return' for a given payoff horizon T is the t_invest at which
Φ(t_invest, T) = 1 — the date beyond which a dollar invested cannot return a
dollar in damage-avoided by T. This is the climate equivalent of the bond
duration limit — the doomsday clock.

The household translation: multiply Φ by (GDP_per_household / GDP_total) to
get the per-household dollar benefit from $1 of mitigation spend per person.

Higher-dimensional structure:
    The full object is a hypersurface in:
        (t_invest, investment_rate, t_payoff, discount_rate, SSP_scenario, γ)
    The 2D surface is the communicable cross-section at fixed scenario and γ.
    The tipping-point isosurface Φ(·,·)=1 is a 1D curve in the 2D domain —
    the 'breakeven frontier', analogous to the yield curve's par line.

References:
    Carbon Tracker / Univ. Exeter (2026) Recalibrating Climate Risk.
    Stern (2006) The Stern Review on the Economics of Climate Change.
    IEA (2023) World Energy Investment.
    Buckley (2026) TIR. doi:10.5281/zenodo.20237288
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class ClimateYieldParameters:
    """
    Parameters for the climate hazard yield surface.

    Damage function calibrated to Carbon Tracker / Univ. Exeter (2026)
    expert elicitation: median damage at 3°C ≈ 10% GDP, γ ≈ 2.5.

    Args:
        damage_at_3C:     GDP damage fraction at 3°C warming        (0.10)
        damage_gamma:     power-law exponent (>1 = nonlinear)        (2.5)
        E_budget_3C:      remaining GtCO2 budget to lock in 3°C      (800.0)
        warming_per_Gt:   °C per GtCO2 cumulative (IPCC AR6 approx)  (0.00375)
        T_baseline_2024:  warming already locked in as of 2024        (1.2)
        BAU_annual_Gt:    BAU annual emissions in 2024                (40.0)
        BAU_decline_rate: annual fractional decline in BAU emissions  (0.005)
        mitigation_eff_2025: Mt CO2 avoided per $bn invested in 2025  (15.0)
        mitigation_eff_decay: annual decline in mitigation effectiveness (0.04)
        transition_cost_frac: green investment premium as % of GDP     (0.02)
        discount_rate:    social discount rate for NPV                (0.03)
        GDP_world_bn:     world GDP in $bn                            (100_000)
        GDP_per_hh:       GDP per household (e.g. Canada)             (55_000)
        hh_per_GDP_bn:    households per $bn of GDP                   (18_000)
    """
    damage_at_3C:        float = 0.10
    damage_gamma:        float = 2.5
    E_budget_3C:         float = 800.0
    warming_per_Gt:      float = 0.00375
    T_baseline_2024:     float = 1.2
    BAU_annual_Gt:       float = 40.0
    BAU_decline_rate:    float = 0.005
    mitigation_eff_2025: float = 15.0
    mitigation_eff_decay:float = 0.04
    transition_cost_frac:float = 0.02
    discount_rate:       float = 0.03
    GDP_world_bn:        float = 100_000.0
    GDP_per_hh:          float = 55_000.0
    hh_per_GDP_bn:       float = 18_000.0


# ---------------------------------------------------------------------------
# Core damage and emissions functions
# ---------------------------------------------------------------------------

def bau_temperature(year: float, p: ClimateYieldParameters) -> float:
    """
    BAU temperature at a given year (°C above pre-industrial).

    Simplified: BAU emissions 40 Gt/yr declining slowly → cumulative emissions
    → temperature via linear climate sensitivity (IPCC AR6 best estimate).

    Capped at 5°C to avoid extrapolating the damage function into implausible
    territory (though some IPCC scenarios do reach 5°C+ by 2100).
    """
    years_since_2024 = max(year - 2024, 0)
    # Cumulative BAU emissions from 2024 to year
    if p.BAU_decline_rate > 0:
        E_cum = p.BAU_annual_Gt * (1 - np.exp(-p.BAU_decline_rate * years_since_2024)) / p.BAU_decline_rate
    else:
        E_cum = p.BAU_annual_Gt * years_since_2024
    dT = E_cum * p.warming_per_Gt
    return float(np.minimum(p.T_baseline_2024 + dT, 5.0))


def nze_temperature(year: float, p: ClimateYieldParameters,
                    nze_halving_years: float = 6.0) -> float:
    """
    NZE-pathway temperature: emissions halve every `nze_halving_years` years.

    IEA NZE 2050: emissions halve roughly every 6 years from 2024,
    reaching near-zero by 2050.
    """
    years_since_2024 = max(year - 2024, 0)
    decline_rate = np.log(2) / nze_halving_years
    if decline_rate > 0:
        E_cum = p.BAU_annual_Gt * (1 - np.exp(-decline_rate * years_since_2024)) / decline_rate
    else:
        E_cum = p.BAU_annual_Gt * years_since_2024
    dT = E_cum * p.warming_per_Gt
    return float(np.minimum(p.T_baseline_2024 + dT, 5.0))


def damage_fraction(T_celsius: float, p: ClimateYieldParameters) -> float:
    """
    GDP damage as a fraction, given temperature T above pre-industrial.

    Power-law calibrated to Carbon Tracker 2026:
        damage(T) = damage_at_3C * (T / 3) ^ gamma

    Properties:
        damage(3°C) = damage_at_3C = 10% GDP (median expert estimate)
        gamma=2.5   → superlinear; damage at 4°C ≈ 22%, at 5°C ≈ 40%
        gamma=1.0   → linear (Stern 2006 approx, now considered too conservative)
        gamma=4.0   → extreme nonlinearity (tail risk estimate)
    """
    T_norm = max(T_celsius, 0.0) / 3.0
    return float(p.damage_at_3C * T_norm ** p.damage_gamma)


def mitigation_effectiveness(t_invest: float, p: ClimateYieldParameters) -> float:
    """
    Mt CO2 avoided per $bn of green investment made at year t_invest.

    Declines over time due to:
    - Learning curve exhaustion (solar/wind already cheap; next-cheapest is harder)
    - Shrinking carbon budget (each remaining GtCO2 requires more expensive abatement)
    - Political economy: easy wins (power sector) taken first; harder sectors later

    Floor at 1.0 Mt/$bn to avoid division-by-zero and represent residual value
    of late investment (e.g. carbon removal at high cost).
    """
    years_since_2025 = max(t_invest - 2025, 0)
    eff = p.mitigation_eff_2025 * np.exp(-p.mitigation_eff_decay * years_since_2025)
    return float(max(eff, 1.0))


# ---------------------------------------------------------------------------
# The yield surface Φ(t_invest, t_payoff)
# ---------------------------------------------------------------------------

def yield_surface_point(
    t_invest: float,
    t_payoff: float,
    p: ClimateYieldParameters,
    investment_duration: float = 25.0,
) -> float:
    """
    Compute one point on the climate yield surface.

    Φ(t_invest, t_payoff) = damage_avoided($bn world GDP) per $bn invested at t_invest,
    measured at t_payoff.

    Interpretation:
        Φ > 1:  positive ROI — the investment returns more than it costs
        Φ = 1:  breakeven (the 'par line' of the climate yield surface)
        Φ < 1:  negative ROI — too late to recover the investment cost by t_payoff
        Φ = 0:  t_payoff ≤ t_invest (benefit cannot precede investment)

    Args:
        t_invest:           year of investment
        t_payoff:           year when damage-avoided benefit is measured
        p:                  ClimateYieldParameters
        investment_duration: years over which the investment avoids emissions (default 25)

    Returns:
        Φ(t_invest, t_payoff) in $bn damage avoided per $bn invested.
    """
    if t_payoff <= t_invest:
        return 0.0

    eff = mitigation_effectiveness(t_invest, p)
    duration = min(investment_duration, t_payoff - t_invest)
    E_avoided_Gt = eff * duration / 1000.0   # convert Mt → Gt

    dT = E_avoided_Gt * p.warming_per_Gt
    T_bau = bau_temperature(t_payoff, p)
    T_mit = max(T_bau - dT, p.T_baseline_2024)

    d_bau = damage_fraction(T_bau, p)
    d_mit = damage_fraction(T_mit, p)

    damage_avoided_frac = d_bau - d_mit
    damage_avoided_bn   = damage_avoided_frac * p.GDP_world_bn

    return float(damage_avoided_bn)   # per $bn invested


def yield_surface(
    t_invest_range: Optional[np.ndarray] = None,
    t_payoff_range: Optional[np.ndarray] = None,
    p: Optional[ClimateYieldParameters] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the full climate yield surface over a grid.

    Returns:
        (T_invest, T_payoff, Phi) where Phi[i, j] = Φ(t_invest[i], t_payoff[j]).
        The lower triangle (t_payoff ≤ t_invest) is set to NaN.

    Usage::

        T_i, T_p, Phi = yield_surface()
        plt.contourf(T_i, T_p, Phi.T)  # note transpose for (x=invest, y=payoff)
    """
    if p is None:
        p = ClimateYieldParameters()
    if t_invest_range is None:
        t_invest_range = np.arange(2024, 2051)
    if t_payoff_range is None:
        t_payoff_range = np.arange(2030, 2105, 2)

    Phi = np.zeros((len(t_invest_range), len(t_payoff_range)))
    for i, ti in enumerate(t_invest_range):
        for j, tp in enumerate(t_payoff_range):
            if tp > ti:
                Phi[i, j] = yield_surface_point(ti, tp, p)
            else:
                Phi[i, j] = np.nan

    T_i, T_p = np.meshgrid(t_invest_range, t_payoff_range, indexing='ij')
    return T_i, T_p, Phi


def breakeven_frontier(
    t_payoff_range: Optional[np.ndarray] = None,
    p: Optional[ClimateYieldParameters] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The breakeven frontier: for each payoff horizon T, the latest t_invest
    at which Φ(t_invest, T) ≥ 1 (positive ROI).

    This is the 'doomsday clock' — the last date to act for the investment
    to pay back by T.  Investing after this date still helps, but the
    discounted return by T is less than the cost.

    Returns (t_payoff_range, t_breakeven) — both shape (n,).
    """
    if p is None:
        p = ClimateYieldParameters()
    if t_payoff_range is None:
        t_payoff_range = np.array([2040, 2050, 2060, 2075, 2100])

    breakevens = []
    for tp in t_payoff_range:
        t_invests = np.arange(2024, tp)
        phis = np.array([yield_surface_point(float(ti), float(tp), p) for ti in t_invests])
        # Find last t_invest where Phi >= 1
        above = np.where(phis >= 1.0)[0]
        if len(above):
            breakevens.append(float(t_invests[above[-1]]))
        else:
            breakevens.append(2024.0)   # already past breakeven at start

    return t_payoff_range, np.array(breakevens)


# ---------------------------------------------------------------------------
# Household dollar translation
# ---------------------------------------------------------------------------

def household_benefit_trajectory(
    p: Optional[ClimateYieldParameters] = None,
    scenario_name: str = "IEA NZE vs BAU",
    nze_halving_years: float = 6.0,
) -> dict:
    """
    Translate climate action into per-household dollar benefits over time.

    Computes the annual and cumulative benefit per household from full
    NZE-pathway mitigation compared to BAU, in today's dollars.

    Returns a dict with arrays:
        years:          2024..2100
        T_bau:          BAU temperature trajectory
        T_nze:          NZE temperature trajectory
        D_bau:          BAU damage fraction
        D_nze:          NZE damage fraction
        annual_benefit: $/household/year (damage avoided vs BAU)
        cumul_benefit:  $/household cumulative
        annual_cost:    $/household/year (transition cost premium)
        cumul_cost:     $/household cumulative
        net_benefit:    cumul_benefit - cumul_cost
        npv_net:        NPV at p.discount_rate
    """
    if p is None:
        p = ClimateYieldParameters()

    years = np.arange(2024, 2101)
    T_bau = np.array([bau_temperature(y, p) for y in years])
    T_nze = np.array([nze_temperature(y, p, nze_halving_years) for y in years])
    D_bau = np.array([damage_fraction(T, p) for T in T_bau])
    D_nze = np.array([damage_fraction(T, p) for T in T_nze])

    # Annual benefit per household: damage_avoided * GDP_per_hh
    annual_benefit = (D_bau - D_nze) * p.GDP_per_hh

    # Annual transition cost: premium paid above BAU investment ($/hh/yr)
    # Peaks 2024–2050 at transition_cost_frac * GDP_per_hh; declines after
    annual_cost = np.where(
        years <= 2050,
        p.transition_cost_frac * p.GDP_per_hh,
        0.005 * p.GDP_per_hh,   # maintenance cost post-transition
    )

    cumul_benefit = np.cumsum(annual_benefit)
    cumul_cost    = np.cumsum(annual_cost)
    net_benefit   = cumul_benefit - cumul_cost

    discount = (1 / (1 + p.discount_rate)) ** (years - 2024)
    npv_net = np.cumsum((annual_benefit - annual_cost) * discount)

    return {
        "years":          years,
        "scenario":       scenario_name,
        "T_bau":          T_bau,
        "T_nze":          T_nze,
        "D_bau":          D_bau,
        "D_nze":          D_nze,
        "annual_benefit": annual_benefit,
        "cumul_benefit":  cumul_benefit,
        "annual_cost":    annual_cost,
        "cumul_cost":     cumul_cost,
        "net_benefit":    net_benefit,
        "npv_net":        npv_net,
    }


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------

# IPCC AR6 scenario damage calibration (median expert estimates)
DAMAGE_SCENARIOS = {
    "Optimistic (γ=1.5, 6% at 3°C)":   ClimateYieldParameters(damage_gamma=1.5, damage_at_3C=0.06),
    "Median (γ=2.5, 10% at 3°C)":       ClimateYieldParameters(damage_gamma=2.5, damage_at_3C=0.10),
    "Pessimistic (γ=3.5, 15% at 3°C)":  ClimateYieldParameters(damage_gamma=3.5, damage_at_3C=0.15),
    "Tail risk (γ=5.0, 25% at 3°C)":    ClimateYieldParameters(damage_gamma=5.0, damage_at_3C=0.25),
}

# Key payoff horizons for doomsday clock
DOOMSDAY_HORIZONS = np.array([2035, 2040, 2050, 2060, 2075, 2100])
