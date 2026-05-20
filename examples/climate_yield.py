"""
Example: Climate Hazard Yield Surface — the investment case for climate action.

Climate mitigation is an investment: pay now (green premium), receive later
(damage avoided).  The payoff depends on BOTH when you invest and when you
collect — a 2D surface, not a 1D yield curve.

This example shows:
  1. Per-household dollar benefits: what full NZE action puts in your pocket
     at 5, 10, 25 years vs. BAU
  2. The yield surface Φ(t_invest, t_payoff): ROI from $1bn of green
     investment made in year t_invest, measured at year t_payoff
  3. The breakeven frontier (doomsday clock): for each payoff horizon,
     the last year to invest and still recover the cost by that date
  4. Nonlinearity: how the convex damage function (γ=2.5) creates
     tipping-point amplification near 3°C

References:
    Carbon Tracker / Univ. Exeter (2026) Recalibrating Climate Risk.
    Buckley (2026) TIR. doi:10.5281/zenodo.20237288
"""

import numpy as np
from econiac.economics.climate_yield import (
    ClimateYieldParameters,
    household_benefit_trajectory,
    yield_surface,
    yield_surface_point,
    breakeven_frontier,
    DAMAGE_SCENARIOS,
    DOOMSDAY_HORIZONS,
)


# ---------------------------------------------------------------------------
# 1. Per-household dollars: what does full climate action put in your pocket?
# ---------------------------------------------------------------------------

print("=" * 70)
print("1. Per-household benefit: full NZE action vs. BAU (median scenario)")
print("=" * 70)

p = ClimateYieldParameters()
traj = household_benefit_trajectory(p)

print(f"\n  Median damage scenario: {p.damage_at_3C*100:.0f}% GDP at 3°C, γ={p.damage_gamma}")
print(f"  GDP per household: ${p.GDP_per_hh:,.0f}/yr")
print(f"  Transition cost: {p.transition_cost_frac*100:.0f}% of GDP/yr until 2050")
print()
print(f"  {'Year':>6}  {'T_BAU':>6}  {'T_NZE':>6}  {'Damage_BAU':>11}  {'Annual_ben':>11}  "
      f"{'Cumul_ben':>10}  {'Cumul_cost':>11}  {'Net_ben':>10}")
print(f"  {'-'*82}")

key_years = [2029, 2034, 2049, 2074, 2099]   # 5, 10, 25, 50, 75 years out
for yr in key_years:
    idx = yr - 2024
    t = traj
    label = f"+{yr-2024}yr"
    print(f"  {yr:>4} ({label:>5})  "
          f"{t['T_bau'][idx]:5.2f}°  "
          f"{t['T_nze'][idx]:5.2f}°  "
          f"{t['D_bau'][idx]*100:9.1f}%  "
          f"${t['annual_benefit'][idx]:>10,.0f}  "
          f"${t['cumul_benefit'][idx]:>9,.0f}  "
          f"${t['cumul_cost'][idx]:>10,.0f}  "
          f"${t['net_benefit'][idx]:>9,.0f}")

# Find payback year (net_benefit > 0)
payback_idx = np.where(traj['net_benefit'] > 0)[0]
if len(payback_idx):
    payback_yr = traj['years'][payback_idx[0]]
    print(f"\n  Investment payback year: {payback_yr} "
          f"({payback_yr - 2024} years from today)")
print()
print(f"  NPV at {p.discount_rate*100:.0f}% discount rate:")
for yr in [2050, 2075, 2100]:
    idx = yr - 2024
    print(f"    {yr}: ${traj['npv_net'][idx]:>8,.0f}/household")
print()


# ---------------------------------------------------------------------------
# 2. Yield surface: Φ(t_invest, t_payoff)
# ---------------------------------------------------------------------------

print("=" * 70)
print("2. Climate yield surface Φ(t_invest, t_payoff)")
print("   $bn damage avoided per $bn invested at t_invest, measured at t_payoff")
print("=" * 70)

t_invest_labels = [2025, 2030, 2035, 2040, 2045, 2050]
t_payoff_labels = [2040, 2050, 2060, 2075, 2100]

print(f"\n  {'invest\\payoff':>14}", end="")
for tp in t_payoff_labels:
    print(f"  {tp:>8}", end="")
print()
print(f"  {'-'*56}")

for ti in t_invest_labels:
    print(f"  {ti:>14}", end="")
    for tp in t_payoff_labels:
        if tp <= ti:
            print(f"  {'—':>8}", end="")
        else:
            phi = yield_surface_point(float(ti), float(tp), p)
            marker = "✓" if phi >= 1.0 else " "
            print(f"  {phi:>7.1f}{marker}", end="")
    print()

print()
print("  ✓ = ROI ≥ 1 (positive return; dollar invested returns more than a dollar)")
print("  Values in $bn damage-avoided per $bn invested")
print()


# ---------------------------------------------------------------------------
# 3. Breakeven frontier (the doomsday clock)
# ---------------------------------------------------------------------------

print("=" * 70)
print("3. The doomsday clock: last year to invest for positive ROI by horizon")
print("=" * 70)

_, breakevens = breakeven_frontier(DOOMSDAY_HORIZONS, p)

print(f"\n  {'Payoff horizon':>16}  {'Last year to act':>17}  {'Window remaining':>17}  {'Φ at deadline':>14}")
print(f"  {'-'*68}")
for tp, tb in zip(DOOMSDAY_HORIZONS, breakevens):
    window = int(tb) - 2024
    phi_at_tb = yield_surface_point(float(tb), float(tp), p)
    urgency = "🟢 >20yr" if window > 20 else ("🟡 10-20yr" if window > 10 else "🔴 <10yr")
    print(f"  {tp:>16}  {int(tb):>17}  {urgency:>17}  {phi_at_tb:>13.2f}x")

print()
print("  Interpretation: to realise a positive ROI by 2050, the last effective")
print("  investment date under median damage assumptions is ~2043.")
print("  Beyond that date, $1 of green investment returns <$1 in avoided damage by 2050")
print("  — but still positive returns by later horizons.")
print()


# ---------------------------------------------------------------------------
# 4. Damage scenario comparison: optimistic, median, pessimistic, tail
# ---------------------------------------------------------------------------

print("=" * 70)
print("4. Scenario comparison: how γ (nonlinearity) changes the picture")
print("=" * 70)

print(f"\n  Net per-household benefit at 2050 and 2075 by damage scenario:")
print(f"  {'Scenario':45s}  {'Net_2050':>10}  {'Net_2075':>10}  {'Payback_yr':>11}")
print(f"  {'-'*78}")

for name, scenario_p in DAMAGE_SCENARIOS.items():
    t2 = household_benefit_trajectory(scenario_p)
    net_2050 = t2['net_benefit'][2050 - 2024]
    net_2075 = t2['net_benefit'][2075 - 2024]
    payback_idx = np.where(t2['net_benefit'] > 0)[0]
    payback = str(t2['years'][payback_idx[0]]) if len(payback_idx) else ">2100"
    print(f"  {name:45s}  ${net_2050:>9,.0f}  ${net_2075:>9,.0f}  {payback:>11}")

print()
print("  Even in the optimistic scenario the cumulative benefit exceeds cost by 2075.")
print("  In the tail-risk scenario (γ=5, consistent with tipping-point interactions)")
print("  the payback is much earlier and the 2075 benefit is transformative.")
print()


# ---------------------------------------------------------------------------
# 5. The geometry: why this is a surface, not a curve
# ---------------------------------------------------------------------------

print("=" * 70)
print("5. Geometric structure of the climate yield surface")
print("=" * 70)

print("""
  Standard bond yield curve:    r(T)           — 1D function of maturity
  Climate yield surface:        Φ(t_i, t_p)    — 2D triangular surface
  Full climate yield object:    Φ(t_i, rate, t_p, γ, scenario, r_discount)
                                               — hypersurface in 6D+

  Key geometric features:

  1. Triangular domain: t_p > t_i always (causality constraint)
     Analogous to: forward rate surface f(T1, T2) in fixed income

  2. Positive curvature near tipping points (γ > 1):
     d²Φ/dE² > 0 — each additional avoided GtCO2 returns more than the last
     Analogous to: positive convexity in long bonds near deflation threshold

  3. Declining ridge: the maximum-ROI investment date moves forward over time
     as the carbon budget shrinks.  The ridge Φ(t_invest*, t_payoff) traces
     a curve across the surface — the climate equivalent of the yield curve.

  4. Breakeven isocurve (Φ=1): the 'par line' of the surface.
     Above the line: positive ROI. Below: negative ROI.
     The doomsday clock is where t_payoff-axis crosses this isocurve.

  5. Non-separability: Φ(t_i, t_p) ≠ f(t_i) × g(t_p)
     You cannot factor the surface into independent investment and payoff terms
     because the damage function is evaluated at (T_BAU - ΔT(t_i, t_p)).
     This means no 'discount factor' decomposition exists — the surface must
     be evaluated as a whole.  Investment and redemption are entangled.
""")

print("=" * 70)
print("Climate yield example complete.")
print("=" * 70)
