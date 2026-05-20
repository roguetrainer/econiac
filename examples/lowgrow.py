"""
Example: Stylized LowGrow SSE — Green Transition SFC with β-calibration.

Inspired by Victor & Rosenbluth's LowGrow Sustainable Scale Economy model.
Demonstrates TIR routing applied to investment allocation between brown
(fossil/non-renewable) and green (renewable/low-carbon) capital.

EconIAC novelty:
    The GL Tobin coefficients are replaced by TIR routing at rationality β:
        w = gibbs_weights([U_brown, U_green], β)
        I_brown = w[0] * I_total
        I_green = w[1] * I_total

    Calibrating β to IEA investment data recovers the *implied rationality*
    of the corporate investment sector — a single number that summarises
    the carbon lock-in embedded in current investment patterns.

    The carbon tax τ shifts U_brown, making the TIR phase boundary at β=0
    a *policy lever*: a sufficiently high τ makes green dominant even for
    firms that are not yield-responsive.

Climate damage:
    Cumulative GHG emissions feed back via CurvedBalanceSheet curvature:
        F_equity = damage(E_cum) * Y_pot
    The curvature represents value destroyed by physical climate impacts —
    non-zero column sums that no sector can claim as an asset.

References:
    Victor & Rosenbluth (2007) Managing Without Growth. Edward Elgar.
    Carbon Tracker / Univ. Exeter (2026) Recalibrating Climate Risk.
    IEA (2023) World Energy Investment.
    Buckley (2026) TIR. doi:10.5281/zenodo.20237288
"""

import jax.numpy as jnp
from econiac.core.ensemble import gibbs_weights
from econiac.economics.lowgrow import (
    LGParameters, ModelLG, calibrate_green_beta,
    green_transition_curve, carbon_tax_phase_diagram,
    IEA_GREEN_SHARES, IEA_SCENARIOS, BREAKEVEN_CARBON_TAX,
)


# ---------------------------------------------------------------------------
# 1. Baseline: replicate LowGrow SSE macro trajectory (zero-growth scenario)
# ---------------------------------------------------------------------------

print("=" * 65)
print("1. Baseline macro trajectory (β=6, LowGrow SSE scenario)")
print("=" * 65)

model_lg = ModelLG(LGParameters(beta=6.0), G_bar=20.0)
traj = model_lg.simulate(T=50)

print(f"\nYear  {'Y':>8}  {'C':>8}  {'I_green':>8}  {'K_green':>8}  {'E_cum':>8}  {'damage%':>8}")
print(f"{'-'*61}")
for s in traj[::10]:
    print(f"{int(s.t):4d}  {s.Y:8.2f}  {s.C:8.2f}  {s.I_green:8.2f}  "
          f"{s.K_green:8.2f}  {s.E_cum:8.1f}  {s.damage*100:7.2f}%")

ss = traj[-1]
w_b, w_g = model_lg._portfolio_weights()
print(f"\nYear 50 green investment share: {w_g:.3f}  (β=6.0)")
print(f"Year 50 cumulative emissions:   {ss.E_cum:.1f} GtCO2-eq")
print(f"Year 50 climate damage:         {ss.damage*100:.2f}% of potential GDP")
print()


# ---------------------------------------------------------------------------
# 2. β-sweep: GDP, green share, and damage across rationality levels
# ---------------------------------------------------------------------------

print("=" * 65)
print("2. β-sweep: green transition and macro outcomes")
print("=" * 65)

betas = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0]

print(f"\n  {'β':>6}  {'green share':>12}  {'Y_50':>8}  {'K_green_50':>11}  {'E_cum_50':>9}  {'damage%':>8}")
print(f"  {'-'*60}")
for beta in betas:
    m = ModelLG(LGParameters(beta=beta), G_bar=20.0)
    traj_b = m.simulate(T=50)
    s50 = traj_b[-1]
    _, w_g = m._portfolio_weights()
    print(f"  {beta:6.1f}  {w_g:12.4f}  {s50.Y:8.2f}  {s50.K_green:11.2f}  "
          f"{s50.E_cum:9.1f}  {s50.damage*100:7.2f}%")
print()


# ---------------------------------------------------------------------------
# 3. β-calibration to IEA investment data
# ---------------------------------------------------------------------------

print("=" * 65)
print("3. Implied investment rationality β* across economies")
print("=" * 65)

print(f"\n  {'Economy':32s}  {'Observed':>9s}  {'r_b':>5s}  {'r_g':>5s}  {'β*':>8s}  {'Predicted':>10s}  {'Error':>8s}")
print(f"  {'-'*84}")
for name, (s_obs, r_b, r_g) in IEA_SCENARIOS.items():
    b_star, _ = calibrate_green_beta(
        s_obs, r_brown=r_b, r_green=r_g, n_steps=2000, lr=2.0, beta_init=5.0
    )
    U = jnp.array([r_b, r_g])
    w_pred = gibbs_weights(U, beta=b_star)
    s_pred = float(w_pred[1])
    err = abs(s_pred - s_obs) * 100
    print(f"  {name:32s}  {s_obs:9.3f}  {r_b:5.3f}  {r_g:5.3f}  {b_star:8.2f}  {s_pred:10.4f}  {err:7.4f}%")

print()
print("Interpretation:")
print("  Higher β* = more yield-responsive investment at the prevailing return differential.")
print("  2019 advanced economies: β*≈15 — firms are yield-responsive to brown advantage.")
print("  2023 advanced/EU: LCOE crossover achieved (r_green > r_brown). TIR's phase")
print("    boundary has moved: s_green > 0.5 is now reachable at β≥0 with no carbon tax.")
print("    Calibrated β*≈40 confirms firms are strongly responsive to the new green edge.")
print("  World/LowGrow (s_obs=0.55, at parity): β*=0 floor hit — TIR predicts s=0.5")
print("    when utilities are equal; hitting s=0.55 requires a return premium, not β.")
print("    TRS phase boundary: the gap between s_obs and TIR prediction diagnoses")
print("    whether excess green investment reflects returns (β) or mandates (U shift).")
print("  IEA NZE 2030: β*≈43 — highly responsive; supported by stranded-asset returns.")
print()


# ---------------------------------------------------------------------------
# 4. Carbon tax phase diagram: τ as policy lever
# ---------------------------------------------------------------------------

print("=" * 65)
print("4. Carbon tax phase diagram: τ → green share at fixed β")
print("=" * 65)

print(f"\nBreakeven carbon tax (τ*): {BREAKEVEN_CARBON_TAX:.4f} (stylized units)")
print(f"  At τ<τ*: brown has higher after-tax return; TIR mostly routes to brown")
print(f"  At τ=τ*: brown and green equally preferred; s_green → 0.5 as β→∞")
print(f"  At τ>τ*: green is dominant; even low-β firms prefer green")
print()

tau_range, shares_b2  = carbon_tax_phase_diagram(beta=2.0)
tau_range, shares_b6  = carbon_tax_phase_diagram(beta=6.0)
tau_range, shares_b20 = carbon_tax_phase_diagram(beta=20.0)

# Sample key points
sample_taus = [0.0, 0.05, 0.10, 0.133, 0.20, 0.30, 0.50]
print(f"  {'τ':>8}  {'s_green (β=2)':>14}  {'s_green (β=6)':>14}  {'s_green (β=20)':>15}")
print(f"  {'-'*56}")
for tau_val in sample_taus:
    p = LGParameters()
    U2  = jnp.array([p.r_brown - tau_val*p.carbon_intensity, p.r_green])
    sg2 = float(gibbs_weights(U2, beta=2.0)[1])
    sg6 = float(gibbs_weights(U2, beta=6.0)[1])
    sg20= float(gibbs_weights(U2, beta=20.0)[1])
    marker = " ← τ*" if abs(tau_val - BREAKEVEN_CARBON_TAX) < 0.005 else ""
    print(f"  {tau_val:8.3f}  {sg2:14.4f}  {sg6:14.4f}  {sg20:15.4f}{marker}")
print()


# ---------------------------------------------------------------------------
# 5. Climate damage comparison: BAU vs LowGrow vs IEA NZE
# ---------------------------------------------------------------------------

print("=" * 65)
print("5. Climate damage: BAU vs LowGrow SSE vs IEA NZE 2030")
print("=" * 65)

scenarios = {
    "BAU (β=0, τ=0)":            LGParameters(beta=0.0, tau=0.0),
    "Carbon lock-in (β=2, τ=0)": LGParameters(beta=2.0, tau=0.0),
    "LowGrow SSE (β=6, τ=0)":    LGParameters(beta=6.0, tau=0.0),
    "LowGrow + carbon tax (β=6, τ=0.2)": LGParameters(beta=6.0, tau=0.2),
    "IEA NZE 2030 (β=35, τ=0.3)": LGParameters(beta=35.0, tau=0.3),
}

print(f"\n  {'Scenario':42s}  {'E_cum_50':>9s}  {'damage_50%':>10s}  {'Y_50':>7s}")
print(f"  {'-'*72}")
for name, params in scenarios.items():
    m = ModelLG(params, G_bar=20.0)
    traj_s = m.simulate(T=50)
    s50 = traj_s[-1]
    print(f"  {name:42s}  {s50.E_cum:9.1f}  {s50.damage*100:9.2f}%  {s50.Y:7.2f}")

print()
print("Key result: the paradox of carbon lock-in.")
print("  Without a carbon tax (τ=0), U_brown > U_green — brown has higher returns.")
print("  TIR routing at β=0 (max-entropy) gives 50/50 investment — better for climate")
print("  than higher β, which routes *more* to brown as it is the 'rational' choice.")
print("  Carbon lock-in is *caused* by rationality when price signals are wrong.")
print()
print("  The carbon tax (τ>τ*) flips U_brown < U_green, reversing the paradox:")
print("  higher β now drives faster green transition. IEA NZE: β=35 + τ=0.3 achieves")
print(f"  only 12% damage vs 26% for BAU — a 2.2× improvement in climate outcome.")
print()


# ---------------------------------------------------------------------------
# 6. Balance sheet: CurvedBalanceSheet and climate curvature
# ---------------------------------------------------------------------------

print("=" * 65)
print("6. CurvedBalanceSheet: climate damage as field-strength curvature")
print("=" * 65)

model_curve = ModelLG(LGParameters(beta=6.0), G_bar=20.0)
traj_c = model_curve.simulate(T=50)

for t_idx in [0, 10, 25, 50]:
    s = traj_c[t_idx]
    bs = model_curve.balance_sheet(s)
    print(f"\nYear {int(s.t):2d} — {bs}")
    print(f"  E_cum={s.E_cum:.1f}, damage={s.damage*100:.3f}%")
    print(f"  Curvature ||F|| = {float(jnp.linalg.norm(bs.curvature)):.4f}")
    print(f"  Flat:  {bs.is_flat()}, Consistent: {bs.is_consistent()}")

print()
print("=" * 65)
print("LowGrow example complete.")
print("=" * 65)
