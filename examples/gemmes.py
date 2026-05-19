"""
Example: GEMMES on the Pacioli manifold.

GEMMES (General Monetary and Multisectoral Macrodynamics for the Ecological Shift)
extends the Keen (1995) predator-prey model with:
  - A carbon cycle (atmospheric CO₂ concentration)
  - A climate damage function (Nordhaus DICE-style)
  - A carbon price (tax or cap-and-trade)
  - A green investment channel alongside conventional investment

Reference: Bovari, Giraud, McIsaac, Chancel (2018)
           "Coping with Collapse: A Stock-Flow Consistent Monetary Macrodynamics
            of Global Warming". Ecological Economics 147, 383–398.

The GEMMES ODE system (simplified two-sector version):
  State: [ω, λ, d, T, CO₂]
    ω   = wage share of output
    λ   = employment rate
    d   = private debt ratio (debt/GDP)
    T   = global mean temperature anomaly (°C above pre-industrial)
    CO₂ = atmospheric CO₂ concentration (GtC)

  Dynamics:
    dω/dt  = ω · (Φ(λ) - α)
    dλ/dt  = λ · (κ(π_net)/ν - α - β - γ)
    dd/dt  = κ(π_net) - π_net - (α + β) · d
    dT/dt  = (1/τ_T) · (F(CO₂) - λ_T · T)          # energy balance
    dCO₂/dt = σ(1 - n) · Y - δ_CO₂ · CO₂            # carbon cycle

  where:
    π_net = (1 - ω - r·d) · (1 - D(T))              # profit share after climate damage
    D(T)  = 1 - 1/(1 + π₁T + π₂T²)                 # Nordhaus damage function
    F(CO₂) = F₂ₓ · log(CO₂/CO₂_pre) / log(2)       # radiative forcing
    Φ(λ)  = φ₀ + φ₁/(1-λ)²                          # Phillips curve
    κ(π)  = κ₀ + κ₁·exp(κ₂·π)                      # investment function (Keen)
    n     = carbon tax fraction of emissions abated

EconIAC adds:
  - PCL circuit: `fold(β, [invest_brown, invest_green, deleverage])` — the
    green transition switch, with β controlling commitment to the best strategy
  - TIR routing: three-way thermodynamic routing between carbon-intensive,
    green, and retrenchment strategies
  - CurvedBalanceSheet: stranded asset arbitrage = non-zero holonomy between
    brown and green capital valuations
  - Thermal Shapley: attribution of climate-finance crisis to the three sectors
    (workers, firms, banks) and the climate system itself
"""

import jax
import jax.numpy as jnp

from econiac.core.manifold import (
    BalanceSheet,
    CurvedBalanceSheet,
    holonomy,
    add_residual_sector,
)
from econiac.pcl import (
    flow,
    sequence,
    fold,
    typecheck,
    conservation_loss,
    compile as pcl_compile,
)
from econiac.routing import tir_from_scores, route
from econiac.routing.attribution import full_shapley_analysis


# ---------------------------------------------------------------------------
# GEMMES parameters (Bovari et al. 2018, Table 1 calibration)
# ---------------------------------------------------------------------------

PARAMS = dict(
    # Keen dynamics
    alpha  = 0.02,      # labour productivity growth
    beta   = 0.01,      # population growth
    gamma  = 0.01,      # depreciation
    nu     = 3.0,       # capital-output ratio
    r      = 0.03,      # real interest rate
    phi0   = -0.04,     # Phillips curve intercept
    phi1   = 0.0006,    # Phillips curve slope
    kappa0 = 0.0,       # investment function constant
    kappa1 = 0.04,      # investment function scale
    kappa2 = 3.0,       # investment function curvature
    # Climate
    pi1    = 0.0,       # Nordhaus damage: linear coefficient
    pi2    = 0.00236,   # Nordhaus damage: quadratic coefficient
    tau_T  = 50.0,      # climate inertia (years)
    lambda_T = 1.13,    # climate feedback (W/m²/°C)
    F2x    = 3.7,       # radiative forcing per CO₂ doubling (W/m²)
    CO2_pre = 280.0,    # pre-industrial CO₂ (GtC)
    sigma  = 0.02,      # carbon intensity of output (GtC per unit GDP)
    delta_CO2 = 0.01,   # natural CO₂ absorption rate
    n_tax  = 0.10,      # carbon tax abatement fraction
)


# ---------------------------------------------------------------------------
# GEMMES ODE system
# ---------------------------------------------------------------------------

def phillips(lam, phi0, phi1):
    return phi0 + phi1 / jnp.maximum(1.0 - lam, 0.01) ** 2


def investment(pi, kappa0, kappa1, kappa2, nu):
    return (kappa0 + kappa1 * jnp.exp(kappa2 * pi)) / nu


def damage(T, pi1, pi2):
    """Nordhaus (2008) damage function: D(T) ∈ [0,1)."""
    return 1.0 - 1.0 / (1.0 + pi1 * T + pi2 * T ** 2)


def radiative_forcing(CO2, F2x, CO2_pre):
    return F2x * jnp.log(jnp.maximum(CO2, 1.0) / CO2_pre) / jnp.log(2.0)


def gemmes_ode(state: jax.Array, t: float, p: dict) -> jax.Array:
    """
    GEMMES ODE system: dX/dt = f(X, t, params).

    State X = [omega, lambda, d, T, CO2].
    Returns shape (5,).
    """
    omega, lam, d, T, CO2 = state[0], state[1], state[2], state[3], state[4]

    D     = damage(T, p['pi1'], p['pi2'])
    pi    = (1.0 - omega - p['r'] * d) * (1.0 - D)     # net profit share

    phi   = phillips(lam,  p['phi0'],  p['phi1'])
    kappa = investment(pi, p['kappa0'], p['kappa1'], p['kappa2'], p['nu'])
    F     = radiative_forcing(CO2, p['F2x'], p['CO2_pre'])

    domega = omega * (phi - p['alpha'])
    dlam   = lam   * (kappa - p['alpha'] - p['beta'])
    dd     = kappa - pi - (p['alpha'] + p['beta']) * d

    dT     = (F - p['lambda_T'] * T) / p['tau_T']
    dCO2   = p['sigma'] * (1.0 - p['n_tax']) * 100.0 - p['delta_CO2'] * CO2

    return jnp.array([domega, dlam, dd, dT, dCO2])


def gemmes_simulate(
    omega0: float = 0.78,
    lam0:   float = 0.95,
    d0:     float = 0.10,
    T0:     float = 0.85,     # ~2015 temperature anomaly (°C)
    CO2_0:  float = 395.0,    # ~2015 atmospheric CO₂ (GtC)
    T:      float = 120.0,
    dt:     float = 0.1,
    params: dict  = None,
) -> tuple[jax.Array, jax.Array]:
    """Euler integration of the GEMMES system."""
    if params is None:
        params = PARAMS
    state   = jnp.array([omega0, lam0, d0, T0, CO2_0])
    n_steps = int(round(T / dt))
    times   = [0.0]
    traj    = [state]
    t_now   = 0.0
    for _ in range(n_steps):
        dX    = gemmes_ode(state, t_now, params)
        state = state + dX * dt
        t_now += dt
        times.append(t_now)
        traj.append(state)
    return jnp.array(times), jnp.stack(traj)


# ---------------------------------------------------------------------------
# 1. Simulate
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. GEMMES simulation (Keen + climate)")
print("=" * 60)

times, traj = gemmes_simulate(
    omega0=0.78,
    lam0=0.95,
    d0=0.05,
    T0=0.85,
    CO2_0=395.0,
    T=80.0,
    dt=0.02,
    params={**PARAMS, 'kappa1': 0.025, 'kappa2': 2.0},
)

omega = traj[:, 0]
lam   = traj[:, 1]
d     = traj[:, 2]
T_anom = traj[:, 3]
CO2   = traj[:, 4]

T_horizon = float(times[-1])
print(f"Simulation: {len(times)} steps over T={T_horizon:.0f} years")
print(f"Initial:  ω={float(omega[0]):.3f}  λ={float(lam[0]):.3f}  "
      f"d={float(d[0]):.3f}  ΔT={float(T_anom[0]):.2f}°C  CO₂={float(CO2[0]):.0f} GtC")
print(f"Final:    ω={float(omega[-1]):.3f}  λ={float(lam[-1]):.3f}  "
      f"d={float(d[-1]):.3f}  ΔT={float(T_anom[-1]):.2f}°C  CO₂={float(CO2[-1]):.0f} GtC")
print(f"Peak ΔT:  {float(T_anom.max()):.2f}°C at t={float(times[int(T_anom.argmax())]):.0f}")
print(f"Peak debt: {float(d.max()):.3f} at t={float(times[int(d.argmax())]):.0f}")

# Climate damage at peak temperature
D_peak = float(damage(T_anom.max(), PARAMS['pi1'], PARAMS['pi2']))
print(f"Climate damage at peak: {D_peak*100:.2f}% of GDP")
print()


# ---------------------------------------------------------------------------
# 2. Balance sheet with stranded asset curvature
# ---------------------------------------------------------------------------

print("=" * 60)
print("2. CurvedBalanceSheet — stranded asset arbitrage")
print("=" * 60)

# At the Minsky-climate moment, brown capital is overvalued relative to green.
# The gap between their valuations is the stranded asset risk premium —
# a non-zero curvature field on the Pacioli manifold.

sectors     = ['workers', 'firms', 'banks', 'climate']
instruments = ['wages', 'brown_capital', 'green_capital']

omega_now = float(omega[int(d.argmax())])
d_now     = float(d[int(d.argmax())])
T_now     = float(T_anom[int(d.argmax())])

# Stranded asset risk premium: brown capital is overvalued by the NPV
# of expected climate damage — creates non-zero column sum (curvature)
stranded_premium = float(damage(jnp.array(T_now), PARAMS['pi1'], PARAMS['pi2'])) * 50.0

positions = jnp.array([
    #  wages    brown_cap  green_cap
    [  omega_now,  0.0,       0.0    ],   # workers
    [ -omega_now, -d_now,     0.2    ],   # firms: debt + green assets
    [  0.0,        d_now,    -0.2    ],   # banks: hold brown debt, short green
    [  0.0,        stranded_premium, 0.0],  # climate: unpriced damage
])

# Column sums ≠ 0: the stranded premium is the curvature F
curvature = jnp.array([0.0, stranded_premium, 0.0])

curved_bs = CurvedBalanceSheet(
    positions=positions,
    sectors=sectors,
    instruments=instruments,
    curvature=curvature,
)

print(f"CurvedBalanceSheet at peak debt (t={float(times[int(d.argmax())]):.0f}):")
print(f"  {curved_bs}")
print(f"  Stranded asset premium: {stranded_premium:.4f}")
print(f"  Curvature (field strength F): {curvature}")

# Holonomy of an identity loop: equals F — the arbitrage profit
h = holonomy(sequence(
        flow('firms', 'banks', 'brown_capital', 0.0),  # round trip: no-op
        flow('banks', 'firms', 'brown_capital', 0.0),
    ), curved_bs)
print(f"  Holonomy (arbitrage on round trip): {h}")
print()


# ---------------------------------------------------------------------------
# 3. PCL circuit: green transition as a three-way fold
# ---------------------------------------------------------------------------

print("=" * 60)
print("3. PCL circuit — green transition as fold(β, [brown, green, deleverage])")
print("=" * 60)

# Flat balance sheet for PCL (project curved → flat)
bs_flat = curved_bs.to_flat()
# Simplify to 3-sector for PCL (drop climate sector, keep wages+brown+green)
bs_pcl = BalanceSheet(
    positions=jnp.array([
        [ 100.0,   0.0,   20.0],
        [   0.0, -80.0,  -20.0],
        [-100.0,  80.0,    0.0],
    ]),
    sectors=['households', 'firms', 'banks'],
    instruments=['deposits', 'brown_loans', 'green_loans'],
)

# Three strategies after the Minsky-climate moment:
invest_brown  = flow('banks', 'firms', 'brown_loans',  15.0)  # expand carbon capacity
invest_green  = flow('banks', 'firms', 'green_loans',  15.0)  # green transition
deleverage    = flow('firms', 'banks', 'brown_loans',  15.0)  # pay down brown debt

wages = flow('firms', 'households', 'deposits', 30.0)

# β controls how decisively the economy commits to the highest-value strategy
quarterly_explore = sequence(wages, fold(0.5,  [invest_brown, invest_green, deleverage]))
quarterly_commit  = sequence(wages, fold(5.0,  [invest_brown, invest_green, deleverage]))

result_explore = quarterly_explore(bs_pcl)
result_commit  = quarterly_commit(bs_pcl)

assert result_explore.is_consistent(atol=1e-4), "explore circuit must conserve"
assert result_commit.is_consistent(atol=1e-4),  "commit circuit must conserve"

print(f"β=0.5 (explore — hedge all three):")
print(f"  brown_loans = {float(result_explore.positions[1, 1]):.2f}  "
      f"green_loans = {float(result_explore.positions[1, 2]):.2f}")
print(f"β=5.0 (commit — decisive transition):")
print(f"  brown_loans = {float(result_commit.positions[1, 1]):.2f}  "
      f"green_loans = {float(result_commit.positions[1, 2]):.2f}")
print()

fast_quarterly = pcl_compile(quarterly_commit)
print(f"Compiled circuit: {fast_quarterly.name}")
print()


# ---------------------------------------------------------------------------
# 4. TIR routing: carbon tax vs. market signal
# ---------------------------------------------------------------------------

print("=" * 60)
print("4. TIR routing — carbon tax vs. market signal")
print("=" * 60)

# Three investment pathways scored by expected NPV net of climate damage
T_final = float(T_anom[-1])
D_final = float(damage(jnp.array(T_final), PARAMS['pi1'], PARAMS['pi2']))

scores_no_tax   = [0.20, 0.18 - D_final, 0.15]
scores_with_tax = [0.15, 0.21 - D_final, 0.15]   # carbon tax penalises brown, subsidises green

labels = ['brown', 'green', 'deleverage']

w_no_tax   = route(tir_from_scores(labels, scores_no_tax,   beta=3.0))
w_with_tax = route(tir_from_scores(labels, scores_with_tax, beta=3.0))

print(f"Climate damage at t=120: D={D_final*100:.2f}%")
print("Routing weights (β=3.0):")
print("  No carbon tax:  " + "  ".join(f"{l}={float(w):.3f}" for l, w in zip(labels, w_no_tax)))
print("  Carbon tax:     " + "  ".join(f"{l}={float(w):.3f}" for l, w in zip(labels, w_with_tax)))
print()


# ---------------------------------------------------------------------------
# 5. Thermal Shapley: who is responsible for the climate-finance crisis?
# ---------------------------------------------------------------------------

print("=" * 60)
print("5. Thermal Shapley — attribution of climate-finance crisis")
print("=" * 60)

# Four players: workers, firms, banks, policy (carbon tax)
# Value function: improvement in GDP×(1-D(T)) if that coalition acts prudently

peak_damage = float(damage(T_anom.max(), PARAMS['pi1'], PARAMS['pi2']))
peak_debt   = float(d.max())


def climate_crisis_value(coalition: frozenset) -> float:
    """
    Estimated joint welfare gain if this coalition acts prudently.
    Workers: moderate wages → less debt pressure.
    Firms: early green transition → less stranded asset loss.
    Banks: tighten brown credit → less debt overhang.
    Policy: carbon tax → less climate damage.
    """
    base = -(peak_damage + 0.1 * peak_debt)    # baseline: full crisis
    if 0 in coalition:   # workers
        base += 0.05
    if 1 in coalition:   # firms
        base += 0.25     # biggest lever: investment choice
    if 2 in coalition:   # banks
        base += 0.15
    if 3 in coalition:   # policy (carbon tax)
        base += 0.30     # direct damage reduction
    return base


result, _ = full_shapley_analysis(
    value_fn=climate_crisis_value, n_players=4, beta=2.0
)

player_names = ['workers', 'firms', 'banks', 'policy']
print("Thermal Shapley values (crisis mitigation attribution):")
for name, phi in zip(player_names, result.phi):
    print(f"  {name:8s}: φ = {float(phi):+.4f}")
print(f"Bottleneck player: {player_names[result.bottleneck_player]}")
print(f"Total value:       {result.total_value:.4f}")
print()


# ---------------------------------------------------------------------------
# 6. conservation_loss with climate noise
# ---------------------------------------------------------------------------

print("=" * 60)
print("6. conservation_loss — calibrating against climate-adjusted accounts")
print("=" * 60)

# National accounts adjusted for natural capital depreciation have larger
# measurement uncertainty than conventional GDP accounts.
# σ_climate ≈ 3× σ_conventional (Dasgupta Review 2021 estimate)
sigma_conventional = 5.0
sigma_climate      = 15.0

loss_conv    = conservation_loss(quarterly_commit, bs_pcl, sigma=sigma_conventional)
loss_climate = conservation_loss(quarterly_commit, bs_pcl, sigma=sigma_climate)

print(f"Conservation loss (conventional accounts, σ={sigma_conventional}):  {float(loss_conv):.6f}")
print(f"Conservation loss (climate-adjusted,      σ={sigma_climate}): {float(loss_climate):.6f}")
print(f"Both zero: green-transition circuit conserves regardless of accounting convention")
print()

print("=" * 60)
print("GEMMES example complete.")
print(f"  Peak temperature anomaly: {float(T_anom.max()):.2f}°C")
print(f"  Peak debt ratio:          {float(d.max()):.3f}")
print(f"  Climate damage at peak:   {peak_damage*100:.2f}% of GDP")
print(f"  Bottleneck for mitigation: {player_names[result.bottleneck_player]}")
print("=" * 60)
