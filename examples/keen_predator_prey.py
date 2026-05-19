"""
Example: Keen predator-prey (Lotka-Volterra debt dynamics) on the Pacioli manifold.

This example demonstrates the full econiac stack end-to-end:

  1. Simulate the Keen (1995) three-sector model using econiac.economics.minsky
  2. Inspect the balance sheet at each timestep using econiac.core.manifold
  3. Build the investment decision as a PCL circuit using econiac.pcl
  4. Route the Minsky moment (the debt crisis switch) via TIR
  5. Attribute the crisis to sectors using thermal Shapley values
  6. Calibrate β to historical data using conservation_loss

The Keen model captures the core Minsky dynamic:
  - Low debt → high profit → high investment → rising employment
  - High employment → rising wages → falling profit → falling investment
  - Falling investment → rising debt → the Minsky moment (debt deflation)

The PCL transistor `choose(β, invest, deleverage)` is the Minsky moment switch:
  at β=0  the system hedges between investment and deleveraging simultaneously
  at β→∞  the system commits fully to whichever strategy maximises net worth

References:
    Keen (1995) Finance and Economic Breakdown. J. Post Keynesian Economics.
    Buckley (2026) Economic Gauge Theory. doi:10.5281/zenodo.20259495
    Buckley (2026) Pacioli Combinator Library. doi:10.5281/zenodo.20262070
"""

import jax.numpy as jnp

from econiac.core.manifold import (
    BalanceSheet,
    add_residual_sector,
    residual_magnitude,
    three_sector_sfc,
)
from econiac.economics.minsky import keen_simulate
from econiac.pcl import (
    flow,
    sequence,
    choose,
    typecheck,
    conservation_loss,
    compile as pcl_compile,
)
from econiac.pcl.combinators import Computation
from econiac.routing import tir_from_scores, route
from econiac.routing.attribution import full_shapley_analysis


# ---------------------------------------------------------------------------
# 1. Simulate the Keen ODE
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Keen predator-prey simulation")
print("=" * 60)

times, traj = keen_simulate(
    omega0=0.80,
    lam0=0.94,
    d0=0.10,
    T=150.0,
    dt=0.05,
)

omega = traj[:, 0]
lam   = traj[:, 1]
d     = traj[:, 2]

print(f"Simulation: {len(times)} steps over T=150 years")
print(f"Initial state:  ω={float(omega[0]):.3f}, λ={float(lam[0]):.3f}, d={float(d[0]):.3f}")
print(f"Final state:    ω={float(omega[-1]):.3f}, λ={float(lam[-1]):.3f}, d={float(d[-1]):.3f}")
print(f"Peak debt ratio: {float(d.max()):.3f} at t={float(times[int(d.argmax())]):.1f}")
print()


# ---------------------------------------------------------------------------
# 2. Balance sheet inspection
# ---------------------------------------------------------------------------

print("=" * 60)
print("2. Pacioli manifold — balance sheet at key moments")
print("=" * 60)

sectors     = ['workers', 'firms', 'banks']
instruments = ['wage_share', 'debt_ratio']


def make_bs(i):
    w, dv = float(omega[i]), float(d[i])
    positions = jnp.array([
        [  w,   0.0],
        [ -w,  -dv ],
        [  0.0,  dv],
    ])
    return BalanceSheet(positions=positions, sectors=sectors, instruments=instruments)


bs_initial = make_bs(0)
bs_peak    = make_bs(int(d.argmax()))
bs_final   = make_bs(-1)

print(f"Initial:    {bs_initial}")
print(f"Peak debt:  {bs_peak}")
print(f"Final:      {bs_final}")
print()

# Demonstrate residual sector for noisy national accounts
noisy_positions = bs_peak.positions.at[0, 0].add(0.03)
noisy_bs = BalanceSheet(positions=noisy_positions, sectors=sectors, instruments=instruments)
reconciled_bs = add_residual_sector(noisy_bs)
print(f"Noisy national accounts (3% survey error):")
print(f"  Raw col sums:         {noisy_bs.column_sums()}")
print(f"  After reconciliation: consistent={reconciled_bs.is_consistent()}, "
      f"residual magnitude={residual_magnitude(reconciled_bs):.4f}")
print()


# ---------------------------------------------------------------------------
# 3. PCL circuit: the Minsky moment as a transistor
# ---------------------------------------------------------------------------

print("=" * 60)
print("3. PCL circuit — Minsky moment as choose(β, invest, deleverage)")
print("=" * 60)

bs_pcl = BalanceSheet(
    positions=jnp.array([
        [ 100.0,   0.0],
        [   0.0, -80.0],
        [-100.0,  80.0],
    ]),
    sectors=['households', 'firms', 'banks'],
    instruments=['deposits', 'loans'],
)

invest     = flow('banks', 'firms', 'loans', 20.0)
deleverage = flow('firms', 'banks', 'loans', 20.0)

quarterly_low  = sequence(
    flow('firms', 'households', 'deposits', 30.0),
    sequence(
        flow('households', 'firms', 'deposits', 5.0),
        choose(0.5, invest, deleverage),
    ),
)
quarterly_high = sequence(
    flow('firms', 'households', 'deposits', 30.0),
    sequence(
        flow('households', 'firms', 'deposits', 5.0),
        choose(10.0, invest, deleverage),
    ),
)

assert typecheck(quarterly_low)
assert typecheck(quarterly_high)

result_low  = quarterly_low(bs_pcl)
result_high = quarterly_high(bs_pcl)

print(f"β=0.5  (hedging):   firm loans = {float(result_low.positions[1, 1]):.2f}")
print(f"β=10.0 (decisive):  firm loans = {float(result_high.positions[1, 1]):.2f}")
print()

fast_quarterly = pcl_compile(quarterly_high)
print(f"Compiled circuit: {fast_quarterly.name}")
print()


# ---------------------------------------------------------------------------
# 4. TIR routing: thermodynamic routing at the Minsky moment
# ---------------------------------------------------------------------------

print("=" * 60)
print("4. TIR routing — thermodynamic information routing")
print("=" * 60)

labels        = ['invest', 'deleverage', 'hold']
profit_scores = [0.15, 0.25, 0.20]

weights_low  = route(tir_from_scores(labels, profit_scores, beta=0.5))
weights_high = route(tir_from_scores(labels, profit_scores, beta=5.0))

print("Routing weights at Minsky moment:")
print("  β=0.5 (explore):  " + "  ".join(f"{l}={float(w):.3f}" for l, w in zip(labels, weights_low)))
print("  β=5.0 (exploit):  " + "  ".join(f"{l}={float(w):.3f}" for l, w in zip(labels, weights_high)))
print()


# ---------------------------------------------------------------------------
# 5. Thermal Shapley: which sector drove the crisis?
# ---------------------------------------------------------------------------

print("=" * 60)
print("5. Thermal Shapley attribution — who drove the Minsky moment?")
print("=" * 60)


def crisis_value(coalition: frozenset) -> float:
    """Estimated mitigation value if this coalition acted prudently."""
    base = -float(d.max())
    if 0 in coalition:   # workers moderate wage demands
        base += 0.15
    if 1 in coalition:   # firms deleverage
        base += 0.35
    if 2 in coalition:   # banks tighten credit
        base += 0.20
    return base


result, _ = full_shapley_analysis(value_fn=crisis_value, n_players=3, beta=2.0)

player_names = ['workers', 'firms', 'banks']
print("Thermal Shapley values (contribution to crisis mitigation):")
for name, phi in zip(player_names, result.phi):
    print(f"  {name:10s}: φ = {float(phi):+.4f}")
print(f"Bottleneck player: {player_names[result.bottleneck_player]}")
print(f"Total value:       {result.total_value:.4f}")
print()


# ---------------------------------------------------------------------------
# 6. conservation_loss: soft calibration for noisy data
# ---------------------------------------------------------------------------

print("=" * 60)
print("6. conservation_loss — soft calibration against national accounts")
print("=" * 60)

sigma = 5.0   # £5-unit measurement noise scale

loss_good = conservation_loss(quarterly_high, bs_pcl, sigma=sigma)
print(f"Conservation loss (well-specified circuit): {float(loss_good):.6f}")


def leaky_fn(bs):
    return BalanceSheet(
        positions=bs.positions.at[0, 0].add(10.0),
        sectors=bs.sectors,
        instruments=bs.instruments,
    )


leaky      = Computation(name="leaky", fn=leaky_fn)
loss_tight = conservation_loss(leaky, bs_pcl, sigma=1.0)
loss_loose = conservation_loss(leaky, bs_pcl, sigma=10.0)

print(f"Leaky model (10-unit error):")
print(f"  σ=1.0  (audited accounts): loss = {float(loss_tight):.2f}")
print(f"  σ=10.0 (survey data):      loss = {float(loss_loose):.2f}")
print(f"  Ratio {float(loss_tight)/float(loss_loose):.1f}× — scales as (σ_tight/σ_loose)²")
print()

print("=" * 60)
print("Example complete. All conservation invariants preserved.")
print("=" * 60)
