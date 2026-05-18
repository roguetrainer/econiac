"""
Example: Keen predator-prey (Lotka-Volterra debt dynamics) on the Pacioli manifold.

Steve Keen's Minsky model: employment rate λ and debt ratio d as coupled ODEs.
dλ/dt = λ(α - β·d)
dd/dt = d(r - α) + (1-s)·λ

This is a flow on the Pacioli manifold: conservation of sectoral balances
enforces that household income + firm profit + bank interest = GDP.

Planned: requires econiac.economics.minsky (Phase 3)
"""

import numpy as np
from scipy.integrate import solve_ivp

# Keen (1995) parameters
alpha = 0.025   # productivity growth
beta  = 0.01    # debt sensitivity
r     = 0.05    # interest rate
s     = 0.8     # savings rate

def keen_odes(t, y):
    lam, d = y
    dlam = lam * (alpha - beta * d)
    dd   = d * (r - alpha) + (1 - s) * lam
    return [dlam, dd]

sol = solve_ivp(keen_odes, [0, 100], [0.6, 0.5], dense_output=True)
t = np.linspace(0, 100, 500)
lam, d = sol.sol(t)

print(f"Final employment rate: {lam[-1]:.4f}")
print(f"Final debt ratio:      {d[-1]:.4f}")

# TODO: replace with econiac.economics.minsky.keen_predator_prey() (Phase 3)
# TODO: add gradient-based calibration to national accounts data
