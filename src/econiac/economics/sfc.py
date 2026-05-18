"""
Stock-flow consistent (SFC) engine on the Pacioli manifold.

Every Godley-Lavoie SFC model is a Pacioli manifold instance: sectors are nodes,
flows are edges, ∂²=0 is the accounting identity. The gauge group (R_{>0},×) acts
on nominal values; gauge-invariant observables are dimensionless ratios.

Differentiable: calibrate SFC model parameters by gradient descent on
national accounts data. Compute ∂(GDP)/∂(fiscal_multiplier) in one backward pass.

References:
  Buckley (2026) Topology of Conservation, doi:10.5281/zenodo.20234853
  Buckley (2026) Economic Gauge Theory, doi:10.5281/zenodo.20259495
"""

# TODO Phase 3: implement
# - SFCModel(sectors, flows, parameters) — Pacioli manifold instance
# - from_godley_table(table) -> SFCModel
# - step(state, dt) -> next_state  [differentiable via JAX]
# - simulate(model, T, dt) -> trajectory
# - calibrate(model, data, loss_fn) -> fitted_parameters  [gradient descent]
# - social_multiplier(model, beta) -> χ(β)  [phase transition diagnostic]
