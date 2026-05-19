"""
Maslov-Gibbs Einsum (MGE): partition function, routing weights, β-schedule.

Z(β) = Σ exp(β·U_i)
w_i(β) = exp(β·U_i) / Z(β)   [Gibbs routing weights]
F(β) = -β⁻¹ ln Z(β)          [TIR free energy]

As β→0: uniform (maximum entropy, pure exploration)
As β→∞: argmax (tropical limit, pure exploitation)

References:
  Buckley (2026) MGE: doi:10.5281/zenodo.17981393
  Buckley (2026) TIR: doi:10.5281/zenodo.20237288
"""

# TODO Phase 1: implement
# - partition_function(utilities, beta) -> Z
# - gibbs_weights(utilities, beta) -> w array
# - free_energy(utilities, beta) -> F
# - choose(beta, *candidates) -> MGE-weighted combination
# - beta_schedule(beta_0, beta_final, n_steps, schedule='linear'|'geometric'|'adiabatic')
