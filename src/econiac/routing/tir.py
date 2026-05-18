"""
Thermodynamic Information Routing (TIR): universal Gibbs routing primitive.

Four axioms: candidates, admissibility geometry, β, Gibbs output.
The Gibbs weights are the unique routing primitive preserving:
  (i)  conformal invariance — scale-free in utility units
  (ii) symplectic structure — Hamiltonian flow, no information dissipation
  (iii) adiabatic invariance — β-schedule tracks free energy minimum

Eight independent rediscoveries: McFadden, Sims, McKelvey-Palfrey, Jaynes,
Gibbs, Maslov, Goel, Friston — all derived the same theorem.

Reference: Buckley (2026) TIR, doi:10.5281/zenodo.20237288
"""

# TODO Phase 4: implement
# - TIRInstance(candidates, utilities, geometry, beta)
# - route(tir_instance) -> Gibbs weights
# - escape_arrow(tir_instance) -> bool  [does this instance escape Arrow's theorem?]
# - beta_schedule(beta_0, beta_final, n_steps, spectral_gap) -> adiabatic schedule
# - free_energy(utilities, beta) -> F(β) = -<w,U> + β⁻¹H(w)
