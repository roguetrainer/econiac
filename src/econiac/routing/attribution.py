"""
Thermal Shapley values: differentiable attribution with latent bottleneck detection.

φ_i(β) = Gibbs-weighted average marginal contribution of player i.
Λ_i(β) = |∂φ_i/∂β| — latent bottleneck index.
As β→∞: Laplace concentration on the bottleneck permutation.
Pacioli constraint: Σ φ_i(β) = 0 for inter-sectoral SFC attribution.

Reference: Buckley (2026) Thermal Attribution, doi:10.5281/zenodo.20236870
"""

# TODO Phase 4: implement
# - thermal_shapley(value_function, n_players, beta) -> φ array
# - bottleneck_index(value_function, n_players, beta_range) -> Λ array
# - tropical_limit(value_function, n_players) -> bottleneck player index
# - pacioli_attribution(sfc_model, beta) -> sector attributions summing to 0
