"""
Foreign exchange as connection curvature on the Pacioli manifold.

S_{A/B} is a connection on the currency bundle.
Triangular arbitrage = non-zero holonomy: USD→EUR→GBP→USD ≠ 1.
No-arbitrage = flat connection: F = dA + A∧A = 0.
Interest rate parity = flatness of combined IR + FX connection.

Reference: Buckley (2026) Currency Bundles, doi:10.5281/zenodo.20242355
"""

# TODO Phase 2: implement
# - FXConnection(base, quote, spot_rate, ir_base, ir_quote)
# - holonomy(path_of_currencies) -> arbitrage factor (1.0 = no arbitrage)
# - curvature(fx_matrix) -> F tensor measuring arbitrage
# - covered_interest_parity_residual(fx, ir_domestic, ir_foreign) -> float
