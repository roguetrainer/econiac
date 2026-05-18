"""
Example: triangular arbitrage as non-zero holonomy on the currency bundle.

USD → EUR → GBP → USD: if the product of exchange rates ≠ 1, the holonomy
is non-zero and an arbitrage exists. econiac computes this as curvature of
the FX connection on the Pacioli manifold.

Planned: requires econiac.finance.fx (Phase 2)
"""

# Spot rates (illustrative)
S_USDEUR = 0.92   # 1 USD = 0.92 EUR
S_EURGBP = 0.86   # 1 EUR = 0.86 GBP
S_GBPUSD = 1.27   # 1 GBP = 1.27 USD

# Holonomy = product around the loop (should be 1.0 for no arbitrage)
holonomy = S_USDEUR * S_EURGBP * S_GBPUSD
print(f"Holonomy (USD→EUR→GBP→USD): {holonomy:.6f}")
print(f"Arbitrage residual: {holonomy - 1.0:.6f}")
print(f"No-arbitrage condition satisfied: {abs(holonomy - 1.0) < 1e-4}")

# TODO: replace with econiac.finance.fx.holonomy() when Phase 2 is implemented
