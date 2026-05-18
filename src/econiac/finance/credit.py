"""
Credit risk: survival probabilities as parallel transport; CVA as curvature integral.

S(t) = exp(-∫λ(s)ds) is holonomy of the hazard rate connection along the time axis.
CVA = E[LGD · D(τ) · 1_{τ<T}] = ∫ curvature over the trade lifetime.
Default correlation = non-zero curvature between two obligor connections.

References:
  Buckley (2026) Credit Bundles, doi:10.5281/zenodo.20257596
  Buckley (2026) XVA as Curvature, doi:10.5281/zenodo.20257724
"""

# TODO Phase 2: implement
# - HazardRateConnection(maturities, hazard_rates)
# - survival_probability(connection, t) -> S(t)
# - cva(exposure_profile, hazard_connection, discount_curve, lgd) -> float
# - default_correlation(connection_A, connection_B) -> curvature tensor
# - xva_surface(portfolio, counterparty, funding_curve) -> XVA breakdown
