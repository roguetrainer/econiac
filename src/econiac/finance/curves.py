"""
Yield curves as temporal connections on the Pacioli manifold.

The forward rate f(t,T) = -∂ln P(t,T)/∂T is the connection coefficient.
HJM no-arbitrage condition = flatness of the forward-rate connection.
LMM is the lattice (discrete-time) gauge theory version.

Reference: Buckley (2026) Term Structure Bundles, doi:10.5281/zenodo.20244445
"""

# TODO Phase 2: implement
# - YieldCurve(maturities, discount_factors) — connection on time bundle
# - forward_rate(curve, t, T) -> f(t,T)
# - hjm_drift(curve, sigma) -> drift enforcing flatness (HJM condition)
# - holonomy(curve, t1, t2) -> P(t1,t2) as parallel transport
# - from_quantlib(ql_curve) -> YieldCurve  [QuantLib adapter]
