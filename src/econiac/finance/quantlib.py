"""
QuantLib adapter: wrap QuantLib curves and instruments as Pacioli connections.

QuantLib computes risk-free curves, option prices, risk sensitivities.
econiac wraps these as connections on the Pacioli manifold, enabling:
  - XVA as curvature integrals (Paper 299)
  - Gradient-based calibration through QuantLib outputs
  - Gauge-consistent portfolio aggregation

QuantLib is not replaced — it is the upstream curve engine.
"""

# TODO Phase 2: implement
# Requires: pip install QuantLib
#
# - ql_curve_to_connection(ql_yield_term_structure) -> YieldCurve
# - ql_cds_to_hazard(ql_cds_helper, settlement_date) -> HazardRateConnection
# - ql_portfolio_xva(instruments, counterparty, netting_set) -> XVAResult
#   (uses QuantLib for pricing, econiac for curvature integration)
