"""
Connections on the Pacioli manifold: parallel transport, holonomy, curvature.

A connection A is a rule for transporting value along flows.
Curvature F = dA + A∧A measures path-dependence.
Flat connection (F=0) ↔ no arbitrage.

Reference: Buckley (2026) EGT, doi:10.5281/zenodo.20259495
"""

# TODO Phase 1: implement
# - Connection base class: gauge group (R_{>0}, ×)
# - parallel_transport(connection, path) -> holonomy scalar
# - curvature(connection) -> F tensor
# - is_flat(connection, tol=1e-8) -> bool
# - wilson_loop(connection, loop) -> holonomy (= arbitrage factor)
