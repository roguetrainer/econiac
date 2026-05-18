"""
Pacioli manifold: directed graph of institutional money flows with ∂²=0.

The conservation law ∂²=0 (every debit has a credit) is enforced by construction
via the boundary operator on the flow graph. Homology groups H_0, H_1, H_2
classify connected components, circular flows, and enclosed regions.

Reference: Buckley (2026), doi:10.5281/zenodo.20234853
"""

# TODO Phase 1: implement PacioliManifold class
# - from_godley_table(rows, cols, entries) -> PacioliManifold
# - boundary_operator() -> sparse matrix ∂ such that ∂²=0
# - homology(k) -> H_k as vector space
# - enforce_conservation(flows) -> flows satisfying ∂·flows=0
