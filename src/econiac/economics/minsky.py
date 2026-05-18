"""
Minsky-compatible stock-flow DSL: differentiable drop-in for Keen's Minsky.

Keen's Minsky uses Godley tables (stock-flow consistent balance sheets) —
directly the Pacioli manifold. This module provides a compatible DSL that
compiles to JAX for differentiability and GPU acceleration.

The Lotka-Volterra debt dynamics (Keen predator-prey model) is the canonical
example: employment rate and debt ratio as coupled differential equations on
the Pacioli manifold, calibratable by gradient descent.

Reference: Buckley (2026) EGT, doi:10.5281/zenodo.20259495
"""

# TODO Phase 3: implement
# - GodleyTable(rows, cols) — balance sheet as Pacioli manifold node
# - MinskySFCModel — SFC model specified via Godley tables (Minsky-compatible API)
# - keen_predator_prey(alpha, beta, gamma, delta) -> MinskySFCModel
# - from_minsky_file(path) -> MinskySFCModel  [read Minsky .mky files]
# - to_jax(model) -> JAX-compiled forward pass
