"""
Pacioli Combinator Library (PCL): conservation-enforcing DSL for financial computation.

Combinators:
  choose(β, f, g)     — MGE-weighted mixture of f and g at rationality β
  sequence(f, g)      — f then g (non-commutative for non-Abelian gauge)
  parallel(f, g)      — f and g simultaneously, outputs combined by geometry
  fold(β, fs)         — MGE fold over a list of computations

The type system enforces Pacioli conservation: computations violating ∂²=0
are type errors. Compiles to JAX for GPU acceleration and autograd.

Reference: Buckley (2026) PCL, doi:10.5281/zenodo.20262070
"""

# TODO Phase 5: implement
# - choose(beta, f, g) -> Computation
# - sequence(f, g) -> Computation  [note: non-commutative]
# - parallel(f, g, geometry) -> Computation
# - fold(beta, computations) -> Computation
# - compile(computation) -> JAX-compiled function
# - typecheck(computation) -> bool  [verify ∂²=0 conservation]
