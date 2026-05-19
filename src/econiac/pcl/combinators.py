"""
Pacioli Combinator Library (PCL): conservation-enforcing DSL for financial computation.

Every PCL computation is a function  f: BalanceSheet → BalanceSheet  that preserves
the Pacioli conservation law ∂²=0 (every debit has a credit, column sums = 0).

Core combinators:
  choose(β, f, g)        — MGE-weighted mixture of f and g at rationality β
  sequence(f, g)         — f then g (non-commutative for non-Abelian gauge)
  parallel(f, g)         — f and g simultaneously; outputs added
  fold(β, [f₁,…,fₙ])    — MGE fold over a list of computations
  identity()             — pass-through (unit of sequence)
  zero()                 — zero transformation (unit of parallel)
  scale(α, f)            — multiply all flows by scalar α
  flow(from, to, inst, amount) — atomic double-entry transfer
  repeat(n, f)           — apply f n times (discretised ODE step)

Type system:
  typecheck(comp)        — verify ∂²=0 is preserved on a probe balance sheet
  typecheck_strict(comp) — verify ∂²=0 on n random probe balance sheets
  compile(comp)          — return a JAX-jit-compiled version of the computation

Tree inspection:
  depth(comp)            — maximum depth of computation tree
  leaves(comp)           — all leaf nodes
  pretty(comp)           — pretty-print the tree

Reference: Buckley (2026) PCL, doi:10.5281/zenodo.20262070
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.ensemble import gibbs_weights
from econiac.core.manifold import BalanceSheet


# ---------------------------------------------------------------------------
# Computation type — the DSL node
# ---------------------------------------------------------------------------

@dataclass
class Computation:
    """
    A PCL computation node: a named, composable BalanceSheet → BalanceSheet function.

    name:     human-readable label for debugging and repr
    fn:       the actual function; must preserve Pacioli conservation
    arity:    number of children (0 = leaf, 1 = unary, 2+ = n-ary)
    children: child Computation nodes (for tree inspection)
    """
    name:     str
    fn:       Callable[[BalanceSheet], BalanceSheet]
    arity:    int = 0
    children: list['Computation'] = field(default_factory=list)

    def __call__(self, bs: BalanceSheet) -> BalanceSheet:
        return self.fn(bs)

    def __repr__(self) -> str:
        if self.children:
            child_str = ", ".join(repr(c) for c in self.children)
            return f"{self.name}({child_str})"
        return self.name


# ---------------------------------------------------------------------------
# Leaf combinators
# ---------------------------------------------------------------------------

def identity() -> Computation:
    """
    Identity computation: pass-through, no transformation.

    Unit of sequence: sequence(f, identity()) == f == sequence(identity(), f).
    Preserves ∂²=0 trivially.
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        return bs
    return Computation(name="identity", fn=fn, arity=0)


def zero() -> Computation:
    """
    Zero computation: maps any BalanceSheet to the all-zeros BalanceSheet.

    Unit of parallel: parallel(f, zero()) == f.
    The zero BalanceSheet satisfies ∂²=0 (all column sums = 0).
    Note: zero() is NOT the unit of parallel — identity() is, since zero()'s
    delta equals -bs.positions, not 0.
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        return BalanceSheet(
            positions=jnp.zeros_like(bs.positions),
            sectors=bs.sectors,
            instruments=bs.instruments,
        )
    return Computation(name="zero", fn=fn, arity=0)


def scale(alpha: float, f: Computation) -> Computation:
    """
    Scale the output positions of f by scalar α.

    Conservation: ∂²=0 is linear — scaling a zero-column-sum matrix gives a
    zero-column-sum matrix. scale(0, f) == zero(); scale(1, f) == f.
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        result = f(bs)
        return BalanceSheet(
            positions=alpha * result.positions,
            sectors=result.sectors,
            instruments=result.instruments,
        )
    return Computation(name=f"scale({alpha:.3g})", fn=fn, arity=1, children=[f])


def flow(
    from_sector: str,
    to_sector:   str,
    instrument:  str,
    amount:      float,
) -> Computation:
    """
    Atomic double-entry transfer: move `amount` of `instrument` from → to.

    Conservation is enforced by construction:
      from_sector loses `amount` (debit) = to_sector gains `amount` (credit).

    This is the primitive PCL operation; all other combinators compose flows.
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        sectors     = bs.sectors
        instruments = bs.instruments
        if from_sector not in sectors:
            raise ValueError(f"from_sector '{from_sector}' not in {sectors}")
        if to_sector not in sectors:
            raise ValueError(f"to_sector '{to_sector}' not in {sectors}")
        if instrument not in instruments:
            raise ValueError(f"instrument '{instrument}' not in {instruments}")

        i_from = sectors.index(from_sector)
        i_to   = sectors.index(to_sector)
        j_inst = instruments.index(instrument)

        delta = jnp.zeros_like(bs.positions)
        delta = delta.at[i_from, j_inst].add(-amount)
        delta = delta.at[i_to,   j_inst].add(+amount)

        return BalanceSheet(
            positions=bs.positions + delta,
            sectors=bs.sectors,
            instruments=bs.instruments,
        )

    name = f"flow({from_sector}→{to_sector},{instrument},{amount:.3g})"
    return Computation(name=name, fn=fn, arity=0)


# ---------------------------------------------------------------------------
# Binary / n-ary combinators
# ---------------------------------------------------------------------------

def sequence(f: Computation, g: Computation) -> Computation:
    """
    Sequential composition: apply f then g.

    Non-commutative: sequence(f, g) ≠ sequence(g, f) in general — like gauge
    transformations on the Pacioli manifold, order matters.

    Conservation: composition of two ∂²=0-preserving maps is ∂²=0-preserving.
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        return g(f(bs))
    return Computation(name="sequence", fn=fn, arity=2, children=[f, g])


def parallel(f: Computation, g: Computation) -> Computation:
    """
    Parallel composition: apply f and g to the same input; add their position deltas.

    Output = input + (f(input) − input) + (g(input) − input)

    Conservation: sum of two zero-column-sum delta matrices is zero-column-sum.
    Unit: parallel(f, identity()) == f  (identity has zero delta).
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        delta_f = f(bs).positions - bs.positions
        delta_g = g(bs).positions - bs.positions
        return BalanceSheet(
            positions=bs.positions + delta_f + delta_g,
            sectors=bs.sectors,
            instruments=bs.instruments,
        )
    return Computation(name="parallel", fn=fn, arity=2, children=[f, g])


def choose(beta: float, f: Computation, g: Computation) -> Computation:
    """
    MGE-weighted mixture of two computations at rationality β.

    Output positions = w_f · f(bs).positions + w_g · g(bs).positions
    where [w_f, w_g] = softmax(β · [V(f(bs)), V(g(bs))])
    and V(bs) = total net worth = positions.sum().

    At β=0: equal weights (maximum entropy, explore both paths).
    At β→∞: all weight on whichever computation produces higher net worth (argmax).

    Conservation: Gibbs-weighted average of ∂²=0 matrices is ∂²=0
    (w_f · 0 + w_g · 0 = 0 for column sums).

    This is the Maslov-Gibbs Einsum as a PCL combinator: at β→∞ it recovers
    the tropical (max,+) selection; at finite β it is the Gibbs ensemble average.
    """
    def fn(bs: BalanceSheet) -> BalanceSheet:
        result_f = f(bs)
        result_g = g(bs)
        v_f = float(result_f.net_worth().sum())
        v_g = float(result_g.net_worth().sum())
        w   = gibbs_weights(jnp.array([v_f, v_g]), beta=float(beta))
        mixed = float(w[0]) * result_f.positions + float(w[1]) * result_g.positions
        return BalanceSheet(
            positions=mixed,
            sectors=bs.sectors,
            instruments=bs.instruments,
        )
    return Computation(name=f"choose(β={beta:.3g})", fn=fn, arity=2, children=[f, g])


def fold(beta: float, computations: Sequence[Computation]) -> Computation:
    """
    MGE fold over N computations at rationality β.

    Generalises choose() to N alternatives. Each computation is evaluated;
    output is the Gibbs-weighted average, weighted by total net worth of each result.

    At β=0: uniform average over all N computations.
    At β→∞: winner-take-all (the computation producing highest net worth).

    Conservation: Gibbs-weighted average of ∂²=0 matrices is ∂²=0.
    """
    comps = list(computations)
    if not comps:
        raise ValueError("fold requires at least one computation")

    def fn(bs: BalanceSheet) -> BalanceSheet:
        results = [c(bs) for c in comps]
        values  = jnp.array([float(r.net_worth().sum()) for r in results])
        w       = gibbs_weights(values, beta=float(beta))
        mixed   = sum(float(w[k]) * results[k].positions for k in range(len(comps)))
        return BalanceSheet(
            positions=mixed,
            sectors=bs.sectors,
            instruments=bs.instruments,
        )

    names = ", ".join(c.name for c in comps)
    return Computation(
        name=f"fold(β={beta:.3g},[{names}])",
        fn=fn,
        arity=len(comps),
        children=comps,
    )


def repeat(n: int, f: Computation) -> Computation:
    """
    Apply f exactly n times: f∘f∘…∘f (n-fold composition).

    repeat(0, f) == identity().
    repeat(1, f) == f.
    repeat(T, euler_step) integrates an ODE for T timesteps.

    Conservation: preserved by induction — each step preserves ∂²=0.
    """
    if n < 0:
        raise ValueError(f"repeat requires n ≥ 0, got {n}")
    if n == 0:
        return identity()
    if n == 1:
        return Computation(name=f"repeat(1)", fn=f.fn, arity=1, children=[f])

    def fn(bs: BalanceSheet) -> BalanceSheet:
        result = bs
        for _ in range(n):
            result = f(result)
        return result

    return Computation(name=f"repeat({n})", fn=fn, arity=1, children=[f])


# ---------------------------------------------------------------------------
# Type system: conservation checker
# ---------------------------------------------------------------------------

def typecheck(comp: Computation, atol: float = 1e-4) -> bool:
    """
    Verify comp preserves ∂²=0 on a canonical probe BalanceSheet.

    Constructs a 3-sector × 2-instrument balanced balance sheet, applies comp,
    and checks that column sums of the result are ≈ 0.

    Returns True if conservation is preserved, False otherwise.
    """
    sectors     = ['households', 'firms', 'banks']
    instruments = ['deposits', 'loans']
    positions   = jnp.array([
        [ 100.0,    0.0],
        [   0.0, -80.0],
        [-100.0,  80.0],
    ])
    probe = BalanceSheet(positions=positions, sectors=sectors, instruments=instruments)

    try:
        result   = comp(probe)
        col_sums = result.positions.sum(axis=0)
        return bool(jnp.allclose(col_sums, jnp.zeros(2), atol=atol))
    except Exception:
        return False


def typecheck_strict(
    comp:      Computation,
    n_probes:  int   = 10,
    atol:      float = 1e-4,
    rng_seed:  int   = 42,
) -> bool:
    """
    Strict conservation check: test on n_probes random consistent BalanceSheets.

    Each probe is a randomly generated (4 × 2) zero-column-sum matrix.
    Returns True only if all probes pass.
    """
    rng         = np.random.default_rng(rng_seed)
    sectors     = ['s0', 's1', 's2', 's3']
    instruments = ['i0', 'i1']
    n_s, n_i    = len(sectors), len(instruments)

    for _ in range(n_probes):
        free  = rng.standard_normal((n_s - 1, n_i))
        last  = -free.sum(axis=0, keepdims=True)
        pos   = jnp.array(np.concatenate([free, last], axis=0))
        probe = BalanceSheet(positions=pos, sectors=sectors, instruments=instruments)
        try:
            result   = comp(probe)
            col_sums = result.positions.sum(axis=0)
            if not bool(jnp.allclose(col_sums, jnp.zeros(n_i), atol=atol)):
                return False
        except Exception:
            return False

    return True


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile(comp: Computation) -> Computation:
    """
    Return a JAX-jit-compiled version of comp.

    The compiled computation traces through the positions tensor via XLA,
    giving GPU/TPU acceleration. Sector/instrument metadata is treated as
    static Python state (captured in the closure).
    """
    def compiled_fn(bs: BalanceSheet) -> BalanceSheet:
        sectors     = bs.sectors
        instruments = bs.instruments

        @jax.jit
        def _inner(positions: jax.Array) -> jax.Array:
            result = comp(BalanceSheet(
                positions=positions,
                sectors=sectors,
                instruments=instruments,
            ))
            return result.positions

        return BalanceSheet(
            positions=_inner(bs.positions),
            sectors=sectors,
            instruments=instruments,
        )

    return Computation(
        name=f"compiled({comp.name})",
        fn=compiled_fn,
        arity=comp.arity,
        children=comp.children,
    )


# ---------------------------------------------------------------------------
# Tree inspection utilities
# ---------------------------------------------------------------------------

def depth(comp: Computation) -> int:
    """Maximum depth of the computation tree."""
    if not comp.children:
        return 0
    return 1 + max(depth(c) for c in comp.children)


def leaves(comp: Computation) -> list[Computation]:
    """All leaf nodes (arity=0) of the computation tree."""
    if not comp.children:
        return [comp]
    result = []
    for c in comp.children:
        result.extend(leaves(c))
    return result


def pretty(comp: Computation, indent: int = 0) -> str:
    """Pretty-print the computation tree."""
    prefix = "  " * indent
    if not comp.children:
        return f"{prefix}{comp.name}"
    lines = [f"{prefix}{comp.name}"]
    for c in comp.children:
        lines.append(pretty(c, indent + 1))
    return "\n".join(lines)
