"""
TIR geometry types: admissibility structures for the four geometry classes.

Each geometry is an admissibility filter: it takes a utility vector U and
returns a masked version where inadmissible candidates have U_i = -∞.

The four types form a non-associativity hierarchy (Paper 294, §4):

  Abelian   — graph connectivity (associative, commutative)
  Fano      — Fano-plane collinearity (non-associative, alternative)
  G2        — Fisher-metric compatibility (non-associative Lie group)
  Catalan   — tree-bracket consistency (non-associative magma)

Usage pattern:
    geom   = FanoGeometry()
    U_mask = geom.mask(U, source=0)
    w      = gibbs_weights(U_mask, beta=2.0)

Reference: Buckley (2026) TIR, doi:10.5281/zenodo.20237288
"""

import enum
from dataclasses import dataclass, field
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


NEG_INF = -1e30   # proxy for -∞ that is JAX-safe (not jnp.inf, avoids NaN in softmax)


# ---------------------------------------------------------------------------
# Geometry type enum
# ---------------------------------------------------------------------------

class GeometryType(enum.Enum):
    ABELIAN = "abelian"
    FANO    = "fano"
    G2      = "g2"
    CATALAN = "catalan"


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------

class Geometry:
    """
    Abstract admissibility geometry.

    Subclasses implement mask(utilities, **kwargs) -> jax.Array
    which returns utilities with inadmissible entries set to NEG_INF.
    """

    geometry_type: GeometryType

    def mask(self, utilities: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError

    def admissible_mask(self, n: int, **kwargs) -> jax.Array:
        """Boolean mask: True = admissible. Shape (n,)."""
        dummy = jnp.zeros(n)
        masked = self.mask(dummy, **kwargs)
        return masked > NEG_INF / 2


# ---------------------------------------------------------------------------
# 1. Abelian network geometry
# ---------------------------------------------------------------------------

@dataclass
class AbelianGeometry(Geometry):
    """
    Admissibility = graph reachability from source node.

    Candidates are agents i ∈ {0, …, n-1}. Agent i is admissible as a
    routing target only if it is reachable from the source in the network.

    adjacency[i,j] = 1 if there is a directed edge i→j, else 0.
    Reachability is computed via BFS over the adjacency matrix.

    This is the simplest geometry: path composition is associative and
    commutative (Abelian). Standard softmax with connectivity masking.
    """

    geometry_type = GeometryType.ABELIAN
    adjacency: np.ndarray   # shape (n, n), integer or bool

    def __post_init__(self):
        n, m = self.adjacency.shape
        if n != m:
            raise ValueError(f"adjacency must be square, got ({n},{m})")
        self.adjacency = np.asarray(self.adjacency, dtype=bool)

    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    def reachable_from(self, source: int) -> np.ndarray:
        """BFS: returns boolean array of nodes reachable from source."""
        n = self.n_nodes
        visited = np.zeros(n, dtype=bool)
        queue = [source]
        visited[source] = True
        while queue:
            node = queue.pop(0)
            for nbr in np.where(self.adjacency[node])[0]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        return visited

    def mask(self, utilities: jax.Array, source: int = 0) -> jax.Array:
        """
        Mask utilities by reachability from source.

        Args:
            utilities: shape (n,)
            source: index of the source node

        Returns:
            utilities with unreachable nodes set to NEG_INF.
        """
        reachable = self.reachable_from(source)
        mask = jnp.array(reachable, dtype=jnp.float32)
        return jnp.where(mask, utilities, NEG_INF)

    @staticmethod
    def complete(n: int) -> 'AbelianGeometry':
        """Complete graph: all agents reachable from any source."""
        adj = np.ones((n, n), dtype=bool)
        np.fill_diagonal(adj, False)
        return AbelianGeometry(adjacency=adj)

    @staticmethod
    def chain(n: int) -> 'AbelianGeometry':
        """Chain graph: 0→1→2→…→n-1."""
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n - 1):
            adj[i, i + 1] = True
        return AbelianGeometry(adjacency=adj)

    def __repr__(self) -> str:
        return f"AbelianGeometry({self.n_nodes} nodes)"


# ---------------------------------------------------------------------------
# 2. Fano simplicial complex geometry
# ---------------------------------------------------------------------------

# The 7 lines of the Fano plane (each line is a set of 3 collinear points)
FANO_LINES: tuple[frozenset, ...] = (
    frozenset({0, 1, 3}),
    frozenset({1, 2, 4}),
    frozenset({2, 3, 5}),
    frozenset({3, 4, 6}),
    frozenset({4, 5, 0}),
    frozenset({5, 6, 1}),
    frozenset({6, 0, 2}),
)

# Precomputed: for each pair (i,j), the set of k that complete a Fano line
def _build_fano_collinear() -> dict[frozenset, list[int]]:
    """For each pair (i,j), which k makes {i,j,k} a Fano line?"""
    col = {}
    for line in FANO_LINES:
        pts = list(line)
        for a, b, c in [(0,1,2),(0,2,1),(1,2,0)]:
            key = frozenset({pts[a], pts[b]})
            col.setdefault(key, []).append(pts[c])
    return col

_FANO_COLLINEAR = _build_fano_collinear()


@dataclass
class FanoGeometry(Geometry):
    """
    Admissibility = collinearity on the Fano plane.

    The Fano plane has 7 points and 7 lines; each line contains 3 points.
    Candidates must be indexed 0–6 (the 7 Fano points).

    In triple-routing mode: a triple (A,B,C) is admissible iff {A,B,C} lies
    on a Fano line. The octonion associator is zero for such triples.

    In single-candidate mode (mask): given a fixed anchor pair (i,j), the
    admissible third candidates are those k such that {i,j,k} is a Fano line.

    This is a non-associative, alternative geometry. The admissibility
    algebra is the octonion multiplication structure (Baez 2002).
    """

    geometry_type = GeometryType.FANO
    n_candidates: int = 7   # must be 7 for standard Fano plane

    def __post_init__(self):
        if self.n_candidates != 7:
            raise ValueError(
                f"FanoGeometry requires exactly 7 candidates (the Fano points); "
                f"got {self.n_candidates}"
            )

    @staticmethod
    def is_collinear(i: int, j: int, k: int) -> bool:
        """True iff {i,j,k} lies on a Fano line."""
        return frozenset({i, j, k}) in {frozenset(line) for line in FANO_LINES}

    @staticmethod
    def collinear_with(i: int, j: int) -> list[int]:
        """All k such that {i,j,k} is a Fano line."""
        if i == j:
            return []
        return _FANO_COLLINEAR.get(frozenset({i, j}), [])

    def mask(self, utilities: jax.Array, anchor_i: int = 0, anchor_j: int = 1) -> jax.Array:
        """
        Mask utilities to candidates collinear with (anchor_i, anchor_j).

        Given two fixed Fano points as anchors, admits only the unique third
        point on the same Fano line.

        Args:
            utilities: shape (7,)
            anchor_i, anchor_j: two anchor Fano points

        Returns:
            utilities with non-collinear candidates set to NEG_INF.
        """
        admissible = self.collinear_with(anchor_i, anchor_j)
        mask_np = np.zeros(7, dtype=np.float32)
        for idx in admissible:
            mask_np[idx] = 1.0
        mask = jnp.array(mask_np)
        return jnp.where(mask, utilities, NEG_INF)

    def triple_score(self, i: int, j: int, k: int) -> float:
        """
        Admissibility score for triple (i,j,k).
        Returns 0.0 if collinear (admissible), -∞ if not.
        """
        return 0.0 if self.is_collinear(i, j, k) else NEG_INF

    def all_lines(self) -> tuple[frozenset, ...]:
        return FANO_LINES

    def __repr__(self) -> str:
        return "FanoGeometry(7 points, 7 lines)"


# ---------------------------------------------------------------------------
# 3. G₂ statistical manifold geometry
# ---------------------------------------------------------------------------

@dataclass
class G2Geometry(Geometry):
    """
    Admissibility = Fisher-metric compatibility on the G₂ statistical manifold.

    A gradient update δθ is admissible if its Fisher-normalised magnitude is
    below a threshold:

        ⟨δθ, Ψ(θ)⁻¹ δθ⟩ ≤ threshold

    where Ψ(θ) is the Fisher information matrix at current parameters θ.

    In practice Ψ is approximated by its diagonal (per-parameter gradient
    variance), making the check per-coordinate. Stale gradients with large
    Fisher-normalised norm are assigned low Gibbs weight.

    The G₂ manifold is the automorphism group of the octonion algebra and
    acts on the Fano plane, directly connecting this to FanoGeometry.

    This is a non-associative Lie group geometry.
    """

    geometry_type = GeometryType.G2
    threshold: float = 1.0

    def mask(
        self,
        utilities: jax.Array,
        gradients: jax.Array,
        fisher_diag: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Mask gradient candidates by Fisher-metric admissibility.

        Args:
            utilities: shape (n,) — one score per gradient candidate
            gradients: shape (n, d) — n candidate gradient vectors of dimension d
            fisher_diag: shape (d,) — diagonal of Fisher matrix (default: ones)

        Returns:
            utilities with Fisher-inadmissible candidates set to NEG_INF.
        """
        n, d = gradients.shape
        if fisher_diag is None:
            fisher_diag = jnp.ones(d)
        # Fisher-normalised squared norm: ⟨g, Ψ⁻¹ g⟩ = Σ_k g_k² / ψ_k
        inv_fisher = 1.0 / (fisher_diag + 1e-10)
        fisher_norms = jnp.sum(gradients ** 2 * inv_fisher[None, :], axis=1)  # (n,)
        admissible = (fisher_norms <= self.threshold).astype(jnp.float32)
        return jnp.where(admissible, utilities, NEG_INF)

    def fano_compatibility(self, gradient: jax.Array, fano_basis: jax.Array) -> jax.Array:
        """
        Measure how well a gradient aligns with the Fano-Fisher basis.

        The G₂ manifold acts on the Fano plane; gradient compatibility with the
        7 Fano directions measures its 'Fano-Fisher score'.

        Args:
            gradient: shape (d,)
            fano_basis: shape (7, d) — seven Fano-direction basis vectors

        Returns:
            shape (7,) — alignment scores (cosine similarity) with each Fano direction.
        """
        norms = jnp.linalg.norm(fano_basis, axis=1, keepdims=True) + 1e-10
        fano_unit = fano_basis / norms
        g_norm = jnp.linalg.norm(gradient) + 1e-10
        return jnp.dot(fano_unit, gradient) / g_norm   # (7,)

    def __repr__(self) -> str:
        return f"G2Geometry(threshold={self.threshold})"


# ---------------------------------------------------------------------------
# 4. Catalan tree geometry
# ---------------------------------------------------------------------------

def _catalan(n: int) -> int:
    """nth Catalan number C_n = C(2n,n)/(n+1)."""
    from math import comb
    return comb(2 * n, n) // (n + 1)


def _all_binary_trees(leaves: list) -> list:
    """
    Generate all full binary trees over a list of leaves.
    Returns list of nested tuples: a leaf l is represented as l,
    an internal node as (left_subtree, right_subtree).
    """
    if len(leaves) == 1:
        return [leaves[0]]
    trees = []
    for split in range(1, len(leaves)):
        for left in _all_binary_trees(leaves[:split]):
            for right in _all_binary_trees(leaves[split:]):
                trees.append((left, right))
    return trees


def _tree_coalitions(tree, coalition: frozenset | None = None) -> list[frozenset]:
    """Extract all sub-coalitions (subtrees) from a binary tree."""
    if not isinstance(tree, tuple):
        return [frozenset({tree})]
    left, right = tree
    left_leaves = _tree_leaves(left)
    right_leaves = _tree_leaves(right)
    result = [frozenset(left_leaves | right_leaves)]
    result.extend(_tree_coalitions(left))
    result.extend(_tree_coalitions(right))
    return result


def _tree_leaves(tree) -> frozenset:
    if not isinstance(tree, tuple):
        return frozenset({tree})
    return _tree_leaves(tree[0]) | _tree_leaves(tree[1])


@dataclass
class CatalanGeometry(Geometry):
    """
    Admissibility = tree-bracket consistency for cooperative game attribution.

    The Catalan number C_{n-1} counts the number of rooted binary trees on n
    leaves. Each tree defines a bracket order encoding which sub-coalition formed
    first. A coalition is admissible only if it appears as a subtree of the
    selected bracket structure.

    This is the richest non-associative geometry: a different bracket order
    gives a different geometry and a different characteristic function.
    Changing (A·B)·C to A·(B·C) changes which coalitions are admissible.

    n_players: number of players (leaves of the trees).
    """

    geometry_type = GeometryType.CATALAN
    n_players: int

    @property
    def n_trees(self) -> int:
        return _catalan(self.n_players - 1)

    def all_trees(self) -> list:
        """All rooted binary trees on n_players leaves."""
        return _all_binary_trees(list(range(self.n_players)))

    def admissible_coalitions(self, tree) -> list[frozenset]:
        """
        All coalitions admissible under a given tree bracket order.

        Returns list of frozensets, one per subtree of the tree.
        """
        return _tree_coalitions(tree)

    def mask(
        self,
        utilities: jax.Array,
        tree=None,
        coalition_indices: Optional[dict] = None,
    ) -> jax.Array:
        """
        Mask utilities over coalitions by tree-bracket admissibility.

        Args:
            utilities: shape (n,) where n = number of coalitions being scored
            tree: a binary tree (nested tuple of player indices); if None,
                  uses the left-associative tree ((0,1),2,…)
            coalition_indices: dict mapping frozenset → utility index.
                  If None, candidates are assumed to be the 2^n_players-2
                  non-trivial coalitions in lexicographic order.

        Returns:
            utilities with inadmissible coalition indices set to NEG_INF.
        """
        if tree is None:
            leaves = list(range(self.n_players))
            tree = leaves[0]
            for leaf in leaves[1:]:
                tree = (tree, leaf)

        admissible = self.admissible_coalitions(tree)

        if coalition_indices is None:
            # Default: all non-trivial subsets in sorted order
            all_coalitions = sorted(
                [frozenset(s) for s in _power_set(range(self.n_players))
                 if 1 < len(s) <= self.n_players],
                key=lambda s: (len(s), sorted(s)),
            )
            admissible_set = set(admissible)
            mask = jnp.array(
                [1.0 if c in admissible_set else 0.0 for c in all_coalitions],
                dtype=jnp.float32,
            )
        else:
            mask = jnp.zeros(len(utilities), dtype=jnp.float32)
            for coal in admissible:
                idx = coalition_indices.get(coal)
                if idx is not None:
                    mask = mask.at[idx].set(1.0)

        if mask.shape[0] != utilities.shape[0]:
            raise ValueError(
                f"mask length {mask.shape[0]} != utilities length {utilities.shape[0]}. "
                "Pass coalition_indices or ensure utilities has one entry per non-trivial coalition."
            )
        return jnp.where(mask, utilities, NEG_INF)

    def __repr__(self) -> str:
        return f"CatalanGeometry(n_players={self.n_players}, C_{self.n_players-1}={self.n_trees} trees)"


def _power_set(iterable):
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# ---------------------------------------------------------------------------
# Geometry registry
# ---------------------------------------------------------------------------

def geometry_type_of(geom: Geometry) -> GeometryType:
    return geom.geometry_type
