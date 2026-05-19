"""Tests for econiac.core.geometry — four TIR admissibility geometry types."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.core.geometry import (
    GeometryType,
    AbelianGeometry,
    FanoGeometry,
    FANO_LINES,
    G2Geometry,
    CatalanGeometry,
    geometry_type_of,
    NEG_INF,
)


# ---------------------------------------------------------------------------
# AbelianGeometry
# ---------------------------------------------------------------------------

class TestAbelianGeometry:
    def test_complete_all_reachable(self):
        geom = AbelianGeometry.complete(4)
        reachable = geom.reachable_from(0)
        assert all(reachable)

    def test_chain_reachability(self):
        geom = AbelianGeometry.chain(4)   # 0→1→2→3
        assert geom.reachable_from(0).tolist() == [True, True, True, True]
        # From node 2 only 2,3 reachable
        assert geom.reachable_from(2).tolist() == [False, False, True, True]

    def test_disconnected_graph(self):
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=bool)
        geom = AbelianGeometry(adjacency=adj)
        reachable = geom.reachable_from(0)
        assert reachable[0] and reachable[1]
        assert not reachable[2] and not reachable[3]

    def test_mask_sets_unreachable_to_neg_inf(self):
        geom = AbelianGeometry.chain(3)  # 0→1→2
        U = jnp.array([1.0, 2.0, 3.0])
        U_masked = geom.mask(U, source=0)
        assert jnp.allclose(U_masked[0], 1.0)
        assert jnp.allclose(U_masked[1], 2.0)
        assert jnp.allclose(U_masked[2], 3.0)

    def test_mask_source_2_blocks_0_1(self):
        geom = AbelianGeometry.chain(3)  # 0→1→2, from 2: only 2 reachable
        U = jnp.array([1.0, 2.0, 3.0])
        U_masked = geom.mask(U, source=2)
        assert U_masked[0] <= NEG_INF / 2
        assert U_masked[1] <= NEG_INF / 2
        assert jnp.allclose(U_masked[2], 3.0)

    def test_non_square_adjacency_raises(self):
        with pytest.raises(ValueError):
            AbelianGeometry(adjacency=np.ones((3, 4), dtype=bool))

    def test_geometry_type(self):
        geom = AbelianGeometry.complete(3)
        assert geometry_type_of(geom) == GeometryType.ABELIAN

    def test_repr(self):
        assert 'Abelian' in repr(AbelianGeometry.complete(3))

    def test_admissible_mask_boolean(self):
        geom = AbelianGeometry.chain(4)
        m = geom.admissible_mask(4, source=0)
        assert m.shape == (4,)
        assert m.dtype == jnp.float32 or m.sum() > 0


# ---------------------------------------------------------------------------
# FanoGeometry
# ---------------------------------------------------------------------------

class TestFanoGeometry:
    def test_seven_lines(self):
        assert len(FANO_LINES) == 7

    def test_each_line_has_three_points(self):
        for line in FANO_LINES:
            assert len(line) == 3

    def test_collinear_known_triple(self):
        geom = FanoGeometry()
        assert geom.is_collinear(0, 1, 3)   # {0,1,3} is line 0

    def test_non_collinear_triple(self):
        geom = FanoGeometry()
        assert not geom.is_collinear(0, 1, 2)   # {0,1,2} is not a Fano line

    def test_each_pair_has_exactly_one_collinear(self):
        """Each pair of Fano points lies on exactly one line → one collinear third."""
        geom = FanoGeometry()
        for i in range(7):
            for j in range(7):
                if i != j:
                    col = geom.collinear_with(i, j)
                    assert len(col) == 1, f"pair ({i},{j}) has {len(col)} collinear points"

    def test_mask_admits_one_candidate(self):
        geom = FanoGeometry()
        U = jnp.ones(7)
        U_masked = geom.mask(U, anchor_i=0, anchor_j=1)
        n_admissible = int((U_masked > NEG_INF / 2).sum())
        assert n_admissible == 1

    def test_mask_correct_candidate(self):
        """anchor (0,1) → collinear third is 3."""
        geom = FanoGeometry()
        U = jnp.arange(7, dtype=jnp.float32)
        U_masked = geom.mask(U, anchor_i=0, anchor_j=1)
        # Only index 3 should be finite
        assert U_masked[3] > NEG_INF / 2
        for k in range(7):
            if k != 3:
                assert U_masked[k] <= NEG_INF / 2

    def test_triple_score_collinear(self):
        geom = FanoGeometry()
        assert geom.triple_score(0, 1, 3) == 0.0

    def test_triple_score_non_collinear(self):
        geom = FanoGeometry()
        assert geom.triple_score(0, 1, 2) <= NEG_INF / 2

    def test_wrong_n_candidates_raises(self):
        with pytest.raises(ValueError):
            FanoGeometry(n_candidates=5)

    def test_geometry_type(self):
        geom = FanoGeometry()
        assert geometry_type_of(geom) == GeometryType.FANO

    def test_repr(self):
        assert 'Fano' in repr(FanoGeometry())

    def test_all_lines_returns_seven(self):
        geom = FanoGeometry()
        assert len(geom.all_lines()) == 7

    def test_fano_symmetry(self):
        """Collinearity is symmetric: is_collinear(i,j,k) = is_collinear(k,j,i)."""
        geom = FanoGeometry()
        assert geom.is_collinear(0, 1, 3) == geom.is_collinear(3, 1, 0)
        assert geom.is_collinear(0, 1, 2) == geom.is_collinear(2, 1, 0)


# ---------------------------------------------------------------------------
# G2Geometry
# ---------------------------------------------------------------------------

class TestG2Geometry:
    def test_mask_admits_low_norm_gradients(self):
        geom = G2Geometry(threshold=1.0)
        n, d = 4, 3
        gradients = jnp.array([
            [0.1, 0.1, 0.1],   # small norm → admissible
            [10., 10., 10.],   # large norm → inadmissible
            [0.5, 0.0, 0.0],   # medium → admissible
            [5.0, 5.0, 5.0],   # large → inadmissible
        ])
        U = jnp.zeros(4)
        U_masked = geom.mask(U, gradients=gradients)
        assert U_masked[0] > NEG_INF / 2
        assert U_masked[1] <= NEG_INF / 2
        assert U_masked[2] > NEG_INF / 2
        assert U_masked[3] <= NEG_INF / 2

    def test_fisher_scaling(self):
        """Large Fisher diagonal makes even big gradients admissible."""
        geom = G2Geometry(threshold=1.0)
        gradients = jnp.array([[10.0, 10.0, 10.0]])
        fisher_diag = jnp.array([1000.0, 1000.0, 1000.0])
        U = jnp.zeros(1)
        U_masked = geom.mask(U, gradients=gradients, fisher_diag=fisher_diag)
        assert U_masked[0] > NEG_INF / 2   # normalised norm << threshold

    def test_fano_compatibility_shape(self):
        geom = G2Geometry()
        gradient = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fano_basis = jnp.eye(7)
        scores = geom.fano_compatibility(gradient, fano_basis)
        assert scores.shape == (7,)

    def test_fano_compatibility_aligned(self):
        """A gradient aligned with one Fano direction scores 1 on that direction."""
        geom = G2Geometry()
        gradient = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fano_basis = jnp.eye(7)
        scores = geom.fano_compatibility(gradient, fano_basis)
        assert jnp.allclose(scores[0], 1.0, atol=1e-5)
        assert jnp.allclose(scores[1:], jnp.zeros(6), atol=1e-5)

    def test_geometry_type(self):
        geom = G2Geometry()
        assert geometry_type_of(geom) == GeometryType.G2

    def test_repr(self):
        assert 'G2' in repr(G2Geometry(threshold=2.0))


# ---------------------------------------------------------------------------
# CatalanGeometry
# ---------------------------------------------------------------------------

class TestCatalanGeometry:
    def test_n_trees_3_players(self):
        """C_2 = 2 trees for 3 players."""
        geom = CatalanGeometry(n_players=3)
        assert geom.n_trees == 2

    def test_n_trees_4_players(self):
        """C_3 = 5 trees for 4 players."""
        geom = CatalanGeometry(n_players=4)
        assert geom.n_trees == 5

    def test_all_trees_count(self):
        geom = CatalanGeometry(n_players=3)
        trees = geom.all_trees()
        assert len(trees) == 2

    def test_admissible_coalitions_left_tree(self):
        """Left-associative tree ((0,1),2): coalitions are {0,1}, {0,1,2}."""
        geom = CatalanGeometry(n_players=3)
        tree = ((0, 1), 2)
        admissible = geom.admissible_coalitions(tree)
        admissible_sets = set(admissible)
        assert frozenset({0, 1}) in admissible_sets
        assert frozenset({0, 1, 2}) in admissible_sets

    def test_admissible_coalitions_right_tree(self):
        """Right-associative tree (0,(1,2)): coalitions are {1,2}, {0,1,2}."""
        geom = CatalanGeometry(n_players=3)
        tree = (0, (1, 2))
        admissible = geom.admissible_coalitions(tree)
        admissible_sets = set(admissible)
        assert frozenset({1, 2}) in admissible_sets
        assert frozenset({0, 1, 2}) in admissible_sets

    def test_different_trees_different_admissibility(self):
        """(A·B)·C and A·(B·C) give different admissible coalition sets."""
        geom = CatalanGeometry(n_players=3)
        left = set(geom.admissible_coalitions(((0, 1), 2)))
        right = set(geom.admissible_coalitions((0, (1, 2))))
        assert left != right

    def test_mask_length_matches(self):
        geom = CatalanGeometry(n_players=3)
        # Non-trivial coalitions for 3 players: {0,1},{0,2},{1,2},{0,1,2} → 4
        U = jnp.zeros(4)
        tree = ((0, 1), 2)
        U_masked = geom.mask(U, tree=tree)
        assert U_masked.shape == (4,)

    def test_mask_admits_correct_coalitions(self):
        """Left tree ((0,1),2): {0,1} and {0,1,2} admissible, {0,2} and {1,2} not."""
        geom = CatalanGeometry(n_players=3)
        # Non-trivial coalitions (sorted by size then lex): {0,1},{0,2},{1,2},{0,1,2}
        U = jnp.ones(4)
        tree = ((0, 1), 2)
        U_masked = geom.mask(U, tree=tree)
        # index 0 = {0,1} → admissible
        # index 1 = {0,2} → NOT admissible under left tree
        # index 2 = {1,2} → NOT admissible under left tree
        # index 3 = {0,1,2} → admissible (grand coalition is always in)
        assert U_masked[0] > NEG_INF / 2   # {0,1}
        assert U_masked[1] <= NEG_INF / 2  # {0,2}
        assert U_masked[2] <= NEG_INF / 2  # {1,2}
        assert U_masked[3] > NEG_INF / 2   # {0,1,2}

    def test_geometry_type(self):
        geom = CatalanGeometry(n_players=4)
        assert geometry_type_of(geom) == GeometryType.CATALAN

    def test_repr(self):
        r = repr(CatalanGeometry(n_players=4))
        assert 'Catalan' in r
        assert '5' in r   # C_3 = 5
