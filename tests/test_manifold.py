"""Tests for econiac.core.manifold — Pacioli manifold and balance sheets."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.core.manifold import (
    BalanceSheet,
    GodleyTable,
    PacioliManifold,
    HomologyGroups,
    three_sector_sfc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def simple_balance_sheet():
    """Three-sector, two-instrument consistent balance sheet."""
    positions = jnp.array([
        # deposits  loans
        [ 100.0,    0.0],   # households: asset (deposits)
        [   0.0,  -80.0],   # firms:      liability (loans)
        [-100.0,   80.0],   # banks:      liability (deposits), asset (loans)
    ])
    return BalanceSheet(
        positions=positions,
        sectors=['households', 'firms', 'banks'],
        instruments=['deposits', 'loans'],
    )


def simple_manifold():
    return PacioliManifold.from_edges(
        nodes=['households', 'firms', 'banks'],
        edges=[
            ('wages',    'firms',      'households'),
            ('deposits', 'households', 'banks'),
            ('loans',    'banks',      'firms'),
        ],
    )


# ---------------------------------------------------------------------------
# BalanceSheet tests
# ---------------------------------------------------------------------------

class TestBalanceSheet:
    def test_consistent_column_sums(self):
        bs = simple_balance_sheet()
        assert jnp.allclose(bs.column_sums(), jnp.zeros(2), atol=1e-6)

    def test_is_consistent(self):
        bs = simple_balance_sheet()
        assert bs.is_consistent()

    def test_inconsistent_detected(self):
        positions = jnp.array([[100.0, 0.0], [0.0, -50.0], [-100.0, 80.0]])
        bs = BalanceSheet(
            positions=positions,
            sectors=['h', 'f', 'b'],
            instruments=['deposits', 'loans'],
        )
        assert not bs.is_consistent()

    def test_net_worth(self):
        bs = simple_balance_sheet()
        nw = bs.net_worth()
        assert nw.shape == (3,)
        # households: 100, firms: -80, banks: -20
        assert jnp.allclose(nw, jnp.array([100.0, -80.0, -20.0]), atol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            BalanceSheet(
                positions=jnp.ones((2, 3)),
                sectors=['a', 'b', 'c'],
                instruments=['x', 'y', 'z'],
            )

    def test_properties(self):
        bs = simple_balance_sheet()
        assert bs.n_sectors == 3
        assert bs.n_instruments == 2


# ---------------------------------------------------------------------------
# GodleyTable tests
# ---------------------------------------------------------------------------

class TestGodleyTable:
    def test_consistent_flows(self):
        flows = jnp.array([
            [ 10.0,   0.0],   # households: receive deposits
            [  0.0, -10.0],   # firms: take more loans
            [-10.0,  10.0],   # banks: issue deposits, make loans
        ])
        gt = GodleyTable(flows=flows, sectors=['h', 'f', 'b'], instruments=['d', 'l'])
        assert gt.is_consistent()

    def test_apply_advances_balance_sheet(self):
        bs = simple_balance_sheet()
        flows = jnp.array([[10.0, 0.0], [0.0, -10.0], [-10.0, 10.0]])
        gt = GodleyTable(flows=flows, sectors=bs.sectors, instruments=bs.instruments)
        bs2 = gt.apply(bs)
        assert jnp.allclose(bs2.positions[0, 0], 110.0, atol=1e-6)
        assert bs2.is_consistent()


# ---------------------------------------------------------------------------
# PacioliManifold tests
# ---------------------------------------------------------------------------

class TestPacioliManifold:
    def test_from_edges_shape(self):
        m = simple_manifold()
        assert m.incidence.shape == (3, 3)

    def test_column_sums_zero(self):
        """∂²=0: each edge column sums to zero."""
        m = simple_manifold()
        col_sums = m.incidence.sum(axis=0)
        assert jnp.allclose(col_sums, jnp.zeros(3), atol=1e-6)

    def test_invalid_incidence_raises(self):
        """Incidence matrix with non-zero column sum should raise."""
        B = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # col 0 sums to 1, not 0
        with pytest.raises(ValueError, match="∂²=0 violated"):
            PacioliManifold(incidence=B, nodes=['a', 'b', 'c'], edges=['e1', 'e2'])

    def test_boundary_of_consistent_flows(self):
        m = simple_manifold()
        # A cycle: 1 unit around wages → deposits → loans
        flows = jnp.array([1.0, 1.0, 1.0])
        b = m.boundary(flows)
        assert jnp.allclose(b, jnp.zeros(3), atol=1e-6)

    def test_is_consistent_true(self):
        m = simple_manifold()
        flows = jnp.array([1.0, 1.0, 1.0])
        assert m.is_consistent(flows)

    def test_is_consistent_false(self):
        m = simple_manifold()
        flows = jnp.array([1.0, 0.5, 1.0])   # unbalanced
        assert not m.is_consistent(flows)

    def test_repr(self):
        m = simple_manifold()
        r = repr(m)
        assert 'PacioliManifold' in r
        assert 'H₀' in r

    def test_properties(self):
        m = simple_manifold()
        assert m.n_nodes == 3
        assert m.n_edges == 3


# ---------------------------------------------------------------------------
# Homology tests
# ---------------------------------------------------------------------------

class TestHomology:
    def test_connected_graph_h0_is_1(self):
        """A connected graph has H₀ = 1 (one component)."""
        m = simple_manifold()
        h = m.homology()
        assert h.H0_rank == 1

    def test_cycle_graph_h1_is_1(self):
        """A single cycle has H₁ = 1."""
        m = simple_manifold()   # wages → deposits → loans forms one cycle
        h = m.homology()
        assert h.H1_rank == 1

    def test_tree_has_no_cycles(self):
        """A tree (no cycles) has H₁ = 0."""
        m = PacioliManifold.from_edges(
            nodes=['a', 'b', 'c'],
            edges=[('e1', 'a', 'b'), ('e2', 'a', 'c')],
        )
        h = m.homology()
        assert h.H1_rank == 0

    def test_disconnected_graph_h0_gt_1(self):
        """Two isolated edges have H₀ = 2."""
        m = PacioliManifold.from_edges(
            nodes=['a', 'b', 'c', 'd'],
            edges=[('e1', 'a', 'b'), ('e2', 'c', 'd')],
        )
        h = m.homology()
        assert h.H0_rank == 2

    def test_laplacian_shape(self):
        m = simple_manifold()
        L = m.laplacian()
        assert L.shape == (3, 3)

    def test_laplacian_zero_eigenvalue(self):
        """Connected graph: Laplacian has exactly one zero eigenvalue."""
        m = simple_manifold()
        L = np.array(m.laplacian())
        eigenvalues = np.linalg.eigvalsh(L)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-6)
        assert n_zero == 1


# ---------------------------------------------------------------------------
# Three-sector SFC convenience constructor
# ---------------------------------------------------------------------------

class TestThreeSectorSFC:
    def test_balance_sheet_consistent(self):
        bs, manifold = three_sector_sfc()
        assert bs.is_consistent()

    def test_manifold_connected(self):
        bs, manifold = three_sector_sfc()
        h = manifold.homology()
        assert h.H0_rank == 1

    def test_manifold_has_cycle(self):
        bs, manifold = three_sector_sfc()
        h = manifold.homology()
        assert h.H1_rank == 1

    def test_sectors(self):
        bs, _ = three_sector_sfc()
        assert 'households' in bs.sectors
        assert 'banks' in bs.sectors

    def test_custom_values(self):
        bs, _ = three_sector_sfc(household_deposits=200.0, firm_loans=150.0)
        assert bs.is_consistent()
        # households hold 200 in deposits
        dep_idx = bs.instruments.index('deposits')
        hh_idx = bs.sectors.index('households')
        assert jnp.allclose(bs.positions[hh_idx, dep_idx], 200.0)
