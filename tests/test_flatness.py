"""
Tests for the three flatness-relaxation mechanisms:
  1. ResidualFlow / float sector utilities  (econiac.core.manifold)
  2. CurvedBalanceSheet + holonomy          (econiac.core.manifold)
  3. conservation_loss                      (econiac.pcl.combinators)
"""

import pytest
import jax.numpy as jnp

from econiac.core.manifold import (
    BalanceSheet,
    CurvedBalanceSheet,
    holonomy,
    add_residual_sector,
    add_float_sector,
    residual_magnitude,
    RESIDUAL_SECTOR,
    FLOAT_SECTOR,
)
from econiac.pcl.combinators import (
    flow,
    sequence,
    identity,
    conservation_loss,
    typecheck,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_bs(positions=None):
    """3-sector × 2-instrument balanced balance sheet."""
    if positions is None:
        positions = jnp.array([
            [ 100.0,   0.0],
            [   0.0, -80.0],
            [-100.0,  80.0],
        ])
    return BalanceSheet(
        positions=positions,
        sectors=['households', 'firms', 'banks'],
        instruments=['deposits', 'loans'],
    )


def _unbalanced_bs():
    """3-sector balance sheet that does NOT balance (col sums ≠ 0)."""
    positions = jnp.array([
        [100.0,  0.0],
        [  0.0, -80.0],
        [-95.0,  78.0],   # deliberate 5 and 2 unit discrepancies
    ])
    return BalanceSheet(
        positions=positions,
        sectors=['households', 'firms', 'banks'],
        instruments=['deposits', 'loans'],
    )


# ---------------------------------------------------------------------------
# Mechanism 1a: add_residual_sector
# ---------------------------------------------------------------------------

class TestAddResidualSector:
    def test_adds_residual_sector(self):
        bs = add_residual_sector(_unbalanced_bs())
        assert RESIDUAL_SECTOR in bs.sectors

    def test_result_is_consistent(self):
        bs = add_residual_sector(_unbalanced_bs())
        assert bs.is_consistent(atol=1e-5)

    def test_flat_input_gives_zero_residual(self):
        bs = add_residual_sector(_flat_bs())
        assert bs.is_consistent()
        idx = bs.sectors.index(RESIDUAL_SECTOR)
        assert jnp.allclose(bs.positions[idx], jnp.zeros(2), atol=1e-6)

    def test_residual_absorbs_exact_imbalance(self):
        raw = _unbalanced_bs()
        imbalance = raw.column_sums()        # e.g. [5.0, -2.0]
        bs = add_residual_sector(raw)
        idx = bs.sectors.index(RESIDUAL_SECTOR)
        assert jnp.allclose(bs.positions[idx], -imbalance, atol=1e-6)

    def test_n_sectors_increases_by_one(self):
        raw = _unbalanced_bs()
        bs  = add_residual_sector(raw)
        assert bs.n_sectors == raw.n_sectors + 1

    def test_original_positions_unchanged(self):
        raw = _unbalanced_bs()
        bs  = add_residual_sector(raw)
        assert jnp.allclose(bs.positions[:raw.n_sectors], raw.positions, atol=1e-9)


class TestResidualMagnitude:
    def test_zero_for_flat_input(self):
        bs = add_residual_sector(_flat_bs())
        assert residual_magnitude(bs) == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_for_unbalanced(self):
        bs = add_residual_sector(_unbalanced_bs())
        assert residual_magnitude(bs) > 0.0

    def test_zero_if_no_residual_sector(self):
        assert residual_magnitude(_flat_bs()) == 0.0

    def test_magnitude_equals_imbalance_norm(self):
        raw = _unbalanced_bs()
        imbalance = raw.column_sums()
        bs  = add_residual_sector(raw)
        expected = float(jnp.linalg.norm(imbalance))
        assert residual_magnitude(bs) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Mechanism 1b: add_float_sector
# ---------------------------------------------------------------------------

class TestAddFloatSector:
    def test_adds_float_sector(self):
        bs = add_float_sector(_flat_bs())
        assert FLOAT_SECTOR in bs.sectors

    def test_float_initialised_to_zero(self):
        bs  = add_float_sector(_flat_bs())
        idx = bs.sectors.index(FLOAT_SECTOR)
        assert jnp.allclose(bs.positions[idx], jnp.zeros(2), atol=1e-9)

    def test_consistency_preserved_for_flat_input(self):
        bs = add_float_sector(_flat_bs())
        assert bs.is_consistent()

    def test_float_clears_after_sequence(self):
        """
        Send then receive should leave float at zero.
        """
        bs = add_float_sector(_flat_bs())
        send    = flow('households', FLOAT_SECTOR, 'deposits', 50.0)
        receive = flow(FLOAT_SECTOR, 'firms',      'deposits', 50.0)
        result  = sequence(send, receive)(bs)
        idx     = result.sectors.index(FLOAT_SECTOR)
        assert jnp.allclose(result.positions[idx], jnp.zeros(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Mechanism 2: CurvedBalanceSheet
# ---------------------------------------------------------------------------

class TestCurvedBalanceSheet:
    def _curved(self, F=None):
        positions = jnp.array([
            [ 100.0,   0.0],
            [   0.0, -80.0],
            [-97.0,   79.0],   # col sums = [3.0, -1.0]
        ])
        curvature = F if F is not None else jnp.array([3.0, -1.0])
        return CurvedBalanceSheet(
            positions=positions,
            sectors=['households', 'firms', 'banks'],
            instruments=['deposits', 'loans'],
            curvature=curvature,
        )

    def test_is_consistent_when_col_sums_equal_curvature(self):
        cbs = self._curved()
        assert cbs.is_consistent()

    def test_is_not_flat(self):
        cbs = self._curved()
        assert not cbs.is_flat()

    def test_zero_curvature_is_flat(self):
        positions = jnp.array([
            [ 100.0,   0.0],
            [   0.0, -80.0],
            [-100.0,  80.0],
        ])
        cbs = CurvedBalanceSheet(
            positions=positions,
            sectors=['households', 'firms', 'banks'],
            instruments=['deposits', 'loans'],
            curvature=jnp.zeros(2),
        )
        assert cbs.is_flat()
        assert cbs.is_consistent()

    def test_to_flat_gives_consistent_balance_sheet(self):
        cbs  = self._curved()
        flat = cbs.to_flat()
        assert isinstance(flat, BalanceSheet)
        assert flat.is_consistent()

    def test_to_flat_preserves_non_last_sectors(self):
        cbs  = self._curved()
        flat = cbs.to_flat()
        assert jnp.allclose(flat.positions[:-1], cbs.positions[:-1], atol=1e-9)

    def test_repr_mentions_curved(self):
        cbs = self._curved()
        assert 'curved' in repr(cbs)

    def test_repr_flat_when_zero_curvature(self):
        positions = jnp.array([
            [ 100.0,   0.0],
            [   0.0, -80.0],
            [-100.0,  80.0],
        ])
        cbs = CurvedBalanceSheet(
            positions=positions,
            sectors=['a', 'b', 'c'],
            instruments=['x', 'y'],
            curvature=jnp.zeros(2),
        )
        assert 'flat' in repr(cbs)

    def test_wrong_curvature_shape_raises(self):
        positions = jnp.array([[1., 0.], [0., -1.], [-1., 1.]])
        with pytest.raises(ValueError):
            CurvedBalanceSheet(
                positions=positions,
                sectors=['a', 'b', 'c'],
                instruments=['x', 'y'],
                curvature=jnp.array([1., 2., 3.]),   # wrong shape
            )

    def test_net_worth(self):
        cbs = self._curved()
        nw  = cbs.net_worth()
        assert nw.shape == (3,)


class TestHolonomy:
    def test_flat_manifold_zero_holonomy(self):
        """Identity computation on a flat manifold: holonomy = 0."""
        cbs = CurvedBalanceSheet(
            positions=jnp.array([[100., 0.], [0., -80.], [-100., 80.]]),
            sectors=['a', 'b', 'c'],
            instruments=['x', 'y'],
            curvature=jnp.zeros(2),
        )
        h = holonomy(identity(), cbs)
        assert jnp.allclose(h, jnp.zeros(2), atol=1e-5)

    def test_nonzero_curvature_nonzero_holonomy(self):
        """Curved manifold with identity computation: holonomy equals curvature."""
        F = jnp.array([3.0, -1.0])
        positions = jnp.array([[100., 0.], [0., -80.], [-97., 79.]])
        cbs = CurvedBalanceSheet(
            positions=positions,
            sectors=['a', 'b', 'c'],
            instruments=['x', 'y'],
            curvature=F,
        )
        h = holonomy(identity(), cbs)
        assert jnp.allclose(h, F, atol=1e-5)

    def test_holonomy_shape(self):
        cbs = CurvedBalanceSheet(
            positions=jnp.array([[100., 0.], [0., -80.], [-100., 80.]]),
            sectors=['a', 'b', 'c'],
            instruments=['x', 'y'],
            curvature=jnp.zeros(2),
        )
        h = holonomy(identity(), cbs)
        assert h.shape == (2,)


# ---------------------------------------------------------------------------
# Mechanism 3: conservation_loss
# ---------------------------------------------------------------------------

class TestConservationLoss:
    def test_zero_for_conserving_computation(self):
        comp = flow('households', 'firms', 'deposits', 50.0)
        bs   = _flat_bs()
        loss = conservation_loss(comp, bs, sigma=1.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-8)

    def test_identity_has_zero_loss(self):
        loss = conservation_loss(identity(), _flat_bs(), sigma=1.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-8)

    def test_loss_positive_for_nonconserving(self):
        """Manually inject a non-conserving computation."""
        from econiac.pcl.combinators import Computation
        def bad_fn(bs):
            return BalanceSheet(
                positions=bs.positions.at[0, 0].add(10.0),  # debit without credit
                sectors=bs.sectors,
                instruments=bs.instruments,
            )
        bad  = Computation(name="bad", fn=bad_fn)
        loss = conservation_loss(bad, _flat_bs(), sigma=1.0)
        assert float(loss) > 0.0

    def test_larger_sigma_gives_smaller_loss(self):
        from econiac.pcl.combinators import Computation
        def bad_fn(bs):
            return BalanceSheet(
                positions=bs.positions.at[0, 0].add(10.0),
                sectors=bs.sectors,
                instruments=bs.instruments,
            )
        bad    = Computation(name="bad", fn=bad_fn)
        bs     = _flat_bs()
        loss1  = float(conservation_loss(bad, bs, sigma=1.0))
        loss10 = float(conservation_loss(bad, bs, sigma=10.0))
        assert loss1 > loss10

    def test_loss_scales_with_sigma_squared(self):
        from econiac.pcl.combinators import Computation
        def bad_fn(bs):
            return BalanceSheet(
                positions=bs.positions.at[0, 0].add(10.0),
                sectors=bs.sectors,
                instruments=bs.instruments,
            )
        bad   = Computation(name="bad", fn=bad_fn)
        bs    = _flat_bs()
        l1    = float(conservation_loss(bad, bs, sigma=1.0))
        l2    = float(conservation_loss(bad, bs, sigma=2.0))
        assert l1 / l2 == pytest.approx(4.0, rel=1e-5)

    def test_loss_is_jax_array(self):
        import jax
        loss = conservation_loss(identity(), _flat_bs(), sigma=1.0)
        assert isinstance(loss, jax.Array)

    def test_default_sigma_is_one(self):
        loss_default  = conservation_loss(identity(), _flat_bs())
        loss_explicit = conservation_loss(identity(), _flat_bs(), sigma=1.0)
        assert float(loss_default) == pytest.approx(float(loss_explicit), abs=1e-9)
