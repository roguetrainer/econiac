"""Tests for econiac.finance.credit — survival probabilities, CVA, XVA."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.finance.curves import YieldCurve
from econiac.finance.credit import (
    HazardRateConnection,
    cva,
    CVAResult,
    xva,
    XVAResult,
)


MATURITIES = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])


def flat_hazard(rate=0.02, name="cpty"):
    return HazardRateConnection.flat(rate, MATURITIES, name=name)


def flat_ir(rate=0.05):
    return YieldCurve.flat(rate, MATURITIES)


class TestHazardRateConnection:
    def test_flat_hazard_shape(self):
        h = flat_hazard()
        assert h.hazard_rates.shape == (5,)

    def test_survival_at_zero_is_one(self):
        """Q(τ > 0) = 1."""
        h = flat_hazard()
        sp = float(h.survival_probability(0.0))
        assert abs(sp - 1.0) < 1e-6

    def test_survival_decreasing(self):
        h = flat_hazard()
        sps = [float(h.survival_probability(T)) for T in [0.5, 1.0, 2.0, 5.0]]
        assert all(sps[i] > sps[i + 1] for i in range(len(sps) - 1))

    def test_flat_hazard_survival_formula(self):
        """Q(τ>T) = exp(-h·T) for flat hazard h."""
        h_rate = 0.02
        h = flat_hazard(h_rate)
        T = 2.0
        expected = np.exp(-h_rate * T)
        actual = float(h.survival_probability(T))
        assert abs(actual - expected) < 1e-3

    def test_negative_hazard_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            HazardRateConnection(
                maturities=MATURITIES,
                hazard_rates=jnp.array([-0.01, 0.02, 0.02, 0.02, 0.02]),
                name="bad",
            )

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            HazardRateConnection(
                maturities=MATURITIES,
                hazard_rates=jnp.array([0.02, 0.02]),
                name="bad",
            )

    def test_conditional_survival_composition(self):
        """Q(τ>T1) * Q(τ>T2|τ>T1) = Q(τ>T2)."""
        h = flat_hazard()
        T1, T2 = 1.0, 5.0
        sp_T2 = float(h.survival_probability(T2))
        sp_T1_cond = float(h.survival_probability(T1)) * float(h.conditional_survival(T1, T2))
        assert abs(sp_T2 - sp_T1_cond) < 1e-5

    def test_temporal_curvature_zero(self):
        """Self-consistent hazard curve has zero temporal curvature."""
        h = flat_hazard()
        F = float(h.temporal_curvature(1.0, 5.0))
        assert abs(F) < 1e-5

    def test_default_density_positive(self):
        h = flat_hazard()
        f = float(h.default_density(2.0))
        assert f > 0

    def test_from_survival_probabilities_roundtrip(self):
        """Construct from survival probs → hazard rates → survival probs."""
        h_rate = 0.03
        sps = jnp.exp(-h_rate * MATURITIES)
        h = HazardRateConnection.from_survival_probabilities(MATURITIES, sps)
        # Check survival at 1y
        sp_1 = float(h.survival_probability(1.0))
        assert abs(sp_1 - np.exp(-h_rate * 1.0)) < 0.01

    def test_repr(self):
        r = repr(flat_hazard())
        assert 'HazardRateConnection' in r
        assert 'cpty' in r

    def test_name_stored(self):
        h = flat_hazard(name="Acme Corp")
        assert h.name == "Acme Corp"


class TestCVA:
    def test_cva_positive(self):
        """CVA is always non-negative."""
        exposure = jnp.ones(5) * 1_000_000
        result = cva(exposure, flat_hazard(), flat_ir())
        assert result.cva >= 0

    def test_cva_zero_for_zero_exposure(self):
        exposure = jnp.zeros(5)
        result = cva(exposure, flat_hazard(), flat_ir())
        assert abs(result.cva) < 1e-6

    def test_cva_zero_for_zero_hazard(self):
        exposure = jnp.ones(5) * 1_000_000
        result = cva(exposure, flat_hazard(0.0), flat_ir())
        assert abs(result.cva) < 1e-3

    def test_cva_increases_with_hazard(self):
        exposure = jnp.ones(5) * 1_000_000
        cva_low  = cva(exposure, flat_hazard(0.01), flat_ir()).cva
        cva_high = cva(exposure, flat_hazard(0.05), flat_ir()).cva
        assert cva_high > cva_low

    def test_cva_increases_with_lgd(self):
        exposure = jnp.ones(5) * 1_000_000
        cva_low  = cva(exposure, flat_hazard(), flat_ir(), lgd=0.4).cva
        cva_high = cva(exposure, flat_hazard(), flat_ir(), lgd=0.8).cva
        assert cva_high > cva_low

    def test_cva_result_is_namedtuple(self):
        exposure = jnp.ones(5) * 1_000_000
        result = cva(exposure, flat_hazard(), flat_ir())
        assert isinstance(result, CVAResult)
        assert hasattr(result, 'cva')
        assert hasattr(result, 'default_density')

    def test_cva_profiles_shape(self):
        exposure = jnp.ones(5) * 1_000_000
        result = cva(exposure, flat_hazard(), flat_ir())
        assert result.default_density.shape == (5,)
        assert result.discount_factors.shape == (5,)

    def test_cva_wrong_exposure_length_raises(self):
        exposure = jnp.ones(3) * 1_000_000   # wrong: grid has 5 points
        with pytest.raises(ValueError, match="length"):
            cva(exposure, flat_hazard(), flat_ir())

    def test_cva_custom_grid(self):
        grid = jnp.array([0.5, 1.0, 2.0, 5.0])
        exposure = jnp.ones(4) * 1_000_000
        result = cva(exposure, flat_hazard(), flat_ir(), grid=grid)
        assert result.cva >= 0


class TestXVA:
    def test_xva_returns_namedtuple(self):
        exposure = jnp.ones(5) * 1_000_000
        result = xva(exposure, flat_hazard(), flat_hazard(0.01, name="own"), flat_ir())
        assert isinstance(result, XVAResult)

    def test_cva_component_positive(self):
        exposure = jnp.ones(5) * 1_000_000
        result = xva(exposure, flat_hazard(), flat_hazard(0.01, name="own"), flat_ir())
        assert result.cva >= 0

    def test_dva_component_negative_or_zero(self):
        """DVA is a benefit → should be ≤ 0."""
        exposure = jnp.ones(5) * 1_000_000
        result = xva(exposure, flat_hazard(), flat_hazard(0.01, name="own"), flat_ir())
        assert result.dva <= 0

    def test_fva_zero_for_zero_spread(self):
        exposure = jnp.ones(5) * 1_000_000
        result = xva(exposure, flat_hazard(), flat_hazard(0.01, name="own"),
                     flat_ir(), funding_spread=0.0)
        assert abs(result.fva) < 1e-6

    def test_fva_positive_for_positive_spread(self):
        exposure = jnp.ones(5) * 1_000_000
        result = xva(exposure, flat_hazard(), flat_hazard(0.01, name="own"),
                     flat_ir(), funding_spread=0.005)
        assert result.fva > 0

    def test_total_is_sum(self):
        exposure = jnp.ones(5) * 1_000_000
        result = xva(exposure, flat_hazard(), flat_hazard(0.01, name="own"),
                     flat_ir(), funding_spread=0.002)
        assert abs(result.total - (result.cva + result.dva + result.fva)) < 1e-6
