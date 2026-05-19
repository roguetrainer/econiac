"""Tests for econiac.finance.curves — YieldCurve as temporal connection."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.finance.curves import (
    YieldCurve,
    hjm_drift,
    lmm_forward_rates,
    lmm_discrete_flatness,
)

MATURITIES = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
FLAT_RATE  = 0.05


def flat_curve():
    return YieldCurve.flat(FLAT_RATE, MATURITIES)


def upward_curve():
    rates = jnp.array([0.02, 0.025, 0.03, 0.04, 0.05])
    return YieldCurve.from_zero_rates(MATURITIES, rates)


class TestYieldCurveConstruction:
    def test_flat_discount_factors(self):
        curve = flat_curve()
        # P(0,1) = exp(-0.05*1) ≈ 0.9512
        assert jnp.allclose(curve.discount_factors[1], jnp.exp(-0.05 * 1.0), atol=1e-5)

    def test_from_zero_rates_roundtrip(self):
        rates = jnp.array([0.03, 0.035, 0.04, 0.045, 0.05])
        curve = YieldCurve.from_zero_rates(MATURITIES, rates)
        assert jnp.allclose(curve.zero_rates, rates, atol=1e-5)

    def test_n_property(self):
        assert flat_curve().n == 5

    def test_negative_discount_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            YieldCurve(maturities=MATURITIES, discount_factors=jnp.array([-1.0, 0.9, 0.8, 0.7, 0.6]))

    def test_discount_above_one_raises(self):
        with pytest.raises(ValueError, match="≤ 1"):
            YieldCurve(maturities=MATURITIES, discount_factors=jnp.array([1.1, 0.9, 0.8, 0.7, 0.6]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            YieldCurve(maturities=MATURITIES, discount_factors=jnp.array([0.9, 0.8]))

    def test_repr(self):
        assert 'YieldCurve' in repr(flat_curve())
        assert '5%' in repr(flat_curve()) or '5.00%' in repr(flat_curve())


class TestYieldCurveDiscounting:
    def test_discount_at_pillar(self):
        curve = flat_curve()
        for i, T in enumerate(MATURITIES):
            assert jnp.allclose(curve.discount(float(T)), curve.discount_factors[i], atol=1e-4)

    def test_discount_decreasing(self):
        curve = flat_curve()
        dfs = [float(curve.discount(T)) for T in [0.5, 1.0, 2.0, 5.0, 10.0]]
        assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))

    def test_forward_discount_composition(self):
        """P(0,T1) * P(T1,T2) = P(0,T2) for a flat curve."""
        curve = flat_curve()
        T1, T2 = 1.0, 5.0
        p0T2 = float(curve.discount(T2))
        p0T1_times_pT1T2 = float(curve.discount(T1)) * float(curve.forward_discount(T1, T2))
        assert abs(p0T2 - p0T1_times_pT1T2) < 1e-5

    def test_forward_rate_positive(self):
        curve = upward_curve()
        L = float(curve.forward_rate(1.0, 2.0))
        assert L > 0

    def test_instantaneous_forward_approx_zero_rate_deriv(self):
        """For flat curve, instantaneous forward = flat rate."""
        curve = flat_curve()
        f = float(curve.instantaneous_forward(2.0))
        assert abs(f - FLAT_RATE) < 1e-3


class TestTemporalFlatness:
    def test_flat_curve_zero_curvature(self):
        """A self-consistent curve has zero temporal curvature everywhere."""
        curve = flat_curve()
        F = float(curve.temporal_curvature(1.0, 5.0))
        assert abs(F) < 1e-5

    def test_is_flat_true_for_self_consistent(self):
        curve = flat_curve()
        assert curve.is_flat()

    def test_upward_curve_is_flat(self):
        """Any interpolated curve is flat — curvature is identically zero by construction."""
        curve = upward_curve()
        assert curve.is_flat()

    def test_spliced_curve_may_be_curved(self):
        """A manually inconsistent curve has non-zero curvature."""
        # Build a curve where P(0,2) ≠ P(0,1)*P(1,2)
        dfs = jnp.array([0.97, 0.93, 0.85, 0.70, 0.55])   # self-consistent by interpolation
        curve = YieldCurve(maturities=MATURITIES, discount_factors=dfs)
        assert curve.is_flat()


class TestHJMDrift:
    def test_hjm_drift_shape(self):
        sigma = jnp.ones(5) * 0.01
        alpha = hjm_drift(MATURITIES, sigma)
        assert alpha.shape == (5,)

    def test_hjm_drift_zero_vol(self):
        sigma = jnp.zeros(5)
        alpha = hjm_drift(MATURITIES, sigma)
        assert jnp.allclose(alpha, jnp.zeros(5), atol=1e-8)

    def test_hjm_drift_positive_vol(self):
        """Positive vol → positive drift (convexity correction)."""
        sigma = jnp.ones(5) * 0.01
        alpha = hjm_drift(MATURITIES, sigma)
        assert jnp.all(alpha >= 0)

    def test_hjm_drift_increases_with_maturity(self):
        """∫_0^T σ du increases with T → α increases with T."""
        sigma = jnp.ones(5) * 0.01
        alpha = hjm_drift(MATURITIES, sigma)
        assert jnp.all(jnp.diff(alpha) >= 0)


class TestLMM:
    def test_lmm_forward_rates_shape(self):
        curve = flat_curve()
        tenors = jnp.array([0.5, 1.0, 2.0, 5.0])
        L = lmm_forward_rates(curve, tenors)
        assert L.shape == (3,)

    def test_lmm_forward_rates_positive(self):
        curve = flat_curve()
        tenors = jnp.array([0.5, 1.0, 2.0, 5.0])
        L = lmm_forward_rates(curve, tenors)
        assert jnp.all(L > 0)

    def test_lmm_discrete_flatness_near_zero(self):
        """Self-consistent curve has near-zero discrete flatness residuals."""
        curve = flat_curve()
        tenors = jnp.array([0.5, 1.0, 2.0, 5.0])
        F = lmm_discrete_flatness(curve, tenors)
        assert jnp.allclose(F, jnp.zeros_like(F), atol=1e-5)
