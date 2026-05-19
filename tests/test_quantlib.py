"""Tests for econiac.finance.quantlib — adapters and vanilla bootstrapping."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.finance.quantlib import (
    curve_from_swap_rates,
    ql_cds_to_credit_connection,
    ql_curve_to_yield_curve,
    ql_hazard_to_credit_connection,
)
from econiac.finance.curves import YieldCurve
from econiac.finance.credit import HazardRateConnection


# ---------------------------------------------------------------------------
# QuantLib-dependent tests: skipped when QuantLib is not installed
# ---------------------------------------------------------------------------

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

quantlib_required = pytest.mark.skipif(
    not HAS_QUANTLIB,
    reason="QuantLib not installed (pip install QuantLib)"
)


@quantlib_required
class TestQLCurveAdapter:
    def setup_method(self):
        today = ql.Date(1, 1, 2026)
        ql.Settings.instance().evaluationDate = today
        self.ql_flat = ql.FlatForward(today, 0.05, ql.Actual365Fixed())
        self.today = today

    def test_discount_factors_positive(self):
        curve = ql_curve_to_yield_curve(self.ql_flat, [0.5, 1.0, 2.0, 5.0])
        assert jnp.all(curve.discount_factors > 0)

    def test_discount_factors_in_range(self):
        curve = ql_curve_to_yield_curve(self.ql_flat, [0.5, 1.0, 2.0, 5.0])
        assert jnp.all(curve.discount_factors <= 1.0)

    def test_returns_yield_curve(self):
        curve = ql_curve_to_yield_curve(self.ql_flat, [1.0, 5.0, 10.0])
        assert isinstance(curve, YieldCurve)

    def test_flat_rate_approx(self):
        """Flat 5% curve → P(0,1) ≈ exp(-0.05)."""
        curve = ql_curve_to_yield_curve(self.ql_flat, [1.0, 2.0, 5.0])
        assert abs(float(curve.discount_factors[0]) - np.exp(-0.05)) < 0.005


@quantlib_required
class TestQLHazardAdapter:
    def setup_method(self):
        ql = pytest.importorskip("QuantLib")
        today = ql.Date(1, 1, 2026)
        ql.Settings.instance().evaluationDate = today
        # Flat hazard curve at 2%
        dates = [today + ql.Period(int(T * 365), ql.Days) for T in [0.5, 1, 2, 5]]
        sps   = [np.exp(-0.02 * T) for T in [0.5, 1, 2, 5]]
        self.ql_hazard = ql.SurvivalProbabilityCurve(dates, sps, ql.Actual365Fixed(), ql.NullCalendar())
        self.ql_hazard.enableExtrapolation()

    def test_returns_hazard_connection(self):
        h = ql_hazard_to_credit_connection(self.ql_hazard, [1.0, 2.0, 5.0])
        assert isinstance(h, HazardRateConnection)

    def test_hazard_rates_non_negative(self):
        h = ql_hazard_to_credit_connection(self.ql_hazard, [1.0, 2.0, 5.0])
        assert jnp.all(h.hazard_rates >= 0)


# ---------------------------------------------------------------------------
# CDS bootstrap (no QuantLib needed)
# ---------------------------------------------------------------------------

class TestCDSBootstrap:
    def test_from_cds_spreads_shape(self):
        spreads = {1.0: 50.0, 2.0: 80.0, 5.0: 120.0}   # bps
        h = ql_cds_to_credit_connection(spreads)
        assert h.hazard_rates.shape == (3,)

    def test_from_cds_spreads_positive(self):
        spreads = {1.0: 50.0, 5.0: 100.0}
        h = ql_cds_to_credit_connection(spreads)
        assert jnp.all(h.hazard_rates > 0)

    def test_higher_spread_higher_hazard(self):
        h_low  = ql_cds_to_credit_connection({1.0: 50.0})
        h_high = ql_cds_to_credit_connection({1.0: 200.0})
        assert float(h_high.hazard_rates[0]) > float(h_low.hazard_rates[0])

    def test_name_assigned(self):
        spreads = {1.0: 50.0}
        h = ql_cds_to_credit_connection(spreads, name="Acme")
        assert h.name == "Acme"

    def test_recovery_rate_scales_hazard(self):
        """Higher recovery → higher hazard for same spread (h ≈ s/LGD)."""
        spreads = {1.0: 100.0}
        h_low_rec  = ql_cds_to_credit_connection(spreads, recovery_rate=0.2)  # LGD=0.8
        h_high_rec = ql_cds_to_credit_connection(spreads, recovery_rate=0.6)  # LGD=0.4
        assert float(h_high_rec.hazard_rates[0]) > float(h_low_rec.hazard_rates[0])


# ---------------------------------------------------------------------------
# Swap-rate bootstrap (no QuantLib needed)
# ---------------------------------------------------------------------------

class TestSwapRateBootstrap:
    def test_returns_yield_curve(self):
        curve = curve_from_swap_rates([1, 2, 5], [0.04, 0.045, 0.05])
        assert isinstance(curve, YieldCurve)

    def test_discount_factors_positive(self):
        curve = curve_from_swap_rates([1, 2, 5], [0.04, 0.045, 0.05])
        assert jnp.all(curve.discount_factors > 0)

    def test_discount_factors_decreasing(self):
        curve = curve_from_swap_rates([1, 2, 5, 10], [0.04, 0.045, 0.05, 0.055])
        dfs = np.array(curve.discount_factors)
        assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))

    def test_flat_swap_curve_consistent(self):
        """Flat swap rates → all discount factors follow flat zero rate."""
        flat_rate = 0.05
        tenors = [1, 2, 5]
        curve = curve_from_swap_rates(tenors, [flat_rate] * len(tenors))
        # P(0,1) from swap bootstrap should be close to exp(-0.05)
        assert abs(float(curve.discount_factors[0]) - np.exp(-flat_rate * 0.5)) < 0.02

    def test_n_tenors(self):
        curve = curve_from_swap_rates([1, 2, 3, 5, 7, 10], [0.04]*6)
        assert curve.n == 6
