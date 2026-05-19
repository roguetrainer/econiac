"""Tests for econiac.finance.fx — FX as connection curvature, CIP."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.finance.curves import YieldCurve
from econiac.finance.fx import (
    FXMarket,
    cip_residual,
    swap_line_holonomy,
)


MATURITIES = jnp.array([0.5, 1.0, 2.0, 5.0])


def flat_curve(rate):
    return YieldCurve.flat(rate, MATURITIES)


def consistent_market():
    """USD/EUR/GBP market with consistent spot rates (no triangular arbitrage)."""
    # S_USD/EUR = 0.91, S_EUR/GBP = 0.87, S_USD/GBP = 0.91 * 0.87 = 0.7917
    spot = jnp.array([
        [1.0,   0.91,  0.91 * 0.87],
        [1/0.91, 1.0,  0.87        ],
        [1/(0.91*0.87), 1/0.87, 1.0],
    ])
    return FXMarket(
        currencies=['USD', 'EUR', 'GBP'],
        spot_rates=spot,
        ir_curves=[flat_curve(0.05), flat_curve(0.03), flat_curve(0.04)],
    )


def inconsistent_market():
    """Inconsistent spot rates → non-zero triangular arbitrage."""
    spot = jnp.array([
        [1.0,  0.91, 0.80],    # USD→GBP should be ~0.79, not 0.80
        [1.10, 1.0,  0.87],
        [1.25, 1.15, 1.0 ],
    ])
    return FXMarket(
        currencies=['USD', 'EUR', 'GBP'],
        spot_rates=spot,
        ir_curves=[flat_curve(0.05), flat_curve(0.03), flat_curve(0.04)],
    )


class TestFXMarket:
    def test_n(self):
        mkt = consistent_market()
        assert mkt.n == 3

    def test_consistent_market_arbitrage_free(self):
        mkt = consistent_market()
        assert mkt.is_arbitrage_free(atol=1e-4)

    def test_inconsistent_market_has_arbitrage(self):
        mkt = inconsistent_market()
        assert not mkt.is_arbitrage_free()

    def test_arbitrage_surface_shape(self):
        mkt = consistent_market()
        F = mkt.arbitrage_surface()
        assert F.shape == (3, 3, 3)

    def test_arbitrage_surface_near_zero_for_consistent(self):
        mkt = consistent_market()
        F = mkt.arbitrage_surface()
        assert jnp.max(jnp.abs(F)) < 1e-4

    def test_max_arbitrage_zero_for_consistent(self):
        mkt = consistent_market()
        assert float(mkt.max_arbitrage()) < 1e-4

    def test_max_arbitrage_positive_for_inconsistent(self):
        mkt = inconsistent_market()
        assert float(mkt.max_arbitrage()) > 0.01

    def test_triangular_arbitrage_consistent(self):
        mkt = consistent_market()
        lh = float(mkt.triangular_arbitrage(0, 1, 2))
        assert abs(lh) < 1e-4

    def test_triangular_arbitrage_inconsistent(self):
        mkt = inconsistent_market()
        lh = float(mkt.triangular_arbitrage(0, 1, 2))
        assert abs(lh) > 0.01

    def test_forward_rate_cip(self):
        """CIP forward rate: F = S * P_foreign / P_domestic."""
        mkt = consistent_market()
        T = 1.0
        F_01 = float(mkt.forward_rate(0, 1, T))   # USD→EUR 1y forward
        S_01  = float(mkt.spot_rates[0, 1])
        P_usd = float(mkt.ir_curves[0].discount(T))
        P_eur = float(mkt.ir_curves[1].discount(T))
        expected = S_01 * P_eur / P_usd
        assert abs(F_01 - expected) < 1e-6

    def test_cip_deviation_no_market_forward(self):
        """cip_deviation returns 0 when no market_forward is given."""
        mkt = consistent_market()
        phi = float(mkt.cip_deviation(0, 1, T=1.0))
        assert abs(phi) < 1e-8

    def test_cip_deviation_with_market_forward(self):
        """Market forward above CIP → positive deviation."""
        mkt = consistent_market()
        cip_fwd = float(mkt.forward_rate(0, 1, T=1.0))
        phi = float(mkt.cip_deviation(0, 1, T=1.0, market_forward=cip_fwd * 1.005))
        assert phi > 0

    def test_repr(self):
        mkt = consistent_market()
        r = repr(mkt)
        assert 'FXMarket' in r
        assert 'USD' in r

    def test_wrong_shape_spot_rates_raises(self):
        with pytest.raises(ValueError):
            FXMarket(
                currencies=['USD', 'EUR'],
                spot_rates=jnp.ones((3, 3)),
                ir_curves=[flat_curve(0.05), flat_curve(0.03)],
            )

    def test_wrong_n_curves_raises(self):
        with pytest.raises(ValueError):
            FXMarket(
                currencies=['USD', 'EUR', 'GBP'],
                spot_rates=jnp.ones((3, 3)),
                ir_curves=[flat_curve(0.05)],
            )


class TestCIPResidual:
    def test_cip_holds_when_forward_matches(self):
        """CIP residual is zero when forward = S * exp((r_f - r_d) * T)."""
        S, r_d, r_f, T = 1.10, 0.05, 0.03, 1.0
        F_cip = S * np.exp((r_f - r_d) * T)
        phi = float(cip_residual(S, r_d, r_f, F_cip, T))
        assert abs(phi) < 1e-6

    def test_cip_violation_positive(self):
        """Forward above CIP → positive deviation."""
        S, r_d, r_f, T = 1.10, 0.05, 0.03, 1.0
        F_market = 1.12   # above CIP
        phi = float(cip_residual(S, r_d, r_f, F_market, T))
        assert phi > 0

    def test_cip_violation_negative(self):
        S, r_d, r_f, T = 1.10, 0.05, 0.03, 1.0
        F_market = 1.07   # below CIP
        phi = float(cip_residual(S, r_d, r_f, F_market, T))
        assert phi < 0


class TestSwapLine:
    def test_symmetric_swap_holonomy_zero(self):
        """Symmetric swap line has zero holonomy (no arbitrage on new path)."""
        from econiac.core.connections import fx_connection
        spot = jnp.array([[1.0, 0.91, 0.79], [1.10, 1.0, 0.87], [1.27, 1.15, 1.0]])
        conn = fx_connection(['USD', 'EUR', 'GBP'], spot)
        h = float(swap_line_holonomy(conn, 0, 1, swap_rate=1.0))
        assert abs(h) < 1e-6
