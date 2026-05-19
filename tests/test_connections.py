"""Tests for econiac.core.connections — parallel transport, holonomy, curvature."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.core.connections import (
    Connection,
    parallel_transport,
    wilson_loop,
    log_holonomy,
    curvature,
    is_flat,
    max_curvature,
    curvature_matrix,
    gauge_transform,
    flat_gauge,
    fx_connection,
    discount_connection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def three_currency_flat():
    """Three consistent FX rates — flat connection, no triangular arbitrage.
    USD→EUR: 0.91, EUR→GBP: 0.87, USD→GBP = 0.91 * 0.87 = 0.7917."""
    S = jnp.array([
        [1.0,   0.91,  0.7917],
        [1/0.91, 1.0,  0.87  ],
        [1/0.7917, 1/0.87, 1.0],
    ])
    return Connection.from_rates(S, ['USD', 'EUR', 'GBP'])


def three_currency_curved():
    """Inconsistent FX rates — non-zero curvature (triangular arbitrage)."""
    S = jnp.array([
        [1.0,  0.91, 0.80],   # USD→GBP should be 0.91*0.87≈0.7917, not 0.80
        [1.10, 1.0,  0.87],
        [1.25, 1.15, 1.0 ],
    ])
    return Connection.from_rates(S, ['USD', 'EUR', 'GBP'])


# ---------------------------------------------------------------------------
# Connection construction
# ---------------------------------------------------------------------------

class TestConnectionConstruction:
    def test_from_rates_shape(self):
        conn = three_currency_flat()
        assert conn.log_rates.shape == (3, 3)

    def test_diagonal_zero(self):
        conn = three_currency_flat()
        assert jnp.allclose(jnp.diag(conn.log_rates), jnp.zeros(3), atol=1e-6)

    def test_rates_property(self):
        conn = three_currency_flat()
        assert jnp.allclose(conn.rates[0, 1], 0.91, atol=1e-4)

    def test_from_symmetric_antisymmetric(self):
        S = jnp.array([[1.0, 0.91, 0.79],
                       [1.10, 1.0, 0.87],
                       [1.27, 1.15, 1.0]])
        conn = Connection.from_symmetric(S, ['USD', 'EUR', 'GBP'])
        assert conn.is_antisymmetric()

    def test_n_nodes(self):
        conn = three_currency_flat()
        assert conn.n_nodes == 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            Connection(log_rates=jnp.ones((2, 3)), nodes=['a', 'b', 'c'])

    def test_repr_contains_nodes(self):
        conn = three_currency_flat()
        r = repr(conn)
        assert 'Connection' in r


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------

class TestParallelTransport:
    def test_single_edge(self):
        conn = three_currency_flat()
        # USD→EUR: rate ≈ 0.91
        t = parallel_transport(conn, [0, 1])
        assert jnp.allclose(t, 0.91, atol=1e-4)

    def test_two_edge_path(self):
        conn = three_currency_flat()
        # USD→EUR→GBP: 0.91 * 0.87 ≈ 0.7917
        t = parallel_transport(conn, [0, 1, 2])
        assert jnp.allclose(t, 0.91 * 0.87, atol=1e-3)

    def test_empty_path_returns_one(self):
        conn = three_currency_flat()
        t = parallel_transport(conn, [0])
        assert jnp.allclose(t, 1.0, atol=1e-6)

    def test_positive_result(self):
        conn = three_currency_curved()
        t = parallel_transport(conn, [0, 1, 2])
        assert t > 0


# ---------------------------------------------------------------------------
# Wilson loop / holonomy
# ---------------------------------------------------------------------------

class TestWilsonLoop:
    def test_flat_loop_holonomy_is_one(self):
        """Flat connection → holonomy = 1 around any triangle."""
        conn = three_currency_flat()
        hol = wilson_loop(conn, [0, 1, 2])
        assert jnp.allclose(hol, 1.0, atol=1e-4)

    def test_curved_loop_holonomy_not_one(self):
        """Curved connection → holonomy ≠ 1."""
        conn = three_currency_curved()
        hol = wilson_loop(conn, [0, 1, 2])
        assert not jnp.allclose(hol, 1.0, atol=1e-3)

    def test_log_holonomy_flat_is_zero(self):
        conn = three_currency_flat()
        lh = log_holonomy(conn, [0, 1, 2])
        assert jnp.allclose(lh, 0.0, atol=1e-4)

    def test_log_holonomy_and_wilson_consistent(self):
        conn = three_currency_curved()
        lh = log_holonomy(conn, [0, 1, 2])
        hol = wilson_loop(conn, [0, 1, 2])
        assert jnp.allclose(jnp.exp(lh), hol, atol=1e-6)

    def test_reversed_loop_is_reciprocal(self):
        """For an antisymmetric connection, going around a loop in reverse gives 1/holonomy."""
        S = jnp.array([[1.0, 0.91, 0.80], [1.10, 1.0, 0.87], [1.25, 1.15, 1.0]])
        conn = Connection.from_symmetric(S, ['USD', 'EUR', 'GBP'])
        hol_fwd = wilson_loop(conn, [0, 1, 2])
        hol_rev = wilson_loop(conn, [2, 1, 0])
        assert jnp.allclose(hol_fwd * hol_rev, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

class TestCurvature:
    def test_flat_connection_zero_curvature(self):
        conn = three_currency_flat()
        F = curvature(conn)
        assert jnp.allclose(F, jnp.zeros((3, 3, 3)), atol=1e-4)

    def test_curved_connection_nonzero_curvature(self):
        conn = three_currency_curved()
        F = curvature(conn)
        assert not jnp.allclose(F, jnp.zeros((3, 3, 3)), atol=1e-3)

    def test_curvature_shape(self):
        conn = three_currency_flat()
        F = curvature(conn)
        assert F.shape == (3, 3, 3)

    def test_is_flat_true(self):
        conn = three_currency_flat()
        assert is_flat(conn, atol=1e-4)

    def test_is_flat_false(self):
        conn = three_currency_curved()
        assert not is_flat(conn)

    def test_max_curvature_flat_near_zero(self):
        conn = three_currency_flat()
        assert max_curvature(conn) < 1e-4

    def test_max_curvature_curved_positive(self):
        conn = three_currency_curved()
        assert max_curvature(conn) > 0.01

    def test_curvature_matrix_shape(self):
        conn = three_currency_flat()
        C = curvature_matrix(conn)
        assert C.shape == (3, 3)

    def test_curvature_antisymmetry(self):
        """F[i,j,k] = -F[k,j,i] for antisymmetric connection."""
        conn = Connection.from_symmetric(
            jnp.array([[1.0, 0.91, 0.80], [1.10, 1.0, 0.87], [1.25, 1.15, 1.0]]),
            ['USD', 'EUR', 'GBP'],
        )
        F = curvature(conn)
        # F[0,1,2] + F[2,1,0] should be zero for antisymmetric log_rates
        assert jnp.allclose(F[0, 1, 2] + F[2, 1, 0], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Gauge transformations
# ---------------------------------------------------------------------------

class TestGaugeTransform:
    def test_gauge_transform_preserves_curvature(self):
        """Curvature is gauge-invariant."""
        conn = three_currency_curved()
        log_lam = jnp.array([0.1, -0.2, 0.05])
        conn2 = gauge_transform(conn, log_lam)
        F1 = curvature(conn)
        F2 = curvature(conn2)
        assert jnp.allclose(F1, F2, atol=1e-6)

    def test_trivial_gauge_unchanged(self):
        conn = three_currency_flat()
        conn2 = gauge_transform(conn, jnp.zeros(3))
        assert jnp.allclose(conn.log_rates, conn2.log_rates, atol=1e-6)

    def test_flat_gauge_returns_connection(self):
        conn = three_currency_curved()
        conn_flat = flat_gauge(conn)
        assert isinstance(conn_flat, Connection)

    def test_flat_gauge_preserves_curvature(self):
        conn = three_currency_curved()
        conn_flat = flat_gauge(conn)
        F1 = curvature(conn)
        F2 = curvature(conn_flat)
        assert jnp.allclose(F1, F2, atol=1e-6)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

class TestConvenienceConstructors:
    def test_fx_connection_antisymmetric(self):
        S = jnp.array([[1.0, 0.91, 0.79],
                       [1.10, 1.0, 0.87],
                       [1.27, 1.15, 1.0]])
        conn = fx_connection(['USD', 'EUR', 'GBP'], S)
        assert conn.is_antisymmetric()

    def test_discount_connection_shape(self):
        rates = jnp.array([0.01, 0.02, 0.015])
        conn = discount_connection(['A', 'B', 'C'], rates, dt=1.0)
        assert conn.log_rates.shape == (3, 3)

    def test_discount_connection_diagonal_zero(self):
        rates = jnp.array([0.01, 0.02, 0.015])
        conn = discount_connection(['A', 'B', 'C'], rates, dt=1.0)
        assert jnp.allclose(jnp.diag(conn.log_rates), jnp.zeros(3), atol=1e-6)

    def test_equal_rates_flat(self):
        """Equal interest rates → flat discount connection."""
        rates = jnp.array([0.05, 0.05, 0.05])
        conn = discount_connection(['A', 'B', 'C'], rates, dt=1.0)
        assert is_flat(conn)

    def test_unequal_rates_flat(self):
        """Discount connection from a scalar potential is always flat (exact gradient field)."""
        rates = jnp.array([0.01, 0.05, 0.03])
        conn = discount_connection(['A', 'B', 'C'], rates, dt=1.0)
        # A[i,j] = (r_j - r_i)*dt is an exact 1-form: F[i,j,k]=0 identically
        assert is_flat(conn)
