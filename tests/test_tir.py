"""Tests for econiac.routing.tir — TIRInstance, route, free_energy, escape_arrow."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.routing.tir import (
    TIRInstance,
    route,
    free_energy,
    escape_arrow,
    admissible_count,
    routing_entropy,
    social_multiplier,
    tir_from_scores,
)
from econiac.core.geometry import (
    AbelianGeometry,
    FanoGeometry,
    G2Geometry,
    CatalanGeometry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def abelian_tir(n=4, beta=1.0):
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    geom = AbelianGeometry(adjacency=adj)
    utilities = jnp.arange(n, dtype=jnp.float32)
    return TIRInstance(
        candidates=list(range(n)),
        utilities=utilities,
        geometry=geom,
        beta=beta,
        geometry_kwargs={'source': 0},
    )


def fano_tir(beta=1.0, anchor_i=0, anchor_j=1):
    geom = FanoGeometry()
    utilities = jnp.ones(7, dtype=jnp.float32)
    return TIRInstance(
        candidates=list(range(7)),
        utilities=utilities,
        geometry=geom,
        beta=beta,
        geometry_kwargs={'anchor_i': anchor_i, 'anchor_j': anchor_j},
    )


# ---------------------------------------------------------------------------
# TIRInstance construction
# ---------------------------------------------------------------------------

class TestTIRInstance:
    def test_repr(self):
        tir = abelian_tir()
        assert 'TIRInstance' in repr(tir)

    def test_n_candidates(self):
        tir = abelian_tir(n=5)
        assert tir.n_candidates == 5

    def test_wrong_utility_shape_raises(self):
        geom = AbelianGeometry.complete(3)
        with pytest.raises(ValueError):
            TIRInstance(
                candidates=[0, 1, 2],
                utilities=jnp.ones(4),   # wrong shape
                geometry=geom,
                beta=1.0,
                geometry_kwargs={'source': 0},
            )

    def test_negative_beta_raises(self):
        geom = AbelianGeometry.complete(3)
        with pytest.raises(ValueError):
            TIRInstance(
                candidates=[0, 1, 2],
                utilities=jnp.ones(3),
                geometry=geom,
                beta=-1.0,
                geometry_kwargs={'source': 0},
            )

    def test_tir_from_scores(self):
        tir = tir_from_scores(['a', 'b', 'c'], [1.0, 2.0, 3.0], beta=2.0)
        assert tir.n_candidates == 3
        assert tir.beta == 2.0


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

class TestRoute:
    def test_weights_sum_to_one(self):
        tir = abelian_tir()
        w = route(tir)
        assert abs(float(w.sum()) - 1.0) < 1e-5

    def test_weights_non_negative(self):
        tir = abelian_tir()
        w = route(tir)
        assert bool(jnp.all(w >= 0))

    def test_high_beta_concentrates_on_argmax(self):
        tir = abelian_tir(n=4, beta=100.0)
        w = route(tir)
        best = int(jnp.argmax(tir.utilities))
        assert float(w[best]) > 0.99

    def test_zero_beta_uniform(self):
        tir = abelian_tir(n=4, beta=0.0)
        w = route(tir)
        # All 3 reachable candidates (source=0 excluded by adjacency diagonal)
        # should be nearly equal
        reachable_w = jnp.array([w[i] for i in [1, 2, 3]])
        assert float(reachable_w.std()) < 1e-4

    def test_fano_only_one_admissible(self):
        """Given two anchors, exactly one Fano point is admissible."""
        tir = fano_tir(anchor_i=0, anchor_j=1)
        w = route(tir)
        n_nonzero = int(jnp.sum(w > 1e-6))
        assert n_nonzero == 1

    def test_fano_admissible_on_line(self):
        """The admitted point should be the one on the Fano line {0,1,3}."""
        tir = fano_tir(anchor_i=0, anchor_j=1)
        w = route(tir)
        assert float(w[3]) > 0.99

    def test_isolated_source_zero_weight(self):
        """Node unreachable from source gets zero weight."""
        adj = np.zeros((4, 4), dtype=bool)
        adj[0, 1] = True
        geom = AbelianGeometry(adjacency=adj)
        tir = TIRInstance(
            candidates=list(range(4)),
            utilities=jnp.ones(4),
            geometry=geom,
            beta=1.0,
            geometry_kwargs={'source': 0},
        )
        w = route(tir)
        assert float(w[2]) < 1e-6
        assert float(w[3]) < 1e-6


# ---------------------------------------------------------------------------
# free_energy
# ---------------------------------------------------------------------------

class TestFreeEnergy:
    def test_free_energy_scalar(self):
        U = jnp.array([1.0, 2.0, 3.0])
        F = free_energy(U, beta=1.0)
        assert F.shape == ()

    def test_free_energy_increases_with_beta(self):
        """F(β) = -log(Z)/β increases toward -max(U) as β → ∞."""
        U = jnp.array([0.0, 1.0, 2.0])
        F_lo = float(free_energy(U, beta=0.1))
        F_hi = float(free_energy(U, beta=10.0))
        # F_lo ≈ -log(3)/0.1 < F_hi ≈ -2; F increases with β toward -max(U)
        assert F_hi > F_lo

    def test_free_energy_approaches_neg_max_utility(self):
        """At large β, F → -max(U)."""
        U = jnp.array([1.0, 2.0, 5.0])
        F = float(free_energy(U, beta=1000.0))
        assert abs(F - (-5.0)) < 0.01

    def test_free_energy_zero_beta(self):
        """At β=0, F = -log(n_admissible)."""
        U = jnp.array([1.0, 1.0, 1.0])
        F = float(free_energy(U, beta=0.0))
        assert abs(F - (-np.log(3))) < 1e-4


# ---------------------------------------------------------------------------
# escape_arrow
# ---------------------------------------------------------------------------

class TestEscapeArrow:
    def test_stochastic_escapes(self):
        """Any β < ∞ escapes Arrow's theorem."""
        tir = abelian_tir(beta=1.0)
        assert escape_arrow(tir) is True

    def test_fano_geometry_escapes(self):
        """Non-Abelian geometry escapes even at high β."""
        tir = fano_tir(beta=1e9)
        assert escape_arrow(tir) is True

    def test_abelian_high_beta_does_not_escape(self):
        """Abelian + deterministic (high β) = Arrow applies."""
        tir = abelian_tir(beta=1e9)
        assert escape_arrow(tir) is False


# ---------------------------------------------------------------------------
# admissible_count
# ---------------------------------------------------------------------------

class TestAdmissibleCount:
    def test_complete_graph_all_admissible(self):
        tir = abelian_tir(n=4)
        # BFS from source marks source + 3 neighbours as reachable = 4
        assert admissible_count(tir) == 4

    def test_fano_single_admissible(self):
        tir = fano_tir(anchor_i=0, anchor_j=1)
        assert admissible_count(tir) == 1


# ---------------------------------------------------------------------------
# routing_entropy
# ---------------------------------------------------------------------------

class TestRoutingEntropy:
    def test_entropy_non_negative(self):
        tir = abelian_tir()
        H = float(routing_entropy(tir))
        assert H >= 0

    def test_entropy_decreases_with_beta(self):
        tir_lo = abelian_tir(beta=0.1)
        tir_hi = abelian_tir(beta=10.0)
        assert float(routing_entropy(tir_lo)) > float(routing_entropy(tir_hi))

    def test_entropy_max_uniform(self):
        """Uniform routing (β=0) maximises entropy."""
        tir = tir_from_scores(['a', 'b', 'c'], [1.0, 1.0, 1.0], beta=0.0)
        H = float(routing_entropy(tir))
        assert abs(H - np.log(3)) < 0.01


# ---------------------------------------------------------------------------
# social_multiplier
# ---------------------------------------------------------------------------

class TestSocialMultiplier:
    def test_social_multiplier_ge_one(self):
        tir = abelian_tir()
        chi = float(social_multiplier(tir))
        assert chi >= 1.0

    def test_social_multiplier_le_n(self):
        tir = abelian_tir(n=4, beta=0.0)
        chi = float(social_multiplier(tir))
        # At β=0, χ ≈ n_admissible = 3
        assert chi <= 4.0

    def test_social_multiplier_winner_take_all(self):
        """At very high β, χ → 1."""
        tir = abelian_tir(n=4, beta=100.0)
        chi = float(social_multiplier(tir))
        assert chi < 1.1
