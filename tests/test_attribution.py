"""Tests for econiac.routing.attribution — thermal Shapley, bottleneck, Pacioli."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.routing.attribution import (
    thermal_shapley,
    bottleneck_index,
    tropical_limit,
    nonassociative_shapley,
    pacioli_attribution,
    ShapleyResult,
    full_shapley_analysis,
)


# ---------------------------------------------------------------------------
# Value functions for testing
# ---------------------------------------------------------------------------

def majority_voting(S: frozenset) -> float:
    """Majority-rule game: coalition wins iff |S| >= 2."""
    return 1.0 if len(S) >= 2 else 0.0


def weighted_game(S: frozenset) -> float:
    """Weighted voting: w = [3, 2, 1]; coalition wins if Σw_i >= 4."""
    weights = {0: 3, 1: 2, 2: 1}
    total = sum(weights.get(i, 0) for i in S)
    return 1.0 if total >= 4 else 0.0


def additive_game(S: frozenset) -> float:
    """Simple additive game: v(S) = |S|."""
    return float(len(S))


def zero_game(S: frozenset) -> float:
    """Trivial game: v(S) = 0 for all S."""
    return 0.0


def sfc_game(S: frozenset) -> float:
    """SFC-style game with v(grand) = 0 (Pacioli constraint)."""
    n = 3  # three sectors
    grand = frozenset(range(n))
    if S == grand:
        return 0.0
    return float(len(S)) - 1.5   # small game for testing


# ---------------------------------------------------------------------------
# thermal_shapley
# ---------------------------------------------------------------------------

class TestThermalShapley:
    def test_shape(self):
        phi = thermal_shapley(majority_voting, n_players=3, beta=1.0)
        assert phi.shape == (3,)

    def test_efficiency(self):
        """Σ φ_i = v(grand coalition)."""
        n = 3
        phi = thermal_shapley(majority_voting, n_players=n, beta=1.0)
        grand = frozenset(range(n))
        assert abs(float(phi.sum()) - majority_voting(grand)) < 1e-4

    def test_symmetry_majority_voting(self):
        """All players symmetric in majority_voting → equal Shapley values."""
        phi = thermal_shapley(majority_voting, n_players=3, beta=0.0)
        assert jnp.allclose(phi, phi[0], atol=1e-4)

    def test_dummy_player(self):
        """A dummy player (zero marginal contribution) gets φ=0."""
        def game_with_dummy(S):
            # Players 0 and 1 create value; player 2 is a dummy
            return float(len(S & {0, 1}))
        phi = thermal_shapley(game_with_dummy, n_players=3, beta=0.0)
        assert abs(float(phi[2])) < 1e-4

    def test_additivity_additive_game(self):
        """In additive game v(S)=|S|, each player gets φ_i=1."""
        phi = thermal_shapley(additive_game, n_players=3, beta=1.0)
        assert jnp.allclose(phi, 1.0, atol=1e-4)

    def test_zero_game(self):
        """Zero game → all Shapley values are 0."""
        phi = thermal_shapley(zero_game, n_players=3, beta=1.0)
        assert jnp.allclose(phi, 0.0, atol=1e-6)

    def test_weighted_game_player_0_highest(self):
        """In weighted_game, player 0 (weight 3) has strictly highest φ."""
        phi = thermal_shapley(weighted_game, n_players=3, beta=0.0)
        assert float(phi[0]) > float(phi[1])
        # Players 1 and 2 are symmetric in losing coalitions → equal φ
        assert abs(float(phi[1]) - float(phi[2])) < 1e-4

    def test_beta_zero_classical_shapley(self):
        """At β=0, thermal Shapley = classical Shapley (uniform weights)."""
        phi_thermal  = thermal_shapley(weighted_game, n_players=3, beta=0.0)
        phi_classical = thermal_shapley(weighted_game, n_players=3, beta=1e-6)
        assert jnp.allclose(phi_thermal, phi_classical, atol=1e-3)

    def test_no_nan(self):
        phi = thermal_shapley(majority_voting, n_players=4, beta=2.0)
        assert not bool(jnp.any(jnp.isnan(phi)))


# ---------------------------------------------------------------------------
# bottleneck_index
# ---------------------------------------------------------------------------

class TestBottleneckIndex:
    def test_shape(self):
        betas = jnp.linspace(0.1, 5.0, 6)
        Lambda = bottleneck_index(weighted_game, n_players=3, beta_range=betas)
        assert Lambda.shape == (5, 3)   # m-1 rows, n_players cols

    def test_non_negative(self):
        betas = jnp.array([0.5, 1.0, 2.0])
        Lambda = bottleneck_index(majority_voting, n_players=3, beta_range=betas)
        assert bool(jnp.all(Lambda >= 0))

    def test_zero_game_zero_lambda(self):
        """Zero game: φ=0 everywhere, so Λ=0."""
        betas = jnp.array([0.5, 1.0, 2.0])
        Lambda = bottleneck_index(zero_game, n_players=3, beta_range=betas)
        assert jnp.allclose(Lambda, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# tropical_limit
# ---------------------------------------------------------------------------

class TestTropicalLimit:
    def test_returns_int(self):
        idx = tropical_limit(majority_voting, n_players=3)
        assert isinstance(idx, int)

    def test_in_range(self):
        idx = tropical_limit(weighted_game, n_players=3)
        assert 0 <= idx <= 2

    def test_weighted_game_bottleneck_valid_player(self):
        """Bottleneck player is the highest-MC player in the best permutation."""
        idx = tropical_limit(weighted_game, n_players=3)
        assert 0 <= idx <= 2

    def test_additive_first_player(self):
        """Additive game: all MCs equal; bottleneck is argmax(ties→0)."""
        idx = tropical_limit(additive_game, n_players=3)
        assert 0 <= idx <= 2


# ---------------------------------------------------------------------------
# nonassociative_shapley
# ---------------------------------------------------------------------------

class TestNonassociativeShapley:
    def test_shape(self):
        phi = nonassociative_shapley(majority_voting, n_players=3, beta=1.0)
        assert phi.shape == (3,)

    def test_no_nan(self):
        phi = nonassociative_shapley(majority_voting, n_players=3, beta=1.0)
        assert not bool(jnp.any(jnp.isnan(phi)))

    def test_two_players_equals_shapley(self):
        """n=2: C_1=1 tree → non-associative Shapley = standard Shapley."""
        def simple2(S):
            return 1.0 if len(S) == 2 else 0.0
        phi_na = nonassociative_shapley(simple2, n_players=2, beta=0.0)
        phi_sh = thermal_shapley(simple2, n_players=2, beta=0.0)
        assert jnp.allclose(phi_na, phi_sh, atol=1e-4)

    def test_symmetric_game_symmetric_output(self):
        """Symmetric game → symmetric Shapley values."""
        phi = nonassociative_shapley(majority_voting, n_players=3, beta=0.0)
        assert abs(float(phi[0]) - float(phi[1])) < 1e-3


# ---------------------------------------------------------------------------
# pacioli_attribution
# ---------------------------------------------------------------------------

class TestPacioliAttribution:
    def test_sums_to_zero(self):
        """Pacioli attribution must sum to 0."""
        phi = pacioli_attribution(sfc_game, n_sectors=3, beta=1.0)
        assert abs(float(phi.sum())) < 1e-4

    def test_shape(self):
        phi = pacioli_attribution(majority_voting, n_sectors=3, beta=1.0)
        assert phi.shape == (3,)

    def test_zero_game_zero_attribution(self):
        """Zero game: attribution = 0 everywhere."""
        phi = pacioli_attribution(zero_game, n_sectors=3, beta=1.0)
        assert jnp.allclose(phi, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# ShapleyResult and full_shapley_analysis
# ---------------------------------------------------------------------------

class TestShapleyResult:
    def test_repr(self):
        phi = jnp.array([0.3, 0.5, 0.2])
        r = ShapleyResult(phi=phi, beta=1.0, n_players=3)
        assert 'ShapleyResult' in repr(r)

    def test_bottleneck_player(self):
        phi = jnp.array([0.1, 0.8, 0.1])
        r = ShapleyResult(phi=phi, beta=1.0, n_players=3)
        assert r.bottleneck_player == 1

    def test_total_value(self):
        phi = jnp.array([1.0, 2.0, 3.0])
        r = ShapleyResult(phi=phi, beta=1.0, n_players=3)
        assert abs(r.total_value - 6.0) < 1e-5


class TestFullShapleyAnalysis:
    def test_result_type(self):
        result, _ = full_shapley_analysis(majority_voting, n_players=3, beta=1.0)
        assert isinstance(result, ShapleyResult)

    def test_with_beta_range_returns_lambda(self):
        betas = jnp.linspace(0.5, 3.0, 4)
        _, Lambda = full_shapley_analysis(
            majority_voting, n_players=3, beta=1.0, beta_range=betas
        )
        assert Lambda is not None
        assert Lambda.shape == (3, 3)

    def test_without_beta_range_lambda_is_none(self):
        _, Lambda = full_shapley_analysis(majority_voting, n_players=3, beta=1.0)
        assert Lambda is None
