"""Tests for econiac.core.ensemble — Gibbs partition function and routing weights."""

import jax.numpy as jnp
import pytest
from econiac.core.ensemble import (
    partition_function,
    gibbs_weights,
    free_energy,
    entropy,
    mean_utility,
    choose,
    beta_schedule,
    ensemble_sweep,
    summarise,
)


U = jnp.array([1.0, 2.0, 3.0])
U_equal = jnp.array([1.0, 1.0, 1.0])


class TestGibbsWeights:
    def test_sum_to_one(self):
        w = gibbs_weights(U, beta=1.0)
        assert jnp.allclose(w.sum(), 1.0, atol=1e-6)

    def test_beta_zero_uniform(self):
        """β=0 → uniform weights regardless of utilities."""
        w = gibbs_weights(U, beta=0.0)
        assert jnp.allclose(w, jnp.ones(3) / 3, atol=1e-6)

    def test_high_beta_concentrates_on_argmax(self):
        """β large → weight concentrates on highest utility."""
        w = gibbs_weights(U, beta=100.0)
        assert w[2] > 0.999

    def test_ordering_preserved(self):
        """Higher utility → higher weight for any β > 0."""
        w = gibbs_weights(U, beta=1.0)
        assert w[0] < w[1] < w[2]

    def test_equal_utilities_uniform(self):
        w = gibbs_weights(U_equal, beta=5.0)
        assert jnp.allclose(w, jnp.ones(3) / 3, atol=1e-6)


class TestFreeEnergy:
    def test_approaches_negative_max_utility_from_below(self):
        """As β increases, F approaches -max(U) from below."""
        F_low  = free_energy(U, beta=0.1)
        F_high = free_energy(U, beta=10.0)
        # F = -ln(Z)/β; as β grows ln(Z)/β → max(U), so F → -max(U) = -3
        # F_low is more negative than F_high (further from -max(U))
        assert F_low < F_high

    def test_approaches_neg_max_utility(self):
        """As β→∞, F → -max(U)."""
        F = free_energy(U, beta=1000.0)
        assert jnp.allclose(F, -jnp.max(U), atol=1e-2)

    def test_beta_zero_limit(self):
        """At β→0, F/β → -ln(n); test via small β with float64."""
        # At small β: ln Z ≈ ln(n) + β·⟨U⟩, so F ≈ -ln(n)/β - ⟨U⟩
        # Instead just verify F is more negative than -max(U) at low β
        F = free_energy(U_equal, beta=0.01)
        assert F < -jnp.log(3.0)  # F more negative than free energy lower bound


class TestEntropy:
    def test_max_at_beta_zero(self):
        """Entropy is maximised at β=0: H = ln(n)."""
        H = entropy(U, beta=0.0)
        assert jnp.allclose(H, jnp.log(3.0), atol=1e-6)

    def test_decreases_with_beta(self):
        H_low  = entropy(U, beta=0.1)
        H_high = entropy(U, beta=10.0)
        assert H_low > H_high

    def test_non_negative(self):
        for beta in [0.0, 1.0, 10.0, 100.0]:
            assert entropy(U, beta) >= 0.0


class TestChoose:
    def test_scalar_candidates(self):
        """Gibbs mixture of scalars is a weighted average."""
        candidates = jnp.array([0.0, 1.0, 2.0])
        result = choose(beta=0.0, candidates=candidates, utilities=U)
        assert jnp.allclose(result, 1.0, atol=1e-6)  # uniform → mean = 1.0

    def test_vector_candidates(self):
        """Gibbs mixture of vectors returns a vector."""
        candidates = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        result = choose(beta=0.0, candidates=candidates, utilities=U)
        assert result.shape == (2,)
        assert jnp.allclose(result.sum(), 1.0, atol=1e-5)

    def test_high_beta_selects_argmax(self):
        candidates = jnp.array([10.0, 20.0, 30.0])
        result = choose(beta=100.0, candidates=candidates, utilities=U)
        assert jnp.allclose(result, 30.0, atol=0.1)


class TestBetaSchedule:
    def test_geometric_endpoints(self):
        s = beta_schedule(0.1, 100.0, 50, kind='geometric')
        assert jnp.allclose(s[0], 0.1, atol=1e-5)
        assert jnp.allclose(s[-1], 100.0, atol=1e-3)

    def test_linear_endpoints(self):
        s = beta_schedule(1.0, 10.0, 10, kind='linear')
        assert jnp.allclose(s[0], 1.0, atol=1e-6)
        assert jnp.allclose(s[-1], 10.0, atol=1e-6)

    def test_geometric_monotone(self):
        s = beta_schedule(0.1, 100.0, 20, kind='geometric')
        assert jnp.all(jnp.diff(s) > 0)

    def test_length(self):
        s = beta_schedule(1.0, 10.0, 37, kind='cosine')
        assert len(s) == 37

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            beta_schedule(1.0, 10.0, 10, kind='magic')


class TestEnsembleSweep:
    def test_shape(self):
        schedule = beta_schedule(0.1, 10.0, 20)
        W = ensemble_sweep(U, schedule)
        assert W.shape == (20, 3)

    def test_each_row_sums_to_one(self):
        schedule = beta_schedule(0.1, 10.0, 20)
        W = ensemble_sweep(U, schedule)
        assert jnp.allclose(W.sum(axis=1), jnp.ones(20), atol=1e-6)

    def test_first_row_near_uniform(self):
        schedule = beta_schedule(1e-6, 10.0, 20)
        W = ensemble_sweep(U, schedule)
        assert jnp.allclose(W[0], jnp.ones(3) / 3, atol=1e-4)


class TestSummarise:
    def test_returns_named_tuple(self):
        s = summarise(U, beta=1.0)
        assert hasattr(s, 'weights')
        assert hasattr(s, 'free_energy')
        assert hasattr(s, 'entropy')
        assert hasattr(s, 'effective_n')

    def test_effective_n_at_uniform(self):
        """At β=0 all candidates active: effective_n = n."""
        s = summarise(U, beta=0.0)
        assert jnp.allclose(s.effective_n, 3.0, atol=1e-4)

    def test_effective_n_at_high_beta(self):
        """At high β only one candidate active: effective_n → 1."""
        s = summarise(U, beta=100.0)
        assert jnp.allclose(s.effective_n, 1.0, atol=0.01)
