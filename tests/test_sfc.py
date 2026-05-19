"""Tests for econiac.economics.sfc — SFCModel, SFCParameters, SFCState."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.core.manifold import three_sector_sfc
from econiac.economics.sfc import SFCModel, SFCParameters, SFCState


def make_model(beta=1.0):
    bs, manifold = three_sector_sfc()
    params = SFCParameters.uniform(manifold.n_edges, beta=beta)
    return SFCModel(manifold=manifold, initial_bs=bs, params=params)


class TestSFCParameters:
    def test_uniform_shape(self):
        p = SFCParameters.uniform(3)
        assert p.propensities.shape == (3,)

    def test_uniform_sum_to_one(self):
        p = SFCParameters.uniform(4)
        assert jnp.allclose(p.propensities.sum(), 1.0, atol=1e-5)

    def test_negative_propensity_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SFCParameters(propensities=jnp.array([-0.1, 0.5, 0.6]))

    def test_beta_stored(self):
        p = SFCParameters.uniform(3, beta=2.5)
        assert p.beta == 2.5


class TestSFCModel:
    def test_repr(self):
        m = make_model()
        r = repr(m)
        assert 'SFCModel' in r
        assert 'H₁' in r

    def test_mismatched_nodes_raises(self):
        bs, manifold = three_sector_sfc()
        # manifold has 3 nodes, but give 4-sector params
        params = SFCParameters.uniform(manifold.n_edges)
        with pytest.raises(ValueError):
            SFCModel(
                manifold=manifold,
                initial_bs=bs,
                params=SFCParameters(propensities=jnp.ones(manifold.n_edges + 1) / (manifold.n_edges + 1)),
            )

    def test_step_returns_sfcstate(self):
        m = make_model()
        s0 = SFCState(balance_sheet=m.initial_bs, t=0.0)
        s1 = m.step(s0, dt=1.0)
        assert isinstance(s1, SFCState)
        assert s1.t == 1.0

    def test_step_preserves_sector_count(self):
        m = make_model()
        s0 = SFCState(balance_sheet=m.initial_bs)
        s1 = m.step(s0, dt=1.0)
        assert s1.balance_sheet.n_sectors == m.initial_bs.n_sectors

    def test_simulate_length(self):
        m = make_model()
        traj = m.simulate(T=5.0, dt=1.0)
        assert len(traj) == 6   # initial + 5 steps

    def test_simulate_time_increases(self):
        m = make_model()
        traj = m.simulate(T=3.0, dt=1.0)
        times = [s.t for s in traj]
        assert times == [0.0, 1.0, 2.0, 3.0]

    def test_gdp_positive(self):
        m = make_model()
        s0 = SFCState(balance_sheet=m.initial_bs)
        assert float(m.gdp(s0)) > 0

    def test_social_multiplier_shape(self):
        m = make_model()
        betas = jnp.array([0.1, 1.0, 10.0])
        chi = m.social_multiplier(betas)
        assert chi.shape == (3,)

    def test_social_multiplier_decreases_with_beta(self):
        """Higher β → more concentrated → lower participation ratio."""
        m = make_model()
        betas = jnp.array([0.01, 1.0, 10.0, 100.0])
        chi = m.social_multiplier(betas)
        # Should be non-increasing
        assert all(float(chi[i]) >= float(chi[i+1]) - 1e-3 for i in range(len(chi)-1))

    def test_social_multiplier_at_low_beta_near_n(self):
        """β → 0: all flows equally likely → effective n ≈ n_flows."""
        m = make_model(beta=0.001)
        betas = jnp.array([0.001])
        chi = m.social_multiplier(betas)
        assert float(chi[0]) > 1.5   # at least more than 1 effective participant

    def test_custom_utility_fn(self):
        """Custom utility function is called instead of default."""
        bs, manifold = three_sector_sfc()
        called = [False]

        def my_utility(bs, params):
            called[0] = True
            return jnp.ones(manifold.n_edges)

        params = SFCParameters.uniform(manifold.n_edges)
        m = SFCModel(manifold=manifold, initial_bs=bs, params=params, utility_fn=my_utility)
        s0 = SFCState(balance_sheet=bs)
        m.step(s0, dt=1.0)
        assert called[0]
