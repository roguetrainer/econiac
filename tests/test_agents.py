"""Tests for econiac.economics.agents — AgentPopulation, WealthUpdateLayer, DABMSimulator."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.economics.agents import (
    AgentPopulation,
    WealthUpdateLayer,
    DABMSimulator,
)


def simple_pop(n=4, n_goods=2, beta=1.0):
    return AgentPopulation.homogeneous(n, n_goods, wealth_per_agent=100.0, beta=beta)


def complete_adj(n):
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    return adj


class TestAgentPopulation:
    def test_homogeneous_shape(self):
        pop = simple_pop()
        assert pop.wealth.shape == (4, 2)

    def test_n_agents(self):
        assert simple_pop(6).n_agents == 6

    def test_n_goods(self):
        assert simple_pop(4, 3).n_goods == 3

    def test_net_worth_shape(self):
        pop = simple_pop()
        assert pop.net_worth().shape == (4,)

    def test_is_consistent(self):
        """homogeneous() constructs a ∂²=0-consistent population."""
        pop = simple_pop(4, 1)
        assert pop.is_consistent()

    def test_betas_shape(self):
        pop = simple_pop(5, 2, beta=2.0)
        assert pop.betas.shape == (5,)
        assert jnp.allclose(pop.betas, 2.0)

    def test_wrong_agent_types_raises(self):
        with pytest.raises(ValueError):
            AgentPopulation(
                wealth=jnp.zeros((3, 2)),
                agent_types=['a', 'b'],   # wrong length
                good_names=['x', 'y'],
                betas=jnp.ones(3),
            )

    def test_wrong_good_names_raises(self):
        with pytest.raises(ValueError):
            AgentPopulation(
                wealth=jnp.zeros((3, 2)),
                agent_types=['a', 'b', 'c'],
                good_names=['x'],   # wrong length
                betas=jnp.ones(3),
            )

    def test_wrong_betas_shape_raises(self):
        with pytest.raises(ValueError):
            AgentPopulation(
                wealth=jnp.zeros((3, 2)),
                agent_types=['a', 'b', 'c'],
                good_names=['x', 'y'],
                betas=jnp.ones(5),   # wrong length
            )

    def test_gini_range(self):
        pop = simple_pop()
        g = float(pop.gini())
        assert -0.1 <= g <= 1.1   # Gini is in [0,1] for non-negative wealth

    def test_repr(self):
        assert 'AgentPopulation' in repr(simple_pop())

    def test_custom_good_names(self):
        pop = AgentPopulation.homogeneous(4, 2, good_names=['USD', 'EUR'])
        assert pop.good_names == ['USD', 'EUR']


class TestWealthUpdateLayer:
    def test_forward_preserves_n_agents(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4), flow_rate=0.1)
        pop2 = layer.forward(pop, dt=1.0)
        assert pop2.n_agents == 4

    def test_forward_preserves_total_wealth(self):
        """Antisymmetric flows conserve total wealth per good."""
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4), flow_rate=0.1)
        pop2 = layer.forward(pop, dt=1.0)
        total_before = float(pop.wealth[:, 0].sum())
        total_after  = float(pop2.wealth[:, 0].sum())
        assert abs(total_before - total_after) < 1e-3

    def test_non_square_adjacency_raises(self):
        with pytest.raises(ValueError):
            WealthUpdateLayer(np.ones((3, 4)))

    def test_zero_flow_rate_no_change(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4), flow_rate=0.0)
        pop2 = layer.forward(pop, dt=1.0)
        assert jnp.allclose(pop.wealth, pop2.wealth)

    def test_forward_returns_agent_population(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4))
        result = layer.forward(pop)
        assert isinstance(result, AgentPopulation)

    def test_isolated_agents_no_flow(self):
        """Disconnected graph → no wealth flows."""
        pop = simple_pop(4, 1)
        adj = np.zeros((4, 4))  # no edges
        layer = WealthUpdateLayer(adj, flow_rate=0.1)
        pop2 = layer.forward(pop, dt=1.0)
        assert jnp.allclose(pop.wealth, pop2.wealth)


class TestDABMSimulator:
    def test_simulate_length(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4))
        sim = DABMSimulator(population=pop, layer=layer)
        traj = sim.simulate(n_steps=5, dt=1.0)
        assert len(traj) == 6   # initial + 5

    def test_simulate_returns_populations(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4))
        sim = DABMSimulator(population=pop, layer=layer)
        traj = sim.simulate(n_steps=3)
        assert all(isinstance(p, AgentPopulation) for p in traj)

    def test_gini_trajectory_shape(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4))
        sim = DABMSimulator(population=pop, layer=layer)
        g = sim.gini_trajectory(n_steps=5)
        assert g.shape == (6,)

    def test_mean_wealth_trajectory_shape(self):
        pop = simple_pop(4, 1)
        layer = WealthUpdateLayer(complete_adj(4))
        sim = DABMSimulator(population=pop, layer=layer)
        mw = sim.mean_wealth_trajectory(n_steps=5)
        assert mw.shape == (6,)

    def test_beta_schedule_applied(self):
        """Annealing beta should change allocation over time."""
        from econiac.core.ensemble import beta_schedule
        sched = beta_schedule(0.1, 10.0, 5)
        pop = simple_pop(4, 1, beta=0.1)
        layer = WealthUpdateLayer(complete_adj(4), flow_rate=0.1)
        sim = DABMSimulator(population=pop, layer=layer, beta_sched=sched)
        traj = sim.simulate(n_steps=4)
        # At step 4 beta should have increased
        assert float(traj[-1].betas[0]) > float(traj[0].betas[0]) - 1e-3
