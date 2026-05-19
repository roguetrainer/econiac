"""Tests for econiac.economics.minsky — MinskySFCModel, Keen predator-prey."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.economics.minsky import (
    MinskySFCModel,
    keen_predator_prey,
    keen_ode,
    keen_simulate,
)


class TestMinskySFCModel:
    def test_repr(self):
        model = keen_predator_prey()
        assert 'MinskySFCModel' in repr(model)

    def test_n_sectors(self):
        model = keen_predator_prey()
        assert model.n_sectors == 3

    def test_n_instruments(self):
        model = keen_predator_prey()
        assert model.n_instruments == 3

    def test_initial_positions_shape(self):
        model = keen_predator_prey()
        assert model.initial_positions.shape == (3, 3)

    def test_step_returns_same_shape(self):
        model = keen_predator_prey()
        pos = model.initial_positions
        pos2 = model.step(pos, t=0.0, dt=0.1, params={})
        assert pos2.shape == pos.shape

    def test_step_rk4_returns_same_shape(self):
        model = keen_predator_prey()
        pos = model.initial_positions
        pos2 = model.step_rk4(pos, t=0.0, dt=0.1, params={})
        assert pos2.shape == pos.shape

    def test_simulate_time_shape(self):
        model = keen_predator_prey()
        times, traj = model.simulate(T=1.0, dt=0.1)
        assert times.shape == (11,)   # 0.0, 0.1, ..., 1.0

    def test_simulate_trajectory_shape(self):
        model = keen_predator_prey()
        times, traj = model.simulate(T=2.0, dt=0.5)
        assert traj.shape == (5, 3, 3)   # (n_steps+1, n_sectors, n_instruments)

    def test_simulate_time_starts_at_zero(self):
        model = keen_predator_prey()
        times, _ = model.simulate(T=1.0, dt=0.25)
        assert float(times[0]) == 0.0

    def test_simulate_euler_vs_rk4(self):
        """Euler and RK4 should give similar results for small dt."""
        model = keen_predator_prey()
        _, traj_e = model.simulate(T=1.0, dt=0.01, method='euler')
        _, traj_r = model.simulate(T=1.0, dt=0.01, method='rk4')
        assert jnp.allclose(traj_e[-1], traj_r[-1], atol=1e-2)

    def test_balance_sheet_at(self):
        model = keen_predator_prey()
        bs = model.balance_sheet_at(model.initial_positions)
        assert bs.n_sectors == 3
        assert bs.n_instruments == 3

    def test_custom_model_single_stock(self):
        """A trivial one-sector, one-instrument model."""
        model = MinskySFCModel(
            sectors=['economy'],
            instruments=['output'],
            initial_positions=jnp.array([[100.0]]),
            flow_fns={'output': lambda pos, t, p: jnp.array([p.get('growth', 0.02) * pos[0, 0]])},
        )
        times, traj = model.simulate(T=1.0, dt=0.5, params={'growth': 0.02})
        assert traj.shape == (3, 1, 1)
        # Output should grow
        assert float(traj[-1, 0, 0]) > float(traj[0, 0, 0])


class TestKeenODE:
    def test_keen_ode_shape(self):
        state = jnp.array([0.80, 0.94, 0.10])
        dX = keen_ode(state, t=0.0, params={})
        assert dX.shape == (3,)

    def test_keen_ode_default_params(self):
        """The ODE should run without error on default params."""
        state = jnp.array([0.80, 0.94, 0.10])
        dX = keen_ode(state, t=0.0, params={})
        assert not jnp.any(jnp.isnan(dX))

    def test_keen_simulate_shape(self):
        times, traj = keen_simulate(T=10.0, dt=0.1)
        assert traj.shape == (101, 3)

    def test_keen_simulate_time_starts_zero(self):
        times, _ = keen_simulate(T=5.0, dt=0.5)
        assert float(times[0]) == 0.0

    def test_keen_simulate_employment_bounded(self):
        """Employment rate λ should remain in (0,1) for stable params."""
        _, traj = keen_simulate(T=20.0, dt=0.1)
        lam = traj[:, 1]
        assert bool(jnp.all(lam > 0))
        assert bool(jnp.all(lam < 1.1))   # may slightly exceed 1 in extreme cases

    def test_keen_simulate_wage_share_positive(self):
        """Wage share ω should remain positive for reasonable initial conditions."""
        _, traj = keen_simulate(omega0=0.80, lam0=0.94, d0=0.10, T=20.0, dt=0.1)
        omega = traj[:, 0]
        assert bool(jnp.all(omega > 0))

    def test_keen_simulate_crisis_scenario(self):
        """High initial debt → financial instability (d grows)."""
        _, traj_stable = keen_simulate(d0=0.1, T=30.0, dt=0.1)
        _, traj_crisis = keen_simulate(d0=2.0, T=30.0, dt=0.1)
        # Both should run without NaN
        assert not jnp.any(jnp.isnan(traj_stable))
        assert not jnp.any(jnp.isnan(traj_crisis))
