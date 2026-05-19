"""Tests for econiac.economics.pysd_backend — SDModel and system dynamics engine."""

import jax.numpy as jnp
import numpy as np
import pytest
from econiac.economics.pysd_backend import SDModel


def sir_model(N=1000.0, I0=10.0) -> SDModel:
    """SIR epidemiology model: classic system dynamics example."""
    model = SDModel()
    model.add_stock('S', initial=N - I0)
    model.add_stock('I', initial=I0)
    model.add_stock('R', initial=0.0)
    model.add_flow('infection', lambda s, t, p: p.get('beta', 0.3) * s['S'] * s['I'] / N,
                   from_stock='S', to_stock='I')
    model.add_flow('recovery', lambda s, t, p: p.get('gamma', 0.1) * s['I'],
                   from_stock='I', to_stock='R')
    return model


def exponential_model(growth=0.05) -> SDModel:
    """Simple exponential growth: dP/dt = r·P."""
    model = SDModel()
    model.add_stock('P', initial=100.0)
    model.add_flow('growth', lambda s, t, p: p.get('r', growth) * s['P'],
                   from_stock=None, to_stock='P')
    return model


class TestSDModel:
    def test_repr(self):
        m = sir_model()
        assert 'SDModel' in repr(m)

    def test_stock_names(self):
        m = sir_model()
        assert set(m.stock_names) == {'S', 'I', 'R'}

    def test_n_stocks(self):
        assert sir_model().n_stocks == 3

    def test_initial_state_shape(self):
        m = sir_model()
        s0 = m.initial_state()
        assert s0.shape == (3,)

    def test_initial_state_values(self):
        m = sir_model(N=1000.0, I0=10.0)
        s0 = m.initial_state()
        total = float(s0.sum())
        assert abs(total - 1000.0) < 1e-4

    def test_simulate_shape(self):
        m = sir_model()
        times, traj = m.simulate(T=10.0, dt=0.5)
        assert traj.shape == (21, 3)   # 10/0.5 + 1 = 21 steps

    def test_simulate_time_starts_zero(self):
        m = sir_model()
        times, _ = m.simulate(T=5.0, dt=1.0)
        assert float(times[0]) == 0.0

    def test_sir_population_conserved(self):
        """S + I + R = N throughout the simulation."""
        m = sir_model(N=1000.0)
        times, traj = m.simulate(T=50.0, dt=0.5)
        totals = traj.sum(axis=1)
        assert jnp.allclose(totals, 1000.0, atol=1.0)   # allow small Euler error

    def test_sir_epidemic_dynamics(self):
        """I should peak then decline in SIR model."""
        m = sir_model()
        times, traj = m.simulate(T=100.0, dt=0.5)
        I_traj = traj[:, 1]
        peak_idx = int(jnp.argmax(I_traj))
        assert peak_idx > 0           # epidemic takes off
        assert peak_idx < len(I_traj) - 1   # and comes back down

    def test_sir_s_monotone_decreasing(self):
        """S is monotone decreasing in SIR model."""
        m = sir_model()
        times, traj = m.simulate(T=50.0, dt=0.5)
        S = np.array(traj[:, 0])
        assert all(S[i] >= S[i+1] - 0.1 for i in range(len(S)-1))

    def test_exponential_growth(self):
        """dP/dt = r·P → P(T) ≈ P0·exp(r·T) for small T."""
        m = exponential_model()
        r = 0.05
        T = 5.0
        times, traj = m.simulate(T=T, dt=0.01, params={'r': r})
        P_final = float(traj[-1, 0])
        P_expected = 100.0 * np.exp(r * T)
        assert abs(P_final - P_expected) / P_expected < 0.01   # < 1% error

    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be closer to the analytic solution than Euler."""
        m = exponential_model()
        r = 0.1
        T = 10.0
        dt = 0.5
        P_analytic = 100.0 * np.exp(r * T)

        _, traj_e = m.simulate(T=T, dt=dt, method='euler', params={'r': r})
        _, traj_r = m.simulate(T=T, dt=dt, method='rk4',  params={'r': r})

        err_euler = abs(float(traj_e[-1, 0]) - P_analytic)
        err_rk4   = abs(float(traj_r[-1, 0]) - P_analytic)
        assert err_rk4 < err_euler

    def test_to_dict_keys(self):
        m = sir_model()
        _, traj = m.simulate(T=10.0, dt=1.0)
        d = m.to_dict(traj)
        assert set(d.keys()) == {'S', 'I', 'R'}

    def test_to_dict_shapes(self):
        m = sir_model()
        _, traj = m.simulate(T=10.0, dt=1.0)
        d = m.to_dict(traj)
        assert d['S'].shape == (11,)

    def test_auxiliary_variable(self):
        """Auxiliary variables are evaluated correctly."""
        m = SDModel()
        m.add_stock('X', initial=10.0)
        m.add_auxiliary('two_X', lambda s, t, p: 2 * s['X'])
        m.add_flow('growth', lambda s, t, p: 0.1 * s['two_X'],
                   from_stock=None, to_stock='X')
        times, traj = m.simulate(T=1.0, dt=0.1, params={})
        # X should grow faster with the auxiliary doubling it
        assert float(traj[-1, 0]) > float(traj[0, 0])

    def test_no_flows_stocks_constant(self):
        """A model with no flows has constant stocks."""
        m = SDModel()
        m.add_stock('K', initial=42.0)
        times, traj = m.simulate(T=5.0, dt=1.0)
        assert jnp.allclose(traj, 42.0)

    def test_require_pysd_raises_without_pysd(self):
        """compile_to_jax raises ImportError when pysd is not installed."""
        try:
            import pysd
            pytest.skip("pysd is installed")
        except ImportError:
            from econiac.economics.pysd_backend import compile_to_jax
            with pytest.raises(ImportError, match="pysd"):
                compile_to_jax("nonexistent.stmx")
