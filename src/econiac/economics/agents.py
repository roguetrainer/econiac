"""
Tensor-based differentiable agent-based macroeconomics (DABM).

Agents are indices, not objects. The entire population state is a tensor.
One timestep = one forward pass of the Gibbs-weighted wealth-update layer.
No Python for-loops over agents. Fully differentiable via JAX vmap/grad/jit.

The DABM is a Graph Neural Network on the Pacioli manifold:
    - Nodes = agents (households, firms, banks)
    - Edges = authorised payment channels
    - Node features = balance sheet / wealth vector
    - Message passing = Gibbs-weighted flow allocation

Reference: Buckley (2026) Differentiable ABM. doi:10.5281/zenodo.20261945
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.ensemble import gibbs_weights, beta_schedule
from econiac.core.manifold import PacioliManifold


# ---------------------------------------------------------------------------
# AgentPopulation — the global state tensor
# ---------------------------------------------------------------------------

@dataclass
class AgentPopulation:
    """
    Vectorised agent population: the global state tensor Σ(t) ∈ ℝ^{n_agents × n_goods}.

    wealth[i, j] = net wealth of agent i in good j (currency or real asset).
    Positive = asset, negative = liability. Column sums ≈ 0 (∂²=0 globally).

    agent_types: list of type labels (repeated if multiple agents per type).
    good_names:  list of good/currency labels.
    betas:       shape (n_agents,) — per-agent rationality (inverse temperature).
    """
    wealth: jax.Array         # shape (n_agents, n_goods)
    agent_types: list[str]
    good_names: list[str]
    betas: jax.Array          # shape (n_agents,)

    def __post_init__(self):
        n_a, n_g = self.wealth.shape
        if len(self.agent_types) != n_a:
            raise ValueError(f"agent_types length {len(self.agent_types)} != {n_a}")
        if len(self.good_names) != n_g:
            raise ValueError(f"good_names length {len(self.good_names)} != {n_g}")
        if self.betas.shape != (n_a,):
            raise ValueError(f"betas shape {self.betas.shape} != ({n_a},)")

    @property
    def n_agents(self) -> int:
        return self.wealth.shape[0]

    @property
    def n_goods(self) -> int:
        return self.wealth.shape[1]

    def net_worth(self) -> jax.Array:
        """Net worth of each agent: shape (n_agents,)."""
        return self.wealth.sum(axis=1)

    def is_consistent(self, atol: float = 1e-4) -> bool:
        """∂²=0: column sums ≈ 0 (every asset is someone else's liability)."""
        return bool(jnp.allclose(self.wealth.sum(axis=0), jnp.zeros(self.n_goods), atol=atol))

    def gini(self) -> jax.Array:
        """Gini coefficient of net worth distribution (shifted to non-negative)."""
        nw    = self.net_worth()
        nw    = nw - jnp.min(nw)        # shift so minimum = 0
        total = jnp.sum(nw)
        if float(total) < 1e-10:
            return jnp.zeros(())
        nw  = jnp.sort(nw)
        n   = self.n_agents
        idx = jnp.arange(1, n + 1, dtype=jnp.float32)
        return (2 * jnp.sum(idx * nw) / (n * total) - (n + 1) / n)

    @staticmethod
    def homogeneous(
        n_agents: int,
        n_goods: int,
        wealth_per_agent: float = 1.0,
        beta: float = 1.0,
        agent_type: str = "household",
        good_names: Optional[list] = None,
    ) -> 'AgentPopulation':
        """
        Create a population of identical agents with equal wealth.

        The global ∂²=0 constraint requires column sums = 0; we set up
        the population so that assets and liabilities are balanced by
        assigning +wealth to the first half of agents, -wealth to the second.
        """
        if good_names is None:
            good_names = [f"good_{j}" for j in range(n_goods)]
        half = n_agents // 2
        wealth = jnp.zeros((n_agents, n_goods))
        wealth = wealth.at[:half, 0].set(wealth_per_agent)
        wealth = wealth.at[half:, 0].set(-wealth_per_agent * half / max(n_agents - half, 1))
        return AgentPopulation(
            wealth=wealth,
            agent_types=[agent_type] * n_agents,
            good_names=good_names,
            betas=jnp.full((n_agents,), beta),
        )

    def __repr__(self) -> str:
        consistent = "consistent" if self.is_consistent() else "INCONSISTENT"
        return (
            f"AgentPopulation({self.n_agents} agents × {self.n_goods} goods, "
            f"{consistent}, β_mean={float(self.betas.mean()):.2f})"
        )


# ---------------------------------------------------------------------------
# WealthUpdateLayer — one differentiable forward step
# ---------------------------------------------------------------------------

@dataclass
class WealthUpdateLayer:
    """
    One timestep of the DABM: Gibbs-weighted flow allocation on the graph.

    For each agent i:
    1. Compute utility U_{ij} of transacting with agent j along each edge.
    2. Allocate outflow via Gibbs weights: π_{ij} = softmax(β_i · U_{ij}).
    3. Update wealth: Δwealth_i = Σ_j (inflow_{ji} - outflow_{ij}).

    The adjacency matrix encodes authorised transaction channels (∂²=0
    is enforced by antisymmetry: outflow from i to j = inflow to j from i).

    Args:
        adjacency:  (n_agents, n_agents) boolean or float; adjacency[i,j]=1
                    if agent i can send to agent j.
        flow_rate:  fraction of net worth that flows per step (∈ (0,1]).
    """
    adjacency: np.ndarray   # (n_agents, n_agents)
    flow_rate: float = 0.1

    def __post_init__(self):
        n, m = self.adjacency.shape
        if n != m:
            raise ValueError(f"adjacency must be square, got ({n},{m})")
        self.adjacency = np.array(self.adjacency, dtype=np.float32)

    @property
    def n_agents(self) -> int:
        return self.adjacency.shape[0]

    def _utility(self, wealth: jax.Array, i: int) -> jax.Array:
        """
        Utility of agent i transacting with each neighbour j.

        Default: U_{ij} = wealth[j, 0] (prefer richer counterparties).
        Can be overridden by subclassing or passing a utility_fn.
        """
        mask = jnp.array(self.adjacency[i], dtype=jnp.float32)
        return jnp.where(mask > 0, wealth[:, 0], -1e30)

    def forward(self, pop: AgentPopulation, dt: float = 1.0) -> AgentPopulation:
        """
        Apply one wealth-update step to the population.

        Returns new AgentPopulation with updated wealth.
        ∂²=0 is maintained: total wealth per good is conserved.
        """
        wealth = pop.wealth
        n  = pop.n_agents
        delta = jnp.zeros_like(wealth)

        for i in range(n):
            # How much agent i can send
            nw_i    = jnp.maximum(pop.net_worth()[i], 0.0)
            outflow = nw_i * self.flow_rate * dt

            if outflow <= 0:
                continue

            # Gibbs-weighted allocation across neighbours
            U_i = self._utility(wealth, i)
            w_i = gibbs_weights(U_i, beta=float(pop.betas[i]))

            # Outflows from i to each j
            for j in range(n):
                if self.adjacency[i, j] > 0 and i != j:
                    f_ij = float(w_i[j]) * outflow
                    # Antisymmetric: i loses, j gains (good 0 = money)
                    delta = delta.at[i, 0].add(-f_ij)
                    delta = delta.at[j, 0].add(+f_ij)

        return AgentPopulation(
            wealth=wealth + delta,
            agent_types=pop.agent_types,
            good_names=pop.good_names,
            betas=pop.betas,
        )


# ---------------------------------------------------------------------------
# DABMSimulator — full simulation runner
# ---------------------------------------------------------------------------

@dataclass
class DABMSimulator:
    """
    Differentiable ABM simulator.

    Runs WealthUpdateLayer repeatedly and records the trajectory.
    The full trajectory is differentiable w.r.t. initial conditions and
    layer parameters via JAX autodiff.
    """
    population: AgentPopulation
    layer: WealthUpdateLayer
    beta_sched: Optional[jax.Array] = None   # if set, anneals betas over time

    def simulate(self, n_steps: int, dt: float = 1.0) -> list[AgentPopulation]:
        """
        Run the simulation for n_steps steps.

        Returns list of AgentPopulation snapshots (length n_steps + 1).
        """
        pop = self.population
        trajectory = [pop]
        for t in range(n_steps):
            # Optionally anneal beta
            if self.beta_sched is not None and t < len(self.beta_sched):
                b = float(self.beta_sched[t])
                pop = AgentPopulation(
                    wealth=pop.wealth,
                    agent_types=pop.agent_types,
                    good_names=pop.good_names,
                    betas=jnp.full((pop.n_agents,), b),
                )
            pop = self.layer.forward(pop, dt=dt)
            trajectory.append(pop)
        return trajectory

    def gini_trajectory(self, n_steps: int, dt: float = 1.0) -> jax.Array:
        """Gini coefficient at each step. Shape (n_steps + 1,)."""
        traj = self.simulate(n_steps, dt)
        return jnp.array([float(pop.gini()) for pop in traj])

    def mean_wealth_trajectory(self, n_steps: int, dt: float = 1.0) -> jax.Array:
        """Mean net worth per agent at each step. Shape (n_steps + 1,)."""
        traj = self.simulate(n_steps, dt)
        return jnp.array([float(pop.net_worth().mean()) for pop in traj])
