"""
Supply Chain Reverse Stress Testing (RST) via Differentiable ADTs.

Implements the Smith et al. (2025) reverse stress testing framework using
JAX-native differentiable logic, exposing the Sum/Product type duality as
PCL combinators.

Core insight: supply chain risk and financial contagion are dual algebraic types.

  Financial risk:   OR  logic  — any counterparty fails → loss accumulates
  Supply chain risk: AND logic — every tier must deliver → min is the bottleneck

This corresponds exactly to the Curry-Howard correspondence:

  Sum type  (TensorSumStep) ↔ PCL parallel / choose  (additive aggregation)
  Product type (TensorMinStep) ↔ PCL fold / sequence  (conjunctive constraint)

The two primitives compose into hybrid networks that model any supply chain
topology: pure bottleneck chains (Leontief), pure market aggregation, or
mixed structures with financial risk at one layer and physical constraints
at another.

Reverse Stress Testing (RST):
  Given a network, a shock scenario, and a minimum acceptable output level,
  RST asks: "What is the minimum inventory buffer at each node that ensures
  survival?"  The answer is the argmin of a differentiable loss over buffer
  allocations — solved by JAX gradient descent.

PCL connection:
  A supply chain network IS a PCL computation tree:
    - TensorSumStep node  → parallel(flow_A, flow_B, ...)  [additive]
    - TensorMinStep node  → fold(β=∞, [tier_1, ..., tier_n])  [conjunctive]
    - Buffer injection    → flow(supplier, factory, "inventory", qty)
    - Full network        → sequence(market_layer, production_layer, delivery_layer)

The EGT (Extended Gauge Theory) interpretation: each tier is a node on
a Pacioli manifold; B_phys flows on edges; H₁ cycles are the bottleneck
loops.  The Laplacian spectrum of the adjacency graph identifies which
H₁ cycle is binding — exactly the same analysis as ModelLG.laplacian_spectrum().

References:
    Smith et al. (2025) Reverse Stress Testing for Supply Chain Resilience.
        arXiv:2511.07289
    Buckley (2026) PCL. doi:10.5281/zenodo.20262070
    differentiable-supply-chain: github.com/roguetrainer/differentiable-supply-chain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.ensemble import gibbs_weights


# ---------------------------------------------------------------------------
# Differentiable logic primitives (JAX-native, mirrors dynamics.py in PyTorch)
# ---------------------------------------------------------------------------

def soft_min(x: jax.Array, temperature: float = 0.1) -> jax.Array:
    """
    Differentiable minimum via LogSumExp trick.

    At temperature → 0: converges to true min().
    At temperature → ∞: converges to mean().

    Used by TensorMinStep (AND/Product type) to propagate bottleneck
    constraints while preserving gradient flow through all branches.
    """
    beta = 1.0 / temperature
    return -(1.0 / beta) * jax.scipy.special.logsumexp(-x * beta)


def soft_min_axis(x: jax.Array, axis: int = -1, temperature: float = 0.1) -> jax.Array:
    """Differentiable minimum along an axis."""
    beta = 1.0 / temperature
    return -(1.0 / beta) * jax.scipy.special.logsumexp(-x * beta, axis=axis)


# ---------------------------------------------------------------------------
# Algebraic data types: the two dual risk primitives
# ---------------------------------------------------------------------------

@dataclass
class SupplyCapacity:
    """
    Product-type (AND / conjunctive) risk state.

    Represents physical production capacity: a value between 0 and 1, where
    1 = full capacity, 0 = complete shutdown.

    The AND semantics: output is limited by the *least available* required input.
    This is the Leontief constraint: if any component is missing, production stops.

    Maps to PCL: fold(β→∞, [tier_1, ..., tier_n]) — winner-take-all on the
    downside (the minimum-capacity input determines output).
    """
    values: jax.Array         # shape (n_nodes,) — capacity ∈ [0, 1]
    temperature: float = 0.1

    def constrain_by(self, dependency_matrix: np.ndarray) -> "SupplyCapacity":
        """
        Apply Leontief constraint: output = SoftMin(required inputs) for each node.

        SoftMin is applied only over actual suppliers (dep[i,j]>0).
        Nodes with no dependencies are unconstrained.
        """
        dep_np = np.array(dependency_matrix)
        n = len(self.values)
        new_values = []
        for i in range(n):
            supplier_indices = np.where(dep_np[i] > 0)[0]
            if len(supplier_indices) == 0:
                new_values.append(self.values[i])
            else:
                supplier_caps = self.values[supplier_indices]
                bottleneck = soft_min_axis(supplier_caps, temperature=self.temperature)
                new_values.append(jnp.minimum(self.values[i], bottleneck))
        return SupplyCapacity(jnp.stack(new_values), self.temperature)

    def add_buffer(self, buffers: jax.Array) -> "SupplyCapacity":
        """Inject inventory buffers: capacity = min(1, capacity + buffer)."""
        return SupplyCapacity(
            jnp.clip(self.values + buffers, 0.0, 1.0),
            self.temperature,
        )

    def __and__(self, other: "SupplyCapacity") -> "SupplyCapacity":
        """AND: take element-wise SoftMin (conjunctive combination)."""
        stacked = jnp.stack([self.values, other.values], axis=0)
        return SupplyCapacity(
            soft_min_axis(stacked, axis=0, temperature=self.temperature),
            self.temperature,
        )


@dataclass
class FinancialRisk:
    """
    Sum-type (OR / disjunctive) risk state.

    Represents financial contagion exposure: a value ≥ 0, where 0 = no exposure,
    1 = full direct exposure.

    The OR semantics: risk from counterparty A plus risk from counterparty B
    accumulate — loss is the sum of weighted exposures.

    Maps to PCL: parallel(flow_A, flow_B) — additive aggregation of flows.
    """
    values: jax.Array         # shape (n_nodes,) — exposure ∈ [0, ∞)
    temperature: float = 0.1

    def propagate(self, adjacency_matrix: jax.Array) -> "FinancialRisk":
        """
        Propagate risk through adjacency weights (contagion step).

        adjacency_matrix[i, j] = weight of exposure from j to i.
        New exposure = current + sum(weights * delta_from_neighbours).
        """
        delta = self.values - jnp.ones_like(self.values)
        new_values = self.values + jnp.einsum('ij,j->i', adjacency_matrix, delta)
        return FinancialRisk(jnp.clip(new_values, 0.0, None), self.temperature)

    def __add__(self, other: "FinancialRisk") -> "FinancialRisk":
        """OR: additive combination of exposures."""
        return FinancialRisk(self.values + other.values, self.temperature)


# ---------------------------------------------------------------------------
# Network steps (mirrors TensorMinStep / TensorSumStep)
# ---------------------------------------------------------------------------

def tensor_min_step(
    capacity: jax.Array,
    dependency_matrix: jax.Array,
    temperature: float = 0.1,
) -> jax.Array:
    """
    One AND/Min propagation step through a dependency graph.

    For each node i:
      - If node i has no dependencies: output = capacity[i]  (unconstrained)
      - If node i has dependencies:    output = min(capacity[i], SoftMin(supplier capacities))

    SoftMin is applied only over the actual supplier inputs (where dep[i,j]>0),
    avoiding the bias introduced by including neutral elements in the LogSumExp.

    Args:
        capacity:          shape (n_nodes,) — current capacity ∈ [0, 1]
        dependency_matrix: shape (n_nodes, n_nodes) — Leontief BOM (numpy array)
        temperature:       SoftMin temperature

    Returns:
        new_capacity: shape (n_nodes,) — after bottleneck constraints applied
    """
    n = len(capacity)
    dep_np = np.array(dependency_matrix)
    new_values = []
    for i in range(n):
        supplier_indices = np.where(dep_np[i] > 0)[0]
        if len(supplier_indices) == 0:
            new_values.append(capacity[i])
        else:
            supplier_caps = capacity[supplier_indices]
            bottleneck = soft_min_axis(supplier_caps, temperature=temperature)
            new_values.append(jnp.minimum(capacity[i], bottleneck))
    return jnp.stack(new_values)


def tensor_sum_step(
    exposure: jax.Array,
    adjacency_matrix: jax.Array,
) -> jax.Array:
    """
    One OR/Sum propagation step through an adjacency graph.

    Args:
        exposure:         shape (n_nodes,) — current risk exposure
        adjacency_matrix: shape (n_nodes, n_nodes) — contagion weights

    Returns:
        new_exposure: shape (n_nodes,) — after additive propagation
    """
    delta = exposure - jnp.ones_like(exposure)
    return jnp.clip(exposure + jnp.einsum('ij,j->i', adjacency_matrix, delta), 0.0, None)


# ---------------------------------------------------------------------------
# Supply chain network parameters
# ---------------------------------------------------------------------------

@dataclass
class SupplyChainParameters:
    """
    Parameters for a supply chain network.

    Args:
        node_names:        list of node labels (e.g. ['Mine', 'Refinery', 'Wire', 'OEM'])
        bom_matrix:        Bill-of-Materials dependency matrix; shape (n, n).
                           bom[i, j] = 1 if node i requires input from node j.
        market_weights:    (optional) market-share aggregation matrix for Sum-type layers
        n_steps:           simulation time steps (RNN unroll depth)
        temperature:       SoftMin temperature (lower = sharper approximation)
    """
    node_names:     list[str]
    bom_matrix:     np.ndarray
    market_weights: Optional[np.ndarray] = None
    n_steps:        int = 20
    temperature:    float = 0.1


# ---------------------------------------------------------------------------
# Copper supply chain case study (from differentiable-supply-chain / Smith 2025)
# ---------------------------------------------------------------------------

COPPER_NODES = [
    "US_OEM",           # Tier 0: end user (wire & cable OEM)
    "US_Refinery",      # Tier 1: refined copper wire
    "CAN_Refinery",     # Tier 1: Canadian refinery
    "CHL_Refinery",     # Tier 1: Chilean refinery
    "CAN_Smelter",      # Tier 2: copper smelting
    "CHL_Smelter",      # Tier 2: copper smelting
    "CHL_Mine",         # Tier 3: copper ore mining
    "DRC_Mine",         # Tier 3: DRC copper mining
]

# BOM matrix: who requires whom
# bom[i, j] = 1 means node i requires output from node j
_N = len(COPPER_NODES)
_idx = {n: i for i, n in enumerate(COPPER_NODES)}

COPPER_BOM = np.zeros((_N, _N), dtype=float)
# US_OEM uses market-weighted sourcing (Sum/OR layer via market_weights — NOT BOM/AND)
# Refineries require smelted copper (AND/Product type — Leontief)
COPPER_BOM[_idx["US_Refinery"],  _idx["CAN_Smelter"]] = 1
COPPER_BOM[_idx["CAN_Refinery"], _idx["CAN_Smelter"]] = 1
COPPER_BOM[_idx["CHL_Refinery"], _idx["CHL_Smelter"]] = 1
# Smelters require ore (AND/Product type — Leontief)
COPPER_BOM[_idx["CAN_Smelter"],  _idx["CHL_Mine"]] = 1
COPPER_BOM[_idx["CHL_Smelter"],  _idx["CHL_Mine"]] = 1
COPPER_BOM[_idx["CHL_Smelter"],  _idx["DRC_Mine"]] = 1

# Market share weights for OEM sourcing (Sum-type layer)
COPPER_MARKET_WEIGHTS = np.zeros((_N, _N), dtype=float)
COPPER_MARKET_WEIGHTS[_idx["US_OEM"], _idx["US_Refinery"]]  = 0.50
COPPER_MARKET_WEIGHTS[_idx["US_OEM"], _idx["CAN_Refinery"]] = 0.30
COPPER_MARKET_WEIGHTS[_idx["US_OEM"], _idx["CHL_Refinery"]] = 0.20

COPPER_CHAIN = SupplyChainParameters(
    node_names=COPPER_NODES,
    bom_matrix=COPPER_BOM,
    market_weights=COPPER_MARKET_WEIGHTS,
    n_steps=10,
    temperature=0.1,
)


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def simulate_chain(
    params: SupplyChainParameters,
    initial_capacity: np.ndarray,
    buffers: Optional[np.ndarray] = None,
) -> dict:
    """
    Simulate supply chain capacity propagation for n_steps.

    Each step: capacity[t+1] = TensorMinStep(capacity[t] + buffers, BOM)

    Args:
        params:            SupplyChainParameters
        initial_capacity:  shape (n_nodes,) — starting capacity ∈ [0, 1]
        buffers:           shape (n_nodes,) — inventory buffers added at each
                           step (default: zeros)

    Returns:
        dict with:
          'capacity_history': shape (n_steps+1, n_nodes) — trajectory
          'final_capacity':   shape (n_nodes,) — after n_steps
          'bottleneck_nodes': list of node indices most constraining at final step
    """
    n = len(params.node_names)
    bom = jnp.array(params.bom_matrix)
    if buffers is None:
        buffers = np.zeros(n)
    buf = jnp.array(buffers)

    capacity = jnp.array(initial_capacity, dtype=float)
    history = [np.array(capacity)]
    mkt = jnp.array(params.market_weights) if params.market_weights is not None else None

    for _ in range(params.n_steps):
        buffered = jnp.clip(capacity + buf, 0.0, 1.0)
        capacity = tensor_min_step(buffered, bom, params.temperature)
        # Apply Sum/OR layer: market-weighted sourcing for nodes with market weights
        if mkt is not None:
            row_sums = mkt.sum(axis=1, keepdims=True)
            has_market = (row_sums > 0).squeeze()
            mkt_output = jnp.einsum('ij,j->i', mkt, capacity)
            # Normalise by row sum (weights may not sum to 1)
            mkt_norm = jnp.where(row_sums.squeeze() > 0, mkt_output / row_sums.squeeze(), capacity)
            capacity = jnp.where(has_market, mkt_norm, capacity)
        history.append(np.array(capacity))

    final = np.array(capacity)
    history_arr = np.stack(history, axis=0)

    # Identify bottleneck: node with lowest final capacity (excluding fully-unconstrained)
    bottleneck_nodes = list(np.argsort(final)[:3])

    return {
        'capacity_history': history_arr,
        'final_capacity': final,
        'bottleneck_nodes': bottleneck_nodes,
    }


# ---------------------------------------------------------------------------
# Reverse stress testing: find minimum buffers to survive a shock
# ---------------------------------------------------------------------------

def reverse_stress_test(
    params: SupplyChainParameters,
    shock_capacity: np.ndarray,
    required_output: float,
    output_node: int = 0,
    budget_weight: float = 0.1,
    n_epochs: int = 200,
    lr: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Reverse Stress Testing: find minimum inventory buffers that survive a shock.

    Solves:
        buffers* = argmin  Σ_i buffers_i            (minimise total inventory)
        subject to: final_capacity[output_node] ≥ required_output

    The constraint is enforced as a differentiable penalty; the whole loss is
    minimised by JAX gradient descent (Adam-like updates).

    Args:
        params:          SupplyChainParameters
        shock_capacity:  shape (n_nodes,) — capacity after the shock (before buffers)
        required_output: minimum acceptable capacity at output_node
        output_node:     index of the node whose output must be preserved
        budget_weight:   weight on buffer cost vs. survival penalty (λ)
        n_epochs:        gradient descent steps
        lr:              learning rate
        seed:            RNG seed for initialisation

    Returns:
        dict with:
          'buffers':         shape (n_nodes,) — optimal buffer allocation
          'loss_history':    shape (n_epochs,) — loss trajectory
          'final_output':    scalar — final capacity at output_node
          'survived':        bool — whether required_output was achieved
          'criticality':     shape (n_nodes,) — gradient magnitude at optimum
                             (large = node is binding constraint)
    """
    n = len(params.node_names)
    bom = jnp.array(params.bom_matrix)
    mkt = jnp.array(params.market_weights) if params.market_weights is not None else None
    shock = jnp.array(shock_capacity, dtype=float)

    rng = np.random.default_rng(seed)
    buffers = jnp.array(rng.uniform(0.0, 0.01, size=n))

    def _propagate(cap):
        for _ in range(params.n_steps):
            cap = tensor_min_step(cap, bom, params.temperature)
            if mkt is not None:
                row_sums = mkt.sum(axis=1)
                mkt_output = jnp.einsum('ij,j->i', mkt, cap)
                mkt_norm = jnp.where(row_sums > 0, mkt_output / jnp.where(row_sums > 0, row_sums, 1.0), cap)
                cap = jnp.where(row_sums > 0, mkt_norm, cap)
        return cap

    def loss_fn(buf):
        cap = jnp.clip(shock + buf, 0.0, 1.0)
        cap = _propagate(cap)
        shortfall = jax.nn.relu(required_output - cap[output_node])
        cost = jnp.sum(jnp.abs(buf))
        return shortfall * 10.0 + cost * budget_weight

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    loss_history = []
    for _ in range(n_epochs):
        loss_val, grads = grad_fn(buffers)
        loss_history.append(float(loss_val))
        buffers = jnp.clip(buffers - lr * grads, 0.0, None)

    # Criticality: |∂final_output / ∂initial_capacity_i| at optimum
    def output_fn(cap_init):
        cap = jnp.clip(cap_init + buffers, 0.0, 1.0)
        return _propagate(cap)[output_node]

    criticality = np.abs(np.array(jax.grad(output_fn)(shock)))

    cap_final = jnp.clip(shock + buffers, 0.0, 1.0)
    for _ in range(params.n_steps):
        cap_final = tensor_min_step(cap_final, bom, params.temperature)

    return {
        'buffers': np.array(buffers),
        'loss_history': np.array(loss_history),
        'final_output': float(cap_final[output_node]),
        'survived': float(cap_final[output_node]) >= required_output,
        'criticality': criticality,
    }


# ---------------------------------------------------------------------------
# Laplacian spectrum of supply chain graph (mirrors ModelLG.laplacian_spectrum)
# ---------------------------------------------------------------------------

def laplacian_spectrum(params: SupplyChainParameters) -> dict:
    """
    Compute the Laplacian spectrum of the supply chain graph.

    Uses the undirected version of the BOM graph for topological analysis.
    The spectrum identifies bottleneck cycles via H₁ topology:

      λ₂ (Fiedler value) = algebraic connectivity — how easily a shock
      at one tier propagates to another.  Small λ₂ → fragile supply chain.

    This is the same Laplacian analysis as ModelLG.laplacian_spectrum() but
    applied to the physical supply network rather than the financial circuit.

    Returns:
        dict with 'eigenvalues', 'fiedler_value', 'fiedler_vector', 'n_components'
    """
    adj = (params.bom_matrix + params.bom_matrix.T)
    adj = (adj > 0).astype(float)
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    n_components = int(np.sum(eigvals < 1e-8))
    fiedler_value = float(eigvals[n_components]) if n_components < len(eigvals) else 0.0

    return {
        'eigenvalues': eigvals,
        'fiedler_value': fiedler_value,
        'fiedler_vector': eigvecs[:, n_components],
        'n_components': n_components,
    }


# ---------------------------------------------------------------------------
# Scenario shocks
# ---------------------------------------------------------------------------

def apply_shock(
    params: SupplyChainParameters,
    baseline_capacity: Optional[np.ndarray] = None,
    shocked_nodes: Optional[list[str]] = None,
    shock_fraction: float = 0.5,
) -> np.ndarray:
    """
    Apply a capacity shock to named nodes.

    Args:
        params:              SupplyChainParameters
        baseline_capacity:   shape (n_nodes,) — defaults to all-ones (full capacity)
        shocked_nodes:       list of node names to shock
        shock_fraction:      remaining capacity after shock (0 = shutdown, 0.5 = half capacity)

    Returns:
        shocked_capacity: shape (n_nodes,)
    """
    n = len(params.node_names)
    if baseline_capacity is None:
        capacity = np.ones(n)
    else:
        capacity = np.array(baseline_capacity, dtype=float)

    if shocked_nodes is None:
        return capacity

    node_idx = {name: i for i, name in enumerate(params.node_names)}
    for name in shocked_nodes:
        if name not in node_idx:
            raise ValueError(f"Node '{name}' not in {params.node_names}")
        capacity[node_idx[name]] = shock_fraction

    return capacity


# ---------------------------------------------------------------------------
# PCL connection: build a PCL Computation tree from a supply chain
# ---------------------------------------------------------------------------

def to_pcl_description(params: SupplyChainParameters) -> str:
    """
    Return a human-readable description of the PCL combinator structure
    corresponding to this supply chain network.

    Each supply chain topology maps to a PCL expression:
      - Pure bottleneck chain: sequence(tier_1_step, tier_2_step, ...)
      - Market aggregation:    parallel(flow_supplier_A, flow_supplier_B, ...)
      - Mixed:                 sequence(parallel(...), fold(...))

    This is diagnostic — it shows the algebraic type of each tier.
    """
    lines = [
        f"Supply chain '{params.node_names[0]} ← ... ← {params.node_names[-1]}'",
        f"  Nodes: {len(params.node_names)}",
        "",
        "  PCL combinator structure:",
        "  ┌─────────────────────────────────────────────────────────────┐",
    ]

    bom = params.bom_matrix
    for i, name in enumerate(params.node_names):
        suppliers = [params.node_names[j] for j in range(len(params.node_names))
                     if bom[i, j] > 0]
        if not suppliers:
            lines.append(f"  │  {name:20s}  [leaf — no dependencies]")
        elif len(suppliers) == 1:
            lines.append(f"  │  {name:20s}  ← sequence({suppliers[0]})")
        else:
            sup_str = ", ".join(suppliers)
            if params.market_weights is not None and params.market_weights[i].sum() > 0:
                lines.append(f"  │  {name:20s}  ← parallel({sup_str})  [SumType / OR]")
            else:
                lines.append(f"  │  {name:20s}  ← fold(β→∞, [{sup_str}])  [ProductType / AND]")

    lines += [
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  Algebraic types:",
        "    Product / AND → fold(β→∞, [...])  — bottleneck: min of inputs",
        "    Sum / OR      → parallel(...)      — contagion: sum of flows",
    ]
    return "\n".join(lines)
