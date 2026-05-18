"""
Tensor-based differentiable agent-based macroeconomics (DABM).

Agents are indices, not objects. The entire population state is a tensor.
One timestep = one forward pass of the antisymmetric wealth-update layer.
No for loops. No object instantiation. Fully differentiable via JAX/PyTorch.

The ABM is a Graph Neural Network: Pacioli manifold = adjacency matrix,
agent state = node features, interaction = message passing.

Reference: Buckley (2026) Differentiable ABM, doi:10.5281/zenodo.20261945
"""

# TODO Phase 3: implement
# - AgentPopulation(n_agents, state_dim) — vectorised agent state tensor
# - WealthUpdateLayer(adjacency, beta) — antisymmetric MGE-weighted update
# - DABMSimulator(population, manifold, beta_schedule)
#     .step(state) -> next_state  [single forward pass, differentiable]
#     .simulate(T) -> trajectory
#     .calibrate(data, loss) -> parameters  [backprop through economy]
