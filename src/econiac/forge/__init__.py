"""
econiac.forge
=============
β-parameterised, autodiff-compatible financial opcodes.

The Forge ISA (Paper 419) and the Meld (Paper 417) applied to
economic and financial systems: discrete rules (balance sheets,
credit constraints, regulatory triggers, netting agreements)
thawed into differentiable soft versions at temperature β*(ρ).

At finite β = β*(ρ), the economy becomes differentiable end-to-end.
One backward pass gives all sensitivities (Greeks, risk factor
loadings, policy multipliers) simultaneously.

β*(ρ) = (3/8) * ln(1 / (1 - ρ))
where ρ = β₁/|E| = (|E| - |V| + components) / |E|
is the load factor of the financial network's H¹ sheaf.
Computable in O(|institutions| + |exposures|) via union-find.

The five soft financial opcodes
--------------------------------
    SPLIT_β  — soft fund flow (one source → many recipients)
    SPLAT_β  — soft netting (many exposures → one net position)
    FLOP_β   — soft restructuring (H¹ correction for funding gaps)
    FLIP_β   — soft asset/liability duality (Hodge star on balance sheet)
    TWIST_β  — soft currency/numéraire change (FX conversion with gradient)

Mirror of thermion.forge but with financial stalks:
    thermion.forge: simplicial cochains, graph topology
    econiac.forge:  balance sheet positions, exposure networks

Implemented in: Paper 425 (The Differentiable Economy).
See forge/API_SKETCH.md for full API design.

Usage::

    from econiac.forge import SPLIT, SPLAT, FLOP, beta_star, ForgeProgram
    from econiac.forge.schedule import BetaSchedule

    # Build the exposure network
    network = exposure_graph(institutions, bilateral_exposures)
    beta = beta_star(network)  # systemic fragility threshold

    # Differentiable balance sheet
    programme = ForgeProgram(beta=beta)
    programme.add(SPLIT(allocation_weights, beta=beta))
    programme.add(SPLAT(netting_matrix, beta=beta))
    programme.add(FLOP(laplacian, beta=beta))

    # Forward + backward: all sensitivities in one pass
    positions = torch.tensor(balance_sheet, requires_grad=True)
    pnl = programme(positions)
    pnl.backward()
    sensitivities = positions.grad  # ∂P&L/∂(all positions) at once
"""

# Imports populated as modules are implemented (Paper 425)
# from .schedule import beta_star, BetaSchedule
# from .opcodes import SPLIT, SPLAT, FLOP, FLIP, TWIST
# from .programme import ForgeProgram
# from .network import exposure_graph, financial_sheaf
