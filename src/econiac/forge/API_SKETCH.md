# econiac.forge — API Sketch

**Status:** Design only. Implementation follows Paper 425.

econiac.forge is the financial layer of the Forge ISA —
the same five opcodes as thermion.forge but with financial stalks:

    thermion.forge stalks:  simplicial cochains (abstract graph)
    econiac.forge stalks:   balance sheet positions, exposure amounts

---

## econiac.forge.schedule

```python
from econiac.forge.schedule import beta_star, BetaSchedule

# β*(ρ) from an exposure network
# network: nx.DiGraph with edge weights = exposure amounts
beta = beta_star(network)
# Returns: float — the systemic fragility threshold
# Interpretation:
#   Low β*  → rigid network (few cycles, β→∞ quickly)
#   High β* → fragile network (many funding loops, slow to freeze)

# Annealing schedule
schedule = BetaSchedule(network, n_steps=500, mode='linear_to_critical')
```

**beta_star implementation:**
```python
def beta_star(network) -> float:
    """
    Systemic fragility threshold for a financial exposure network.
    β*(ρ) = (3/8) * log(1 / (1 - ρ))
    ρ = β₁/|E| = (|E| - |V| + components) / |E|

    β₁ = number of independent funding cycles in the network.
    High β₁ → high ρ → high β* → network is near systemic fragility.
    """
    n_v = network.number_of_nodes()
    n_e = network.number_of_edges()
    n_comp = nx.number_connected_components(network.to_undirected())
    beta_1 = n_e - n_v + n_comp
    rho = beta_1 / max(n_e, 1)
    if rho <= 0: return 0.0
    if rho >= 1: return float('inf')
    return (3/8) * math.log(1 / (1 - rho))
```

---

## econiac.forge.opcodes

### SPLIT_β — soft fund flow

```python
from econiac.forge import SPLIT

# Soft allocation of funds from one source to multiple recipients
split = SPLIT(
    allocation_weights=torch.tensor(weights),  # priority/preference weights
    beta=beta
)

# Forward: source_amount (scalar) → recipient_amounts (vector)
# At β→∞: all funds go to highest-priority recipient (hard priority)
# At β*(ρ): funds distributed according to Gibbs-weighted priorities
# Gradient: ∂amounts/∂weights (sensitivity to priority changes)
amounts = split(source_amount)
```

### SPLAT_β — soft netting

```python
from econiac.forge import SPLAT

# Soft multilateral netting of bilateral exposures
splat = SPLAT(
    netting_matrix=torch.tensor(N),  # netting eligibility matrix (n×n)
    beta=beta
)

# Forward: gross_exposures (n×n) → net_exposure (scalar or vector)
# At β→∞: exact netting (hard in-scope/out-of-scope)
# At β*(ρ): soft netting (Gibbs-weighted eligibility)
# Gradient: ∂net/∂gross (sensitivity to gross exposure changes)
net = splat(gross_exposures)
```

### FLOP_β — soft restructuring / H¹ correction

```python
from econiac.forge import FLOP

# Soft resolution of funding gaps (H¹ obstructions)
flop = FLOP(
    laplacian=network_laplacian,  # L = B1.T @ B1 + B1 @ B1.T
    beta=beta
)

# Forward: funding_gap (vector) → restructured_amounts (vector)
# = direction of minimal-cost restructuring to fill the gap
# At β→∞: exact Green's operator (optimal restructuring)
# At β*(ρ): β-regularised solve (gradient-guided restructuring)
# Gradient: ∂restructuring/∂gap (sensitivity to funding gap size)
restructuring = flop(funding_gap)

# Key application: XVA
# funding_gap = CVA + DVA + FVA residual
# restructuring = hedge ratios that minimise XVA
xva_hedge = flop(xva_residual)
xva_greeks = torch.autograd.grad(xva_residual.sum(), positions)[0]
```

### FLIP_β — soft asset/liability duality

```python
from econiac.forge import FLIP

# Soft duality between asset and liability frames
flip = FLIP(
    volume_weights=torch.tensor(volumes),  # balance sheet item sizes
    beta=beta
)

# Forward: assets (vector) → liabilities (vector) in dual frame
# Gradient: ∂liabilities/∂assets (sensitivity to reclassification)
# Key application: HTM → AFS reclassification impact
liabilities_view = flip(assets)
```

### TWIST_β — soft currency/numéraire change

```python
from econiac.forge import TWIST

# Soft FX conversion with gradient through the rate
twist = TWIST(
    fx_rate=torch.tensor(rate, requires_grad=True),
    beta=beta
)

# Forward: amount_ccy1 → amount_ccy2
# At β→∞: hard conversion at the rate
# Gradient: ∂converted/∂rate (FX delta) — automatic!
converted = twist(amount_in_ccy1)
fx_delta = torch.autograd.grad(converted.sum(), fx_rate)[0]
```

---

## econiac.forge.programme

```python
from econiac.forge import ForgeProgram, beta_star

# Build the financial network sheaf
network = build_exposure_graph(institutions, exposures)
beta = beta_star(network)

# Construct a differentiable stress test
programme = ForgeProgram(beta=beta)
programme.add_split(allocation_weights)   # fund flow
programme.add_splat(netting_matrix)       # netting
programme.add_flop(network_laplacian)     # restructuring

# Run stress test with full sensitivity output
positions = torch.tensor(balance_sheet, dtype=torch.float64,
                          requires_grad=True)
optimizer = torch.optim.LBFGS([positions])

def closure():
    optimizer.zero_grad()
    pnl = programme(positions)
    loss = -pnl.sum()
    loss.backward()
    return loss

optimizer.step(closure)

# Sensitivities = positions.grad
# ∂P&L/∂(each balance sheet position) computed in one backward pass
```

---

## econiac.forge.network — financial sheaf construction

```python
from econiac.forge.network import (
    exposure_graph,      # build nx.DiGraph from institution/exposure data
    financial_sheaf,     # build coboundary matrices B1, B2
    h1_betti,            # compute β₁ = |E| - |V| + components
    systemic_fragility,  # compute β*(ρ) with interpretation
)

# Build from raw data
network = exposure_graph(
    institutions=['BankA', 'BankB', 'BankC', 'CCP'],
    exposures=[
        ('BankA', 'BankB', 100e6),   # £100M exposure A→B
        ('BankB', 'BankC',  50e6),
        ('BankA', 'CCP',   200e6),
        ('BankC', 'CCP',    75e6),
    ]
)

# Sheaf cohomology
sheaf = financial_sheaf(network)
b1 = h1_betti(sheaf)   # number of independent funding cycles

# Systemic risk pre-diagnostic
result = systemic_fragility(network)
# Returns: {
#   'beta_1': 2,           # 2 independent funding cycles
#   'rho': 0.5,            # 50% of edges in cycles
#   'beta_star': 0.260,    # critical temperature
#   'interpretation': 'moderate fragility — 2 funding loops at risk',
#   'regime': 'H1'         # H¹ regime (FLOP can resolve)
# }
```

---

## Module structure

```
econiac/src/econiac/
  forge/
    __init__.py       ← this file (module docstring + imports)
    opcodes.py        ← SPLIT, SPLAT, FLOP, FLIP, TWIST (PyTorch modules)
    schedule.py       ← beta_star(), BetaSchedule
    programme.py      ← ForgeProgram (composable differentiable pipeline)
    network.py        ← exposure_graph(), financial_sheaf(), h1_betti()
    API_SKETCH.md     ← this file
```

---

## Integration with existing econiac modules

```python
# finance.contagion + forge
from econiac.finance.contagion import ContagionNetwork
from econiac.forge import beta_star, FLOP

net = ContagionNetwork.from_data(exposure_data)
beta = beta_star(net.graph)
flop = FLOP(net.laplacian, beta=beta)
# flop now gives differentiable contagion propagation

# economics.sfc + forge
from econiac.economics.sfc import SFCModel
from econiac.forge import ForgeProgram

model = SFCModel.load('uk_macro.yaml')
programme = ForgeProgram.from_sfc(model, beta=beta_star(model.transaction_graph))
# programme is now a differentiable macro model
# ∂GDP/∂(tax_rate) computable by autodiff

# finance.credit + forge
from econiac.finance.credit import CreditPortfolio
from econiac.forge import FLOP

portfolio = CreditPortfolio.from_data(loan_data)
flop = FLOP(portfolio.exposure_laplacian, beta=beta_star(portfolio.network))
xva = flop(portfolio.xva_residual)
xva_greeks = torch.autograd.grad(xva.sum(), portfolio.spreads)[0]
```

---

## Dependencies

econiac.forge requires:
  - torch >= 2.0      (autodiff through opcodes)
  - networkx >= 3.0   (exposure graph construction)
  - numpy             (array operations)

Optional:
  - jax               (JAX-compatible versions)
  - scipy.sparse      (for large networks, sparse Laplacian)

econiac.core and econiac.finance are not required by forge —
forge is a standalone differentiable layer that wraps them.
