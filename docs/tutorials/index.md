# Tutorials

Step-by-step notebooks that build models with EconIAC from scratch.
Each tutorial starts from a concrete economic problem and introduces
library concepts as they are needed.

To run locally:

```bash
pip install "econiac[tutorials]"
jupyter notebook docs/tutorials/keen_predator_prey.ipynb
```

To run in Colab: click the notebook link, then uncomment the
`!pip install econiac jax[cpu]` cell at the top.

---

## Available tutorials

| Notebook | Concepts introduced | Audience |
| --- | --- | --- |
| [MONIAC — the hydraulic economy, differentiable](moniac.ipynb) | Phillips (1950) ODE, PCL conservation, Gibbs slot-cam, `jax.grad` fiscal multiplier, accelerator bifurcation, susceptibility χ early-warning | Everyone — start here |
| [Budget simplex and the demand law](budget_simplex_becker.ipynb) | Budget simplex geometry, `gibbs_weights` β-sweep, β* calibration from choice variance, Smith information transfer index = log Z_β | Economists, theorists |
| [Keen predator-prey](keen_predator_prey.ipynb) | ODE simulation, BalanceSheet, PCL `choose`, TIR routing, thermal Shapley, `conservation_loss` | Economists, macro modellers |
| [GEMMES: Keen + climate](gemmes.ipynb) | `CurvedBalanceSheet`, stranded assets, PCL `fold`, 4-player Shapley, carbon tax routing | Climate economists, central bankers |
| [GL Model PC: portfolio choice](gl_pc.ipynb) | `gibbs_weights`, TIR portfolio allocation, `calibrate_beta`, β* from Flow of Funds data, phase boundary at β=0 | Monetary economists, SFC modellers |
| [LowGrow SSE: green transition](lowgrow.ipynb) | `CurvedBalanceSheet`, climate damage curvature, TIR investment routing, `calibrate_green_beta`, carbon tax phase diagram, carbon lock-in paradox | Climate economists, energy modellers |
| [Cross-currency swap: gauge theory](cross_currency_swap.ipynb) | `FXMarket`, `YieldCurve`, CIP holonomy, `jax.grad` Greeks, cross-gamma Hessian, 3-currency curvature | Quantitative analysts, risk managers, derivatives practitioners |
| [Supply chain RST: differentiable ADTs](supply_chain.ipynb) | `SupplyCapacity` (AND/Product type), `FinancialRisk` (OR/Sum type), `reverse_stress_test`, `laplacian_spectrum`, PCL combinator duality, Curry-Howard correspondence | Supply chain analysts, risk managers |
| [Fraud detection via topology](fraud_detection.ipynb) | `BalanceSheet`, `Connection`, `wilson_loop`, `curvature_matrix`, PCL type-safety, `jax.grad` equity sensitivities, Wilson loop geometric audit | Accountants, auditors, financial regulators |

---

## Why notebooks, not scripts?

The `.py` scripts in `examples/` are the same computations run straight through.
The notebooks add:

- **Prose** explaining the economic interpretation of each step
- **Plots** showing trajectories, phase portraits, routing weights, and attribution bars
- **Colab compatibility** — one cell to install, the rest runs in the browser

The `examples/` scripts are the canonical reference; the notebooks are the tutorial layer.
