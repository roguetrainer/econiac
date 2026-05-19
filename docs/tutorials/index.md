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
| [Keen predator-prey](keen_predator_prey.ipynb) | ODE simulation, BalanceSheet, PCL `choose`, TIR routing, thermal Shapley, `conservation_loss` | Economists, macro modellers |
| [GEMMES: Keen + climate](gemmes.ipynb) | `CurvedBalanceSheet`, stranded assets, PCL `fold`, 4-player Shapley, carbon tax routing | Climate economists, central bankers |

---

## Why notebooks, not scripts?

The `.py` scripts in `examples/` are the same computations run straight through.
The notebooks add:
- **Prose** explaining the economic interpretation of each step
- **Plots** showing trajectories, phase portraits, routing weights, and attribution bars
- **Colab compatibility** — one cell to install, the rest runs in the browser

The `examples/` scripts are the canonical reference; the notebooks are the tutorial layer.
