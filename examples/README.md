# EconIAC Examples

Runnable Python scripts demonstrating the econiac library.
Each script is self-contained and prints results to stdout.

## Running

```bash
pip install econiac jax[cpu] matplotlib
python examples/keen_predator_prey.py
python examples/gemmes.py
python examples/triangular_arbitrage.py
```

## Scripts

| Script | What it demonstrates |
| --- | --- |
| [keen_predator_prey.py](keen_predator_prey.py) | Full econiac stack on the Keen (1995) debt-dynamics model: ODE simulation, PCL `choose` Minsky transistor, TIR routing, thermal Shapley attribution, `conservation_loss` calibration |
| [gemmes.py](gemmes.py) | GEMMES (Bovari et al. 2018): Keen + carbon cycle + Nordhaus climate damage. `CurvedBalanceSheet` stranded assets, PCL `fold` green transition, 4-player Shapley |
| [triangular_arbitrage.py](triangular_arbitrage.py) | FX triangular arbitrage as non-zero holonomy on the currency bundle (stub — full implementation coming with `econiac.finance.fx`) |

## Annotated notebooks

The `docs/tutorials/` folder contains Jupyter notebook versions of the
Keen and GEMMES examples with prose explanations, equations, and plots.

To run a notebook locally:

```bash
pip install "econiac[tutorials]"
jupyter notebook docs/tutorials/keen_predator_prey.ipynb
```

Or open in [Google Colab](https://colab.research.google.com) — uncomment
the `!pip install` cell at the top of each notebook.
