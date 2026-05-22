<!-- markdownlint-disable MD033 -->
# EconIAC

**Differentiable economics on the Pacioli manifold.**

EconIAC is a Python library for building differentiable macroeconomic and financial models grounded in gauge theory, thermodynamics, and double-entry bookkeeping. Named after [MONIAC](https://en.wikipedia.org/wiki/MONIAC) (1949), Bill Phillips's hydraulic computer — EconIAC is MONIAC for the 21st century. [Learn more →](about.md)

<div style="display:flex; gap:0.75rem; margin: 1.5rem 0;">
  <a href="examples/" style="display:inline-block; padding:0.5rem 1.2rem; background:#3f51b5; color:#fff; border-radius:6px; text-decoration:none; font-weight:600; font-size:0.9rem;">Examples</a>
  <a href="tutorials/" style="display:inline-block; padding:0.5rem 1.2rem; background:#fff; color:#3f51b5; border:1.5px solid #3f51b5; border-radius:6px; text-decoration:none; font-weight:600; font-size:0.9rem;">Tutorials</a>
  <a href="https://github.com/roguetrainer/econiac" target="_blank" style="display:inline-block; padding:0.5rem 1.2rem; background:#24292e; color:#fff; border-radius:6px; text-decoration:none; font-weight:600; font-size:0.9rem;">★ Star on GitHub</a>
</div>

```bash
pip install econiac
```

---

## What EconIAC does

<div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:1rem; margin:1.5rem 0;">

<div style="border:1px solid #ddd; border-radius:8px; padding:1rem;">
<strong>Differentiable models</strong><br>
<span style="font-size:0.85rem; color:#555;">Every threshold, choice, and aggregation is a smooth Gibbs relaxation — end-to-end JAX/PyTorch gradients, calibratable by gradient descent.</span>
</div>

<div style="border:1px solid #ddd; border-radius:8px; padding:1rem;">
<strong>Stock-flow consistency</strong><br>
<span style="font-size:0.85rem; color:#555;">Double-entry accounting as a discrete gauge theory. Conservation is enforced algebraically, not by convention.</span>
</div>

<div style="border:1px solid #ddd; border-radius:8px; padding:1rem;">
<strong>Thermal attribution</strong><br>
<span style="font-size:0.85rem; color:#555;">Differentiable Shapley values via the Gibbs ensemble. Attribute systemic risk, carbon tax burden, or supply-chain criticality in one backward pass.</span>
</div>

<div style="border:1px solid #ddd; border-radius:8px; padding:1rem;">
<strong>Financial gauge theory</strong><br>
<span style="font-size:0.85rem; color:#555;">FX, yield curves, and credit spreads as parallel transport on the Pacioli manifold. Triangular arbitrage = non-zero holonomy.</span>
</div>

<div style="border:1px solid #ddd; border-radius:8px; padding:1rem;">
<strong>Reverse stress testing</strong><br>
<span style="font-size:0.85rem; color:#555;">Find the minimum-cost intervention that keeps a supply chain, coalition, or portfolio above a survival threshold — via differentiable optimisation.</span>
</div>

<div style="border:1px solid #ddd; border-radius:8px; padding:1rem;">
<strong>Pacioli Combinator Library</strong><br>
<span style="font-size:0.85rem; color:#555;">A typed DSL for financial flows. <code>flow</code>, <code>sequence</code>, <code>choose</code>, <code>fold</code> — every combinator preserves conservation by construction.</span>
</div>

</div>

---

## What EconIAC adds over mainstream simulation frameworks

Mainstream system dynamics, digital twin, and agent-based modelling frameworks
(Stella, Vensim, AnyLogic, NetLogo) are excellent for building and communicating models.
EconIAC is designed for what comes next: calibration, differentiation, and stress-testing.

| Capability | Mainstream frameworks | EconIAC |
| --- | --- | --- |
| **Exact policy gradients** | Manual parameter sweeping | `jax.grad` — one backward pass |
| **Calibration** | Manual dial-turning | Gradient descent on `calibrate_beta(data)` |
| **Reverse stress testing** | Not supported | Differentiable optimisation over survival threshold |
| **Conservation enforcement** | Modeller discipline | Algebraic — invalid transactions are type errors |
| **Second-order sensitivities** | Not supported | `jax.hessian` — exact cross-gamma in one call |
| **Tipping point early-warning** | Simulate through bifurcation | χ(β) computable before the bifurcation arrives |
| **Differentiable ABMs** | Hard IF/THEN thresholds | Smooth Gibbs relaxations, end-to-end differentiable |
| **GPU/TPU acceleration** | Limited or none | Native via JAX |

EconIAC can import Stella/Vensim models via the `pysd` backend — use the visual
modelling tools you already have, then bring the model into EconIAC to differentiate and calibrate it.

---

## Quick start

```python
from econiac.pcl import flow, sequence, choose, compile, typecheck

# Three sectors, one instrument
wages    = flow("firms", "households", "deposits", 1000.0)
taxes    = flow("households", "government", "deposits", 200.0)
reinvest = flow("households", "firms", "deposits", 500.0)
save     = flow("households", "banks", "deposits", 300.0)

# β=2: lean toward the higher-value strategy, but hedge
quarterly = sequence(wages, sequence(taxes, choose(2.0, reinvest, save)))

assert typecheck(quarterly)
fast = compile(quarterly)
```

Or run the supply-chain reverse stress test:

```python
from econiac.economics.supply_chain import COPPER_CHAIN, reverse_stress_test

result = reverse_stress_test(COPPER_CHAIN, threshold=0.85, beta=3.2)
print(result.criticality_vector)   # which suppliers to buffer first
print(result.min_cost_buffers)     # minimum buffer allocation
```

---

## The core idea: rationality is temperature

Standard economic models treat agents as perfectly rational ($\beta \to \infty$).
EconIAC treats rationality as a temperature parameter $\beta$ — the inverse of decision noise.

$$Z_\beta = \frac{1}{\beta} \ln \sum_i e^{\beta U_i}$$

This single substitution turns every argmax into a differentiable softmax, every
threshold into a smooth sigmoid, every Leontief minimum into a SoftMin. The result
is an end-to-end differentiable model calibratable from data.

At $\beta \to \infty$ you recover the classical model exactly — Nash equilibria,
Leontief multipliers, Shapley values. At finite $\beta^*$ (calibrated from observed
choice variance) you get policy gradients, reverse stress tests, and early-warning
signals for tipping points.

See [Why EconIAC?](why/README.md) for motivation from economic first principles.

---

## Papers

EconIAC is the software companion to the [Portfolio G](https://roguetrainer.github.io/adelic-simplicial-architecture/portfolios/portfolio-g/) papers of the Adelic Simplicial Architecture (ASA). Key papers:

| Paper | What it grounds |
| --- | --- |
| [289 — The Temperature of Rationality](https://doi.org/10.5281/zenodo.20234841) | Gibbs ensemble as economic foundation |
| [291 — The Topology of Conservation](https://doi.org/10.5281/zenodo.20234853) | Double-entry as discrete gauge theory |
| [293 — Thermal Attribution](https://doi.org/10.5281/zenodo.20236870) | Differentiable Shapley values |
| [294 — Thermodynamic Information Routing](https://doi.org/10.5281/zenodo.20237288) | TIR unified framework |
| [295 — Currency Bundles](https://doi.org/10.5281/zenodo.20242355) | FX as connection curvature |
| [298 — Credit Bundles](https://doi.org/10.5281/zenodo.20257596) | Survival probabilities as parallel transport |
| [311 — The Climate Hazard Yield Surface](https://doi.org/10.5281/zenodo.20291646) | Climate investment geometry |

---

## Modules

| Module | What it does |
| --- | --- |
| `econiac.core` | `BalanceSheet`, Gibbs weights, manifold geometry |
| `econiac.routing` | TIR routing, thermal Shapley attribution |
| `econiac.pcl` | Pacioli Combinator Library — conservation-enforcing DSL |
| `econiac.economics` | Macro models: Keen, GEMMES, LowGrow, supply-chain RST, climate yield |
| `econiac.finance` | FX, yield curves, credit spreads, XVA — gauge-theoretic finance |
