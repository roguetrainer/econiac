# EconIAC for RegTech and Model Vendors

The bilateral/triangular/systemic risk framework is a **new product category**:
cohomological risk infrastructure that no existing vendor provides.

---

## The gap in the market

Current risk infrastructure is built around $H^0$ and $H^1$ computations:

- **Market risk systems** (Murex, Calypso, Finastra): compute bilateral
  sensitivities ($H^0$) and some triangular Greeks ($H^1$, via Monte Carlo)
- **XVA engines** (Quaternion, Acadia, OpenGamma): compute CVA/DVA/FVA/MVA
  as $H^1$ classes, typically via parametric models (Gaussian copula, SABR)
- **Systemic risk platforms** (network analysis, DebtRank, CoVaR): compute
  $H^0$ bilateral propagation or $H^1$ conditional risk; none compute $H^2$

**$H^2$ — systemic risk as a topological invariant — is unaddressed by any
existing vendor.** The computation is straightforward linear algebra on the
financial interaction diagram. The data (trade repository, CCP exposure data,
correlation instrument prices) is available. The market does not yet have a
product that does it.

---

## What the EconIAC platform provides

### The computational primitives

EconIAC implements the Origami ISA — five operators (SPLIT, SPLAT, FLIP, FLOP,
TWIST) that correspond exactly to Čech cohomology operations on the pricing sheaf:

| Opcode | Cohomology operation | Financial meaning |
| --- | --- | --- |
| `SPLIT` | Coboundary $\delta^0: H^0 \to H^1$ | Bilateral price → triangular obstruction |
| `SPLAT` | Integration $\int_\text{fibre}: H^1 \to H^0$ | Triangular risk → price |
| `TWIST` | Gauge transformation on $H^1$ | Numeraire change, measure change |
| `FLIP` | Sheaf dualisation $H^0 \to (H^0)^\vee$ | Time reversal, ket → bra |
| `FLOP` | Trace $(H^0)^\vee \otimes H^0 \to \mathbf{1}$ | Probability rule, discounting |

These are basis-independent, model-free operations. They compute the exact
$H^1$ and $H^2$ classes of any financial interaction diagram given market-observable
input data.

### The differentiable contagion engine

End-to-end differentiable models for all five Hurd contagion channels:

| Channel | EconIAC module | Policy gradient |
| --- | --- | --- |
| Solvency cascade | `finance.contagion.solvency` | $\partial\text{loss}/\partial\text{capital\_ratio}$ |
| Liquidity withdrawal | `finance.contagion.liquidity` | $\partial\text{loss}/\partial\text{coverage}$ |
| Fire sales | `finance.contagion.fire_sales` | $\partial\text{loss}/\partial\text{haircut}$ |
| Bank panic | `finance.contagion.panic` | $\partial\text{loss}/\partial\text{threshold}$ |
| Rehypothecation | `finance.contagion.rehyp` | $\partial\text{loss}/\partial\text{chain\_length}$ |

All five channels share the same sheaf structure; the $H^1$ early-warning
signal is computed identically across all five.

### The $H^2$ stability monitor

A real-time $H^2$ computation pipeline:

```python
from econiac.finance.cohomology import SystemInteractionDiagram, h2_stability

diagram = SystemInteractionDiagram.from_trade_repository(emir_data)
diagram.add_correlation_instruments(cdx_tranches, correlation_swaps)

result = h2_stability(diagram)
# result.h2_trivial: bool — system stable?
# result.critical_triples: list — which institution triples are inconsistent
# result.gradient: dict — ∂H²/∂exposure for each institution pair
```

---

## Product opportunities

### Cohomological stress testing as a service

Regulators need $H^2$ computation but do not have the infrastructure.
A RegTech vendor offering cohomological stress testing as a managed service —
ingesting EMIR trade repository data, computing $H^1$ and $H^2$, reporting
stability indicators — addresses a clear regulatory gap.

**Data pipeline:** EMIR → bilateral exposure matrix → $H^1$ triangles →
$H^2$ stability class → regulatory report.

### Model-free XVA wrong-way risk

The $H^2$ component of XVA (wrong-way risk) currently requires parametric
models that systematically misspecify correlation structure. A model-free
$H^2$-based wrong-way risk calculator — using CCP exposure data and
correlation instrument prices — would be a commercially differentiated
product for XVA desks at major dealers.

### Topological SIFI analytics

The FSB's size-based SIFI designation is widely recognised as incomplete.
A topological SIFI analytics product — computing each institution's $H^2$
contribution to the system and identifying critical network nodes — addresses
a long-standing regulatory need without requiring legislative change (the
$H^2$ contribution can be reported as a supplementary indicator alongside
existing FSB metrics).

---

## Integration

EconIAC is a Python library with:

- **JAX/PyTorch backends** for GPU-accelerated computation
- **`pysd` import** for existing Stella/Vensim models
- **Standard financial data connectors** (Bloomberg, Reuters, EMIR APIs)
- **RESTful API** for integration with existing risk infrastructure

```bash
pip install econiac
```

---

## Papers

| Paper | Content |
| --- | --- |
| [396 — The Unhedgeability Theorem](https://doi.org/10.5281/zenodo.20635479) | Mathematical foundation; unhedgeability theorem; Origami ISA as Čech cohomology |
| [397 — Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) | $H^2$ stress test; SIFI theorem; implementation path |
| [303 — Pacioli Combinator Library](https://doi.org/10.5281/zenodo.20262070) | Conservation-enforcing DSL; typed financial computation |
| [316 — EconIAC/MONIAC](https://doi.org/10.5281/zenodo.20315689) | Platform architecture; differentiable ABM foundation |
