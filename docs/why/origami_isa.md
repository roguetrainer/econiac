# The Origami ISA: EconIAC's Computational Engine

EconIAC's cohomological risk computations — bilateral, triangular, systemic —
are not implemented as ad-hoc algorithms. They are instances of a single
five-opcode instruction set called the **Origami ISA**, whose opcodes correspond
exactly to the operations of Čech cohomology on a sheaf.

This page explains the connection. No prior knowledge of the ISA or of cohomology
is required; the [bilateral · triangular · systemic](cohomology.md) page covers
the financial meaning, and the
[primer](https://doi.org/10.5281/zenodo.20642983) develops it from scratch.

---

## The five opcodes and their cohomological meaning

The Origami ISA has five primitive operations. Each has a precise mathematical
identity and a concrete financial implementation in EconIAC.

| Opcode | Move | Cohomology operation | Financial meaning | EconIAC |
| --- | --- | --- | --- | --- |
| `SPLIT` | $1 \to 4$ Pachner | $\delta^0: H^0 \to C^1$ — coboundary | Bilateral price → triangular obstruction (the $H^1$ class) | `finance.cohomology.split()` |
| `SPLAT` | $4 \to 1$ Pachner | $\int_\text{fibre}: C^1 \to H^0$ — integration over fibre | Triangular risk → price (conditional expectation, SPLAT = pricing map) | `finance.cohomology.splat()` |
| `TWIST` | — | Gauge transformation on $H^1$ | Numeraire change; measure change; change of collateral posting convention | `finance.cohomology.twist()` |
| `FLIP` | $1 \to 3$ | Sheaf dualisation $\mathcal{F} \to \mathcal{F}^\vee$ | Time reversal; ket → bra; asset → liability in double-entry | `finance.cohomology.flip()` |
| `FLOP` | $3 \to 1$ | Trace $\mathcal{F}^\vee \otimes \mathcal{F} \to \mathbf{1}$ | Discounting; probability rule; taking expectation | `finance.cohomology.flop()` |

The **Pentagon identity** — the consistency condition that governs the ISA — is
the statement $d^2 = 0$ for the Čech complex: the coboundary of a coboundary
is zero. In finance this is:

- The **HJM no-arbitrage condition** (discount factors compose consistently)
- The **no-static-arbitrage condition** on the volatility surface
- The **tower property** of conditional expectations (martingale pricing)
- The **$H^2 = 0$ stability condition** (no systemic cascade)

All four are the same equation. The ISA enforces it by construction.

---

## Why this matters for EconIAC

### Model-free by construction

Because the ISA opcodes are basis-independent Čech operations, EconIAC's
cohomological risk computations are **model-free**: they derive from the
topology of the financial interaction diagram and the prices of observable
instruments, not from parametric assumptions about distributions or dynamics.

Standard XVA engines fit a Gaussian copula and compute CVA from it.
EconIAC's `SPLIT` applies the coboundary map to the bilateral credit spread
matrix and reads off the $H^1$ class directly from market prices. No copula.
No calibration. The $H^1$ class *is* the correlation structure.

### Compositionality

The opcodes compose. A sequence of ISA operations is a **circuit** on the
financial interaction diagram. Econiac's Pacioli Combinator Library (PCL)
provides the typed DSL for writing such circuits:

```python
from econiac.pcl import split, splat, twist, flip, flop, sequence

# Price a triangular risk: split bilateral prices → H¹ class → price
xva_circuit = sequence(
    split(bilateral_exposures),   # H⁰ → H¹: find the triangular obstruction
    twist(numeraire="risk_neutral"),  # gauge transform to pricing measure
    splat(conditional_expectation)    # H¹ → H⁰: collapse to price
)
```

Every circuit preserves the Pacioli identity (conservation of value) by
construction — the types enforce it.

### The Pentagon as a runtime check

The Pentagon identity is checkable at runtime. If a sequence of ISA operations
violates $d^2 = 0$, the EconIAC runtime raises a `PentagonViolation` — the
financial equivalent of a type error. This catches:

- Calendar spread arbitrage in a volatility surface
- HJM drift misspecification in an interest rate model
- No-arbitrage violation in a credit spread matrix
- $H^2 \neq 0$ system fragility in a stress scenario

---

## The ISA across scales

The Origami ISA is not specific to finance. Paper 370
([*The Origami ISA as Nature's Universal Computer*](https://doi.org/10.5281/zenodo.20543454))
shows the same five opcodes govern physical processes across 20 orders of
magnitude — from nuclear spectroscopy to quantum computing to molecular
energy transfer.

The reason: all of these systems share the same mathematical structure (a
representation sheaf on an interaction diagram with a Pentagon topology).
The ISA is the universal computational language for this structure.

EconIAC is the financial instantiation. The five opcodes, the Pentagon
identity, and the cohomological hierarchy ($H^0/H^1/H^2$) are the same
objects in finance as in physics — just with different sheaves:

| System | Sheaf | $H^0$ | $H^1$ | $H^2 = 0$ condition |
| --- | --- | --- | --- | --- |
| Nuclear spectroscopy | $SU(2)$ representation sheaf | Selection rules | Racah 6j symbol | Biedenharn–Elliott |
| Quantum computing | Stabiliser sheaf on $W(5,2)$ | Pauli syndromes | Magic valence | Pentagon identity |
| Interest rates | Discount factor sheaf | Bilateral prices | Convexity (HJM) | HJM no-arbitrage |
| Systemic risk | Pricing sheaf on interaction diagram | Bilateral stress | Triangular risk | $H^2$ stability |
| Supply chains | Input-output sheaf | Sector balances | Cross-sector coupling | Stock-flow consistency |

The universality is not an analogy. It is the same theorem — the 6j symbol
is $H^1$ of the relevant sheaf — instantiated for different sheaves.

---

## The connection to Paper 395 (IBAP as Origami ISA)

The Brody–Hughston–Macrina information-based asset pricing (IBAP) framework
is an Origami ISA programme:

- **SPLIT**: hidden market factor $X$ → noisy signal $\sigma t X$ + Brownian
  bridge $\beta_{tT}$ (the information process)
- **SPLAT**: information field $\xi_t$ → asset price $H_t$ (conditional expectation)
- **TWIST**: numeraire change (risk-neutral → terminal measure)
- **FLIP/FLOP**: time reversal and Born rule

The Brownian bridge $\beta_{tT}$ is parallel transport on the prequantum bundle;
its holonomy is the convexity adjustment — the same $H^1$ class that appears
in HJM. Paper 395 (in preparation) develops this in full.

---

## Further reading

| Paper | Content |
| --- | --- |
| [370 — The Origami ISA as Nature's Universal Computer](https://doi.org/10.5281/zenodo.20543454) | Five opcodes across 20 orders of magnitude; universality proof |
| [396 — The 6j Symbol as $H^1$](https://doi.org/10.5281/zenodo.20635479) | ISA opcodes as Čech cohomology operations (§6); five-instance table |
| [397 — Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) | ISA applied to systemic risk; Pentagon = HJM = stability |
| [303 — Pacioli Combinator Library](https://doi.org/10.5281/zenodo.20262070) | The typed DSL for ISA circuits in EconIAC |
| [393 — Projective Geometry as the Mother Tongue of QM](https://doi.org/10.5281/zenodo.20634729) | ISA as the finite-field limit of the Penrose transform |
| Paper 395 — IBAP as Origami ISA *(in preparation)* | Brownian bridge = parallel transport; SPLIT/SPLAT = BHM information process |
