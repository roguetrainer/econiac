# About EconIAC

## The name

**Econ**omic **I**ntegrator **A**nd **C**omputer — named after [MONIAC](https://en.wikipedia.org/wiki/MONIAC) (1949),
Bill Phillips's hydraulic computer that modelled the British economy as a conserved flow system
using tanks, pipes, and valves. EconIAC does the same with differential geometry, the
Maslov-Gibbs partition function, and JAX.

> *"The economy is a flow network. Conservation laws are not optional."*

The name continues a chain: **ENIAC** (1945) → **MONIAC** (1949) → **EconIAC** (2026).

The backronym is doubly apt. EconIAC literally *integrates* ODEs (Keen debt dynamics,
Vensim/Stella models via pysd) and integrates in the measure-theoretic sense (partition
functions $Z(\beta)$, path integrals over the Pacioli manifold). "Integrator" is shared
verbatim with ENIAC's own expansion — *Electronic Numerical **Integrator** And Computer*.

---

## The MONIAC connection

In 1949, the New Zealand economist Bill Phillips built MONIAC: a hydraulic computer that
modelled the British economy as a system of tanks, pipes, and valves, with coloured water
representing money flows. Conservation was enforced physically — what flowed in had to flow
out. It was a working analogue computer that correctly predicted macroeconomic dynamics.

MONIAC stands for **Monetary National Income Analogue Computer** — and is simultaneously a
portmanteau of ENIAC and "money". Phillips coined it to signal that economic computation
deserved the same ambition as the cutting-edge computing of his day.

**What does EconIAC compute?** All of the following are special cases of the same underlying
computation — the free energy of a flow network at inverse temperature $\beta$:

| Output | Description |
| --- | --- |
| Accounting measures | Sectoral balances, XVA — integrals of curvature on the Pacioli manifold |
| Entropies & free energies | The partition function $Z(\beta)$ at the core of TIR and the MGE |
| Sensitivities | Greeks, thermal attribution — automatic differentiation via JAX |
| Optima & equilibria | The high-$\beta$ limit of the Gibbs distribution |
| Calibration weights | The $\beta$-schedule from analogue exploration to discrete commitment |

EconIAC is MONIAC for the 21st century: the same conservation laws, the same flow network,
implemented with differential geometry, thermodynamic routing, and automatic differentiation.

---

## The Pacioli manifold

The **Pacioli manifold** is named after Luca Pacioli (1447–1517), the Franciscan friar who
documented double-entry bookkeeping in his 1494 *Summa de arithmetica* — the first printed
treatment of the system in Europe. The name honours the mathematical structure he documented,
not its invention.

Double-entry bookkeeping was not a European invention. The practice predates Pacioli by
centuries and arrived in Europe via multiple routes:

- **India**: the *Bahi-Khata* system, used by merchants in Rajasthan and Gujarat, dates to at
  least the 12th century.
- **Islamic world**: Abbasid-era *hawala* (8th–13th century) and the accounting practices of
  medieval Islamic merchants documented in the Cairo Geniza show conservation-of-value
  principles structurally equivalent to double-entry.
- **China**: the *四柱清册* (four-column account) system, used from at least the Tang dynasty
  (7th century), encodes the same debit-credit conservation law.
- **Transmission to Europe**: Italian merchants encountered these systems through the spice
  and silk trades. The Florentine merchant Francesco Datini's ledgers (c. 1299) predate
  Pacioli by nearly two centuries.

Pacioli's contribution was to recognise the mathematical structure and give it a systematic
written treatment. By Stigler's Law of Eponymy (*no scientific discovery is named after its
original discoverer*), "Pacioli manifold" is in good historical company. The conservation
law $\partial^2 = 0$ that defines the manifold belongs to no single culture — it is a
universal consequence of the mathematics of flows.

---

## Research programme

EconIAC is the software companion to the
[Portfolio G](https://roguetrainer.github.io/adelic-simplicial-architecture/portfolios/portfolio-g/)
papers of the Adelic Simplicial Architecture (ASA). The theoretical frameworks it implements:

- **Thermodynamic Information Routing (TIR)** — the Gibbs distribution as a universal routing
  primitive across economics, game theory, and computation
  ([Paper 294](https://doi.org/10.5281/zenodo.20237288))
- **Maslov-Gibbs Einsum (MGE)** — the tensor contraction operator executing Gibbs-weighted
  routing; in the tropical limit ($\beta \to \infty$) it dequantizes to max-plus, recovering
  discrete argmax from a continuous sum-product ([Paper 201](https://doi.org/10.5281/zenodo.17981393))
- **Economic Gauge Theory (EGT)** — stock-flow consistency as a discrete gauge theory on the
  Pacioli manifold ([Paper 300](https://doi.org/10.5281/zenodo.20259495))
- **Financial Gauge Theory (FGT)** — exchange rates, yield curves, and credit spreads as
  connections and curvature ([Papers 295–299](https://doi.org/10.5281/zenodo.20242355))
- **Pacioli Combinator Library (PCL)** — a DSL for financial computation that enforces
  conservation by construction ([Paper 306](https://doi.org/10.5281/zenodo.20262070))
- **Differentiable Agent-Based Macroeconomics (DABM)** — fully differentiable macroeconomic
  digital twins calibratable by gradient descent ([Paper 305](https://doi.org/10.5281/zenodo.20261945))

---

## FAQ

**Is EconIAC analogue or digital?**

All three, in historical order:

- **ENIAC (1945)** was *digital* — discrete voltage states, Boolean logic, stored program.
- **MONIAC (1949)** was *analogue* — continuous water flows, physical conservation enforced by hydraulics.
- **EconIAC** is *analogue emulated on a digital computer* — continuous probability flows (the
  Gibbs distribution) running on discrete hardware.

There is a precise sense in which EconIAC is also digital: in the high-$\beta$ limit, Gibbs
weights collapse to a hard argmax — one route takes all the flow, all others get zero. This is
winner-takes-all, i.e. digital switching. At low $\beta$ the weights are a soft mixture: fully
analogue. The $\beta$-schedule interpolates between analogue (exploration) and digital
(commitment) behaviour. The SNAP phase of the MGE is literally this transition — analogue
search crystallising into a discrete decision.

EconIAC is analogue at heart, digital at the boundary, and the boundary is where the answer lives.

---

**Why not just use DSGE / standard macro models?**

Standard DSGE treats agents as perfectly rational ($\beta \to \infty$). This makes the model
non-differentiable at equilibrium — policy gradients don't exist. EconIAC treats rationality
as a calibrated temperature $\beta^*$, which gives exact policy gradients via the Implicit
Function Theorem. At $\beta \to \infty$ the classical DSGE equilibrium is recovered exactly.
The representative agent is the degenerate limit of the Gibbs ensemble, not its foundation.

---

**What is the relationship to ABMs?**

Every discrete threshold rule in a standard agent-based model (if/else, majority vote, min/max)
has a Gibbs-ensemble equivalent parametrised by $\beta$. EconIAC is a differentiable ABM
framework: replace each hard threshold with its smooth Gibbs relaxation, and the entire model
becomes calibratable by gradient descent. At $\beta \to \infty$ the hard rules are recovered.
