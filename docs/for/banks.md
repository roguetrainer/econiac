# EconIAC for Banks and Financial Institutions

> *"Your XVA desk computes triangular risk ($H^1$). Your wrong-way risk model
> is trying to compute a systemic risk ($H^2$) class with $H^1$ tools.
> That is why it keeps breaking in stress scenarios."*

---

## The three risks on your book

Every risk on a bank's book falls into exactly one of three cohomological levels.
Knowing which level determines which instruments hedge it and which models price it.

| Level | Name | Your book | Hedging instrument |
| --- | --- | --- | --- |
| $H^0$ | **Bilateral risk** | Delta, DV01, single-name credit | Forwards, swaps, single-name CDS |
| $H^1$ | **Triangular risk** | Convexity, basis, correlation, CVA, FVA, MVA | Options, swaptions, correlation swaps |
| $H^2$ | **Systemic risk** | Wrong-way risk, systemic wrong-way, capital at SIFI threshold | CCP margining, regulatory capital |

This classification is a **mathematical theorem** (see
[Paper 396](https://doi.org/10.5281/zenodo.20635479)), not a regulatory
convention. A bilateral instrument cannot hedge a triangular risk regardless
of how many bilateral instruments you hold.

---

## XVA: where your desk lives in this picture

| Adjustment | Level | Why | Hedgeable at desk? |
| --- | --- | --- | --- |
| CVA | $H^1$ | Triangle: you, counterparty, underlying | ✅ Credit options, CDS |
| DVA | $H^1$ | Triangle: counterparty, you, funding | ✅ Own-name CDS |
| FVA | $H^1$ | Triangle: you, funding desk, collateral | ✅ Funding swap |
| MVA | $H^1$ | Triangle: you, CCP, variation margin | ✅ Margin rate swap |
| KVA | $H^1$/$H^2$ | Capital depends on SIFI class ($H^2$) | Partial |
| **Wrong-way risk** | **$H^2$** | **Tetrahedron: you, counterparty, market, funding** | **❌ Never** |

**Wrong-way risk is $H^2$.** It is the mutual inconsistency of your CVA,
FVA, and DVA triangular risks. Standard XVA models that sum
CVA + DVA + FVA + MVA compute $H^1$ only. The error — the difference
between the summed XVA and the true total — is exactly the $H^2$
wrong-way risk, and it cannot be reduced by any desk-level model
regardless of sophistication.

The $H^2$ component requires system-level data: correlations between
counterparty credit quality and market moves across the whole network.
This data lives at your CCP and at regulators. The natural model boundary is:

- **XVA desk** → compute $H^1$ (CVA, DVA, FVA, MVA)
- **CRO / risk management** → estimate $H^2$ wrong-way risk from system data
- **Regulator / CCP** → provide system-level $H^2$ correlation data

---

## New capabilities EconIAC provides

### Model-free triangular risk pricing

Standard practice: fit a parametric model (Gaussian copula, SABR, HJM) and
calibrate to market prices.

EconIAC approach: compute the $H^1$ class of the pricing sheaf directly from
market prices of liquid triangular instruments (CDX tranches, correlation swaps,
basket options). No model. No parametric assumption. The $H^1$ class *is* the
correlation structure — model-free by construction.

### Systematic arbitrage detection

A non-trivial $H^1$ class on any triangle of instruments is a structural
market inconsistency: either an arbitrage opportunity or an unhedged risk.
EconIAC scans $H^1$ over all triangles in an instrument universe
simultaneously — a topological arbitrage scanner that requires no parametric
model and identifies structural inconsistencies that pair-by-pair analysis misses.

### KVA at the SIFI boundary

An institution near the SIFI threshold faces KVA that is sensitive to
system-level $H^2$ changes: whether it is designated a SIFI depends on its
$H^2$ contribution to the system, which changes as other institutions' books
change. Monitoring the system's $H^2$ class is commercially valuable to any
large institution near the SIFI boundary.

### Differentiable contagion models

EconIAC's fire-sale and repo-run models are end-to-end differentiable:
the policy gradient $\partial\text{loss}/\partial\text{haircut}$ is computable
in one backward pass. This enables:

- Optimal collateral haircut calibration targeting $H^1 = 0$
- Minimum-cost buffer allocation against cascade scenarios
- Reverse stress testing: find the smallest shock that triggers $H^2 \neq 0$

---

## The silo problem

Your interest rate desk, FX desk, credit desk, and commodity desk each run
separate models calibrated in isolation. Each model is internally valid.

Cross-desk interactions — a rate shock triggering funding stress triggering
credit moves — are $H^2$ effects. They are invisible to any single-desk model
because $H^2$ is a property of the interactions between desks, not of any
individual desk.

EconIAC models the full interaction diagram across desks and computes the
$H^2$ class of your book — the mutual inconsistency of your desks' risk
estimates under joint stress.

---

## Papers

| Paper | Content |
| --- | --- |
| [396 — The 6j Symbol as $H^1$](https://doi.org/10.5281/zenodo.20635479) | Unhedgeability theorem; CVA/FVA/convexity as $H^1$; five-instance table |
| [397 — Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) | XVA section; wrong-way risk as $H^2$; SIFI theorem; KVA at boundary |
| [299 — XVA as Curvature](https://doi.org/10.5281/zenodo.20257724) | CVA/DVA/FVA/MVA as gauge curvature; Burgard–Kjaer PDE as flatness |
| [296 — Term Structure Bundles](https://doi.org/10.5281/zenodo.20244445) | HJM convexity as $H^1$; the $\frac{1}{2}$ from Maslov = the $\frac{1}{2}$ from Itô |
| [398 — The Topology of Risk (Primer)](https://doi.org/10.5281/zenodo.20642983) | Plain-language introduction; XVA table; wrong-way risk explained |
