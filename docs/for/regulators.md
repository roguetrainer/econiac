# EconIAC for Central Banks and Regulators

> *"Current stress tests add up bilateral losses. This framework computes
> whether those losses are mutually consistent — and whether the system will
> amplify or absorb them. The 2008 crisis was an $H^2$ event that no $H^0$
> or $H^1$ tool could have predicted."*

---

## The problem with current stress tests

The Federal Reserve's DFAST, the EBA stress test, and the ECB's SREP all
apply a macroeconomic scenario to each institution's portfolio and aggregate
the losses. This is an $H^0$ computation: it evaluates bilateral exposures
independently and adds them up.

It misses two things:

**Triangular risk ($H^1$):** how bilateral losses at each institution propagate
through shared counterparties, correlated positions, and common funding sources.
XVA desks at sophisticated institutions partially capture this; system-level
stress tests do not.

**Systemic risk ($H^2$):** whether individual institutions' triangular risk
estimates are mutually consistent. When they are not — when the Pentagon identity
fails at the system level — losses amplify rather than absorb. This is the
mechanism of financial crises. **No existing stress test computes $H^2$.**

---

## What EconIAC provides

### The cohomological stress test

A three-tier extension of current stress testing practice:

| Tier | Level | What it computes | Current practice |
| --- | --- | --- | --- |
| 0 | $H^0$ | Sum of bilateral losses | ✅ Standard (DFAST, EBA) |
| 1 | $H^1$ | Propagation through interaction triangles | Partial (XVA, IMM models) |
| 2 | $H^2$ | Topological stability: self-limiting or self-amplifying? | ❌ Not done anywhere |

The Tier 2 output is qualitatively different from Tiers 0 and 1: it is not a
loss estimate but a **stability classification**. A system with $H^2 = 0$ under
stress will absorb losses; a system with $H^2 \neq 0$ will amplify them.
This is detectable before any individual institution breaches a threshold.

### The $H^2$ early-warning indicator

The $H^2$ class of the financial system is computable from **market prices of
liquid correlation instruments** — CDX/iTraxx tranches, correlation swaps,
variance dispersion trades — that are observable in real time.

A rising $H^2$ class signals that institutions' triangular risk estimates are
becoming mutually inconsistent. In the 2008 case, this signal was available
in ABX tranche pricing from late 2006 — six months before individual
institution failures.

**Data required:** trade repository data (EMIR, FSB-LEI, ECB repo statistics)
plus prices of liquid correlation instruments. All available to regulators today.

**Computation:** finite linear algebra on the Čech complex of the financial
interaction diagram. Tractable even for large networks.

### The SIFI theorem

Current SIFI designation uses size metrics (total assets, cross-jurisdictional
activity) defined by the FSB. These are $H^0$ metrics.

**The topological criterion:** an institution is systemically important if and
only if its removal changes the $H^2$ class of the system.

- A large institution with zero $H^2$ contribution can fail safely.
- A small institution that is a critical node in a large $H^2$ class cannot.
- Size is neither necessary nor sufficient for systemic importance.

This gives regulators a principled, computable alternative to the current
size-based framework — one that identifies systemic importance from network
topology rather than balance sheet scale.

### $H^2$-based capital charges

The Basel III/IV correlation trading book capital charge attempts to capture
$H^1$ risk but uses the Gaussian copula — a parametric model that systematically
misspecifies correlation structure. An $H^2$-based capital charge would:

1. Compute the $H^1$ class of each institution's pricing sheaf from market
   prices of triangular instruments (model-free).
2. Compute each institution's $H^2$ contribution to the system.
3. Set systemic risk capital proportional to $H^2$ contribution.

Institutions with zero $H^2$ contribution need no systemic risk capital surcharge.

---

## Relation to existing systemic risk measures

| Measure | Level | What it misses |
| --- | --- | --- |
| DebtRank (Battiston et al. 2012) | $H^0$ | All triangular and systemic effects |
| CoVaR (Adrian & Brunnermeier 2011) | $H^1$ | System-level mutual consistency |
| SRISK (Brownlees & Engle 2017) | $H^1$ | System-level mutual consistency |
| Flood et al. (2017) Betti numbers | Graph topology | Financial content (pricing sheaf) |
| **EconIAC $H^2$** | **$H^2$** | **Nothing — full three-tier picture** |

EconIAC subsumes all existing measures and adds the $H^2$ tier that none of
them compute.

---

## The 2008 crisis as an $H^2$ event

Individual $H^1$ mortgage risks at each institution were locally reasonable.
The cross-institution correlation of mortgage exposures was an $H^2$ class
that no regulator computed. When the $H^2$ class became non-trivial, the
Pentagon identity failed and the cascade began.

An $H^2$ stress test in 2006 would have shown:

- Individual $H^1$ mortgage risks: within limits ✓
- System $H^2$ class: non-trivial and growing — **cascade structurally guaranteed** ✗

---

## Papers

| Paper | Content |
| --- | --- |
| [397 — Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) | Cohomological stress test; SIFI theorem; XVA; 2008 analysis |
| [396 — The Unhedgeability Theorem](https://doi.org/10.5281/zenodo.20635479) | Unhedgeability theorem; five-instance table; Pentagon identity |
| [398 — The Topology of Risk (Primer)](https://doi.org/10.5281/zenodo.20642983) | Plain-language introduction; no mathematics required |
| [291 — The Topology of Conservation](https://doi.org/10.5281/zenodo.20234853) | Double-entry accounting as discrete gauge theory |
| 332 — CHZ Fire Sales *(in preparation)* | Differentiable ABM; sheaf $H^1$ leads cascade by 2–3 periods |
| 333 — European Sovereign Repo Run *(in preparation)* | Calibrated to ECB data; 2022 LDI crisis as $H^2$ event |
