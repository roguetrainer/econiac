# The Topology of Risk

> *"Just because I don't know what the connection is doesn't mean there isn't one."*
> — Douglas Adams, *The Long Dark Tea-Time of the Soul*

In 2008, every major bank's risk models passed their internal validation tests.
The system failed anyway — not because the models were wrong, but because they
were **locally correct and globally inconsistent**.

EconIAC is built on the insight that financial risk has three structural levels,
and that existing tools only address the first two.

---

## The three levels of financial risk

| Level | Name | What it is | Instruments | Who manages it |
| --- | --- | --- | --- | --- |
| $H^0$ | **Bilateral risk** | Risk between two parties | Forwards, swaps | Trading desks |
| $H^1$ | **Triangular risk** | Risk visible only when three parties interact | Options, swaptions, CDOs | XVA desks |
| $H^2$ | **Systemic risk** | Mutual inconsistency of institutions' triangular risks | CCPs, central banks | Regulators, CRO |

**Bilateral risk** ($H^0$) is perfectly hedgeable with forwards and swaps.
Every VaR model, every balance-sheet stress test, every bilateral exposure
report operates at this level.

**Triangular risk** ($H^1$) is the risk that only appears when three parties
interact. No portfolio of bilateral contracts can eliminate it — this is a
mathematical theorem, not a practical limitation. Convexity, basis risk,
correlation risk, CVA, FVA, and the volatility smile are all triangular risks.
Options and swaptions exist *because* $H^1 \neq 0$.

**Systemic risk** ($H^2$) is the risk of the whole: the mutual inconsistency of
individual institutions' triangular risk estimates. Wrong-way risk in XVA is $H^2$.
The 2008 crisis was $H^2$. No existing risk system computes $H^2$.

---

## What EconIAC computes that existing systems cannot

| Computation | Existing tools | EconIAC |
| --- | --- | --- |
| Bilateral stress test | ✅ Standard (VaR, balance sheet) | ✅ |
| Triangular risk ($H^1$) | Partial (XVA desks, CVA) | ✅ Full sheaf cohomology |
| Systemic stability ($H^2$) | ❌ Not computed anywhere | ✅ Pentagon identity test |
| Pre-crisis early warning | Contemporaneous indicators | ✅ $H^1$ leads cascade by 2–3 periods |
| Policy gradient $\partial\text{loss}/\partial\text{haircut}$ | ❌ | ✅ One backward pass |
| SIFI designation | Size-based (FSB) | ✅ Topological — $H^2$ contribution |
| Wrong-way risk in XVA | Parametric approximation | ✅ Exact $H^2$ class |

---

## Who EconIAC is for

- [**Central banks and regulators**](regulators.md) — cohomological stress testing,
  $H^2$ early-warning indicator, topological SIFI designation
- [**Banks and financial institutions**](banks.md) — XVA wrong-way risk, triangular
  risk classification, model-free correlation pricing
- [**RegTech and model vendors**](regtech.md) — the computational platform for
  bilateral/triangular/systemic risk as a service

---

## Start here

The fastest path to understanding the framework:

1. [The Topology of Risk: A Primer](https://doi.org/10.5281/zenodo.20642983) —
   13 pages, no prior mathematics required, Holmes vs Gently framing
2. [The 6j Symbol as $H^1$](https://doi.org/10.5281/zenodo.20635479) —
   the unhedgeability theorem and five-instance table
3. [Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) —
   cohomological stress test, SIFI theorem, XVA section
