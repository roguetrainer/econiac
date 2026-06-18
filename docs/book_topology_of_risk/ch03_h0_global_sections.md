# Chapter 3: When Do Bilateral Rates Fit Together?

> *Three dealers each quote a bilateral FX rate. Do the three quotes agree?*

---

## The example

Three currencies: USD, EUR, GBP. Three dealers each quote one pair:

- Dealer 1: EUR/USD = 1.10
- Dealer 2: GBP/USD = 1.27
- Dealer 3: EUR/GBP = 0.87

Do these rates fit together? If EUR/USD = 1.10 and GBP/USD = 1.27, then EUR/GBP should be $1.10/1.27 = 0.866$. But Dealer 3 quotes 0.87. The three bilateral quotes are mutually inconsistent: there is a triangular arbitrage of approximately 0.4 basis points.

This is the simplest example of a **global section problem**: given locally defined data (one rate per edge), does there exist a globally consistent assignment (a single price for each currency) that reproduces all the bilateral rates simultaneously?

---

## $H^0$: global sections

$H^0$ of the obligation complex $\Gamma$ measures whether locally defined values can be assembled into a globally consistent assignment. For exchange rates, $H^0 = 0$ means triangular arbitrage is absent: all bilateral rates are consistent with a single set of currency prices.

When $H^0 \neq 0$, there is a globally irresolvable inconsistency at the level of node values — not cycles, not voids, but the most basic failure of consistency.

In practice $H^0$ is rarely non-zero for exchange rates (triangular arbitrage is closed quickly). But the same question arises in:

- **Transfer pricing**: do the bilateral prices between divisions of a multinational firm produce a consistent global valuation?
- **Interbank lending rates**: do bilateral quoted rates imply consistent term structures?
- **Collateral valuation**: do three dealers value the same collateral consistently when it passes between them?

---

## The Eisenberg-Noe clearing model as $H^0$

The Eisenberg-Noe (2001) clearing model asks: given a network of bilateral debt obligations, what is the clearing payment vector — the set of payments each firm makes to each other — that is consistent with all bilateral contracts simultaneously?

This is exactly the $H^0$ problem: find global node values (firm equity) consistent with all edge values (bilateral claims). The EN algorithm finds this global section when it exists, and identifies which firms default (which nodes have zero equity) when full repayment is impossible.

The EN model is powerful for bilateral risk. Its limitation — why the 2008 crisis was not predicted by EN-style stress tests — is that it operates entirely at $H^0$. It cannot see funding loops ($H^1$) or hollow tetrahedra ($H^2$). Chapter 5 returns to this.

---

## The formal statement

Given a sheaf $\mathcal{F}$ on $\Gamma$ (an assignment of a value space to each node and edge, with restriction maps between them), a **global section** is an element of $H^0(\Gamma, \mathcal{F})$: a consistent assignment of values to all nodes that restricts correctly to all edges.

For exchange rates: $\mathcal{F}$ assigns $\mathbb{R}_{>0}$ to each node (the currency price) and $\mathbb{R}_{>0}$ to each edge (the bilateral rate). A global section is a set of prices $\{p_i\}$ such that $r_{ij} = p_i/p_j$ for every bilateral rate $r_{ij}$. $H^0 = 0$ iff no triangular arbitrage exists.

---

*Next: [Chapter 4 — Funding Loops and the First Betti Number](ch04_h1_funding_loops.md)*
