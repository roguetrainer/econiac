# Chapter 6: Basis, Convexity, and Wrong-Way Risk

> *You cannot hedge a loop with an edge. The residual is $H^1$.*

---

## The example

An XVA desk runs a book of interest rate swaps with 50 counterparties. It hedges its net delta with a single offsetting position in the swap market. The bilateral exposures are hedged.

But the desk also has CVA: credit valuation adjustment, the expected loss from counterparty defaults. CVA depends on the correlation between counterparty credit quality and the value of the swap — wrong-way risk. If the counterparty is most likely to default precisely when the swap is most valuable to the desk, the CVA is large and the hedge is imperfect.

This imperfection cannot be hedged away with more bilateral contracts. It is an $H^1$ class: a property of the triangle formed by (desk, counterparty, swap market), not of any individual edge.

---

## $H^1$ financial objects

| Object | The triangle | Why it is $H^1$ |
|---|---|---|
| CVA wrong-way risk | Desk -- counterparty -- collateral market | Default correlation is a property of the cycle, not any edge |
| Convexity (rates) | Bond -- swap -- futures | The convexity adjustment arises from the closed path through the three instruments |
| Cross-currency basis | USD funding -- EUR funding -- FX forward | CIP deviation = holonomy of the currency triangle |
| Correlation risk | Asset A -- Asset B -- basket option | Basket price depends on the triangle, not the bilateral correlations alone |
| FVA | Desk -- treasury -- funding market | Funding value adjustment is a property of the three-node funding cycle |

All are $H^1$: they are properties of cycles that cannot be reduced to properties of edges. No portfolio of bilateral contracts can hedge them away — this is a theorem (the Unhedgeability Theorem), not a practical observation about market incompleteness.

---

## The Unhedgeability Theorem

**Theorem:** No portfolio of $H^0$ instruments (forwards, bilateral swaps) can reduce $H^1$ risk to zero. The $H^1$ class of the CVA triangle is invariant under addition of bilateral hedges.

The intuition: adding a bilateral contract adds an edge to $\Gamma$. Adding an edge can reduce $\beta_1$ only if it closes a cycle — but CVA wrong-way risk is a property of a cycle that already exists. Adding more edges to a different part of the network does not remove an existing cycle.

Options and swaptions exist *because* $H^1 \neq 0$. They are the instruments that address triangular risk, not bilateral risk. This is why the options market was born: to hedge the $H^1$ residual that the swap market cannot touch.

---

## Convexity as $H^1$

The convexity adjustment between a bond and a futures contract is the simplest example of $H^1$ in rates. A bond, a forward rate agreement, and a futures contract on the same underlying form a triangle. The convexity adjustment is the holonomy of this triangle: the difference between the bond price implied by the futures and the actual bond price.

Before the introduction of options, convexity was managed heuristically. Once framed as an $H^1$ class, it is clear that: (a) it cannot be eliminated by bilateral contracts; (b) it can be priced model-free from the triangle structure; (c) it must be present whenever the three instruments form an independent cycle.

---

## What the XVA desk can and cannot do

| XVA component | Cohomological level | Hedgeable? |
|---|---|---|
| Mark-to-market (MtM) | $H^0$ | Yes — bilateral swap |
| CVA (expected loss) | $H^1$ | Partially — CDS on counterparty |
| Wrong-way CVA | $H^1$ | No — the correlation is the $H^1$ class |
| FVA | $H^1$ | No — funding loop is the $H^1$ class |
| KVA | $H^1$/$H^2$ boundary | No |
| Wrong-way risk in systemic stress | $H^2$ | No — $H^2$ Impossibility Theorem |

The XVA desk lives at the $H^1$ boundary. It can hedge the $H^0$ components cleanly and approximate the $H^1$ components with options and CDS, but the true $H^1$ residual (wrong-way risk, FVA loop) is irreducible.

---

*Next: [Chapter 7 — The Hollow Tetrahedron](ch07_hollow_tetrahedron.md)*
