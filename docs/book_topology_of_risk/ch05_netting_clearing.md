# Chapter 5: Netting and Clearing

> *Bilateral netting removes edges. CCP novation fills triangles. Neither removes all loops.*

---

## The example

Four dealers: A, B, C, D. Each has a bilateral IRS position with the other three — six edges in total, forming a complete graph $K_4$. The gross notional is £10bn per edge; £60bn total.

**Bilateral netting** within each pair: A and B net their positions against each other, reducing their bilateral exposure from £10bn gross to perhaps £1bn net. The edge remains but is smaller. $\beta_1$ is unchanged: the loops are still there.

**CCP novation**: A and B each novate their trade to a central counterparty (CCP). Now A has a contract with the CCP, and B has a contract with the CCP. The original A--B edge is replaced by two edges (A--CCP and B--CCP). But this adds the CCP as a new node — a hub connecting all four dealers. The star graph has $\beta_1 = 0$: no funding loops. The loops have been filled, not just reduced.

This is the topological distinction between netting and clearing.

---

## Removing edges vs filling triangles

| Operation | Topological effect | Effect on $\beta_1$ |
|---|---|---|
| Bilateral netting | Reduces edge weight | None — loop still exists |
| Multilateral netting | May remove edges | Reduces $\beta_1$ if loop is broken |
| CCP novation (full) | Replaces cycle with star | $\beta_1 \to 0$ for cleared products |
| Partial novation | Replaces some legs | $\beta_1$ reduced but not zero |

The Netting Hierarchy Theorem (Paper 439) makes this precise: bilateral netting $\subset$ multilateral netting $\subset$ CCP novation in terms of $\beta_1$ reduction. Each is strictly more powerful than the previous, but none can reduce $\beta_2$ to zero.

---

## The $H^2$ Impossibility Theorem

CCP clearing can reduce $\beta_1$ to zero for a single product class. But financial institutions trade multiple products: IRS, CDS, FX, repo, equity derivatives. A loop that passes through IRS on one leg and FX on another is not cleared by any single CCP.

The **$H^2$ Impossibility Theorem**: no combination of bilateral netting, multilateral netting, or CCP novation by private institutions can reduce $\beta_2(\Gamma)$ to zero. The irresolvable four-party conflicts (hollow tetrahedra) survive any netting or clearing arrangement that operates product by product.

This is why the 2008 crisis was not resolved by close-out netting: the problem was not bilateral or triangular, it was $H^2$. Novation closed the loops on individual products; the cross-product hollow tetrahedra remained.

---

## The EN model revisited

Chapter 3 introduced Eisenberg-Noe as a global section problem ($H^0$). We can now see its limitation precisely. EN finds the clearing payment vector for a given network of bilateral debts. It operates on the 0-skeleton and 1-skeleton only. It cannot:

- Detect funding loops ($H^1$) that amplify before a default is triggered
- Identify hollow tetrahedra ($H^2$) that survive close-out

The 2008 EN-style stress tests returned green because the bilateral exposures were manageable. The crisis was driven by $H^1$ amplification (repo run) and $H^2$ irresolvability (AIG, Lehman prime brokerage). These were invisible to the tools in use.

---

*Next: [Chapter 6 — Basis, Convexity, Wrong-Way Risk](ch06_h1_unhedgeable.md)*
