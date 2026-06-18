# Chapter 7: The Hollow Tetrahedron

> *Four firms. Six bilateral contracts. Four consistent triangles. One irresolvable conflict.*

---

## The example

Four institutions: a bank (A), an insurer (B), a money market fund (C), and a hedge fund (D). Each pair has a bilateral contract — six edges forming a complete graph $K_4$.

Each triangle is internally consistent:
- {A, B, C}: A's swap with B, B's repo with C, C's commercial paper held by A — all consistent
- {A, B, D}: consistent
- {A, C, D}: consistent
- {B, C, D}: consistent

Every bilateral stress test passes. Every triangle closes. And yet: the four triangles together form a **hollow tetrahedron** — a 2-sphere with no 3-simplex filling it. There is no single settlement mechanism that can resolve all six bilateral claims simultaneously, because the four triangular faces are consistent individually but mutually contradictory as a whole.

This is $H^2 \neq 0$.

---

## What makes it hollow

A **filled** tetrahedron has a 3-simplex inside it: a four-way settlement mechanism that can resolve all claims at once. Think of a CCP that clears all four firms across all products simultaneously — the CCP is the interior point.

A **hollow** tetrahedron has no such interior. The four firms each have bilateral close-out rights, but exercising A's rights against B interferes with C's rights against B, which interferes with D's rights against C. The claims are mutually inconsistent. No sequence of bilateral settlements resolves them all; some creditor always ends up with less than their bilateral contract entitles them to.

The technical condition: the tetrahedron is hollow iff the 2-cycle formed by the four triangular faces is not a boundary — it does not bound any 3-chain in $\Gamma$. This is precisely the statement that it represents a non-trivial element of $H^2(\Gamma)$.

---

## Why bilateral close-out fails

When Lehman Brothers failed in September 2008, its prime brokerage clients (hedge funds) held assets at LBIE (the UK entity). Those assets had been rehypothecated: lent out to other counterparties by Lehman as collateral. The four parties — Lehman, the hedge funds, Barclays (which acquired parts of the US business), and the LBIE administrators — held mutually inconsistent claims.

Each bilateral claim was valid. Each triangular sub-claim was consistent. But the four-party structure was a hollow tetrahedron: the LBIE administration lasted over a decade because there was no interior point — no settlement mechanism that could satisfy all claims simultaneously.

This is $\beta_2 > 0$. The Impossibility Theorem says it could not have been resolved by any netting or clearing arrangement operating on the individual bilateral contracts. The only resolution was exogenous: a court-supervised process that imposed a settlement from outside the network.

---

## Measuring $\beta_2$

$\beta_2(\Gamma)$ counts the number of independent hollow tetrahedra in the obligation complex. Each one represents an irresolvable four-party conflict — a potential Lehman LBIE situation.

A system with $\beta_2 = 0$ can always be resolved by some combination of bilateral close-out and multilateral netting. A system with $\beta_2 > 0$ cannot: some residual will always remain. The size of $\beta_2$ measures the irreducible systemic complexity — the part of systemic risk that no private arrangement can eliminate.

---

## The SIFI theorem

An institution $i$ is a **systemically important financial institution (SIFI)** in the topological sense if $\Delta\beta_2(i) = \beta_2(\Gamma) - \beta_2(\Gamma \setminus \{i\}) > 0$: removing $i$ from the network reduces $\beta_2$.

This is a precise, model-free definition. It does not depend on size, leverage, or any parametric model of contagion. An institution is systemically important if and only if it is a vertex of a hollow tetrahedron — if its obligations complete an irresolvable four-party conflict.

Under this definition, size is neither necessary nor sufficient for SIFI status: a small institution that bridges two otherwise-disconnected subnetworks may have $\Delta\beta_2 > 0$, while a very large institution operating entirely within a well-connected subnetwork may have $\Delta\beta_2 = 0$.

---

*Next: [Chapter 8 — The 2008 Crisis as an $H^2$ Event](ch08_2008_crisis.md)*
