# The Topology of Risk

## Identifying, Measuring, and Controlling Inconsistency in Financial Systems

*Ian R. C. Buckley*

---

This book develops a single idea from first principles: **financial risk is fundamentally about inconsistency**.

A bilateral contract is consistent with another bilateral contract. Three bilateral contracts may be mutually consistent — or they may form a funding loop that cannot be closed. Four consistent triangles may still fail to fit together. Each level of inconsistency has a name, a measure, and a set of instruments that can (or cannot) address it. The three levels are $H^0$, $H^1$, and $H^2$.

No prior knowledge of algebraic topology is assumed. Every concept is introduced through a worked example that a credit risk manager, XVA desk, or regulator would recognise. Definitions follow examples, not the other way round.

---

## Who this book is for

- **Risk managers and XVA desks** who want to understand why certain risks cannot be hedged away, not just as a practical observation but as a theorem
- **Regulators and central bankers** who want a framework that distinguishes systemic risk ($H^2$) from bilateral and triangular risk, and that gives exact measures rather than heuristics
- **Fund managers** facing basis risk, wrong-way risk, LDI, and currency overlay problems that standard models handle poorly
- **Quantitative analysts** who want the mathematical foundations without category theory or spectral sequences

---

## How to read this book

The chapters build sequentially. Chapters 1–2 establish the geometric language (simplicial complexes, boundary operators). Chapters 3–5 cover $H^0$ and $H^1$ — the bilateral and triangular levels that existing tools partially address. Chapters 6–8 build to $H^2$ and the 2008 crisis.

Each chapter opens with a numerical example, derives the concept from that example, and closes with the formal definition.

---

## Chapters

1. [Double-entry bookkeeping as $\partial^2 = 0$](ch01_double_entry.md) — the simplest homology
2. [The balance sheet as a simplicial complex](ch02_balance_sheet_complex.md) — nodes, edges, triangles
3. [When do bilateral rates fit together?](ch03_h0_global_sections.md) — $H^0$ and global sections
4. [Funding loops and the first Betti number](ch04_h1_funding_loops.md) — $H^1$ and independent cycles
5. [Netting and clearing](ch05_netting_clearing.md) — removing edges vs filling triangles; the EN model as $H^0$
6. [Basis, convexity, wrong-way risk](ch06_h1_unhedgeable.md) — the unhedgeable residual as an $H^1$ class
7. [The hollow tetrahedron](ch07_hollow_tetrahedron.md) — when four consistent triangles still fail to close; $H^2$
8. [The 2008 crisis as an $H^2$ event](ch08_2008_crisis.md) — correlation desks, AIG, Lehman prime brokerage

---

## Relationship to the papers

The EconIAC papers (available at [zenodo.org/communities/econiac](https://zenodo.org/communities/econiac/)) contain the formal theorems and proofs. This book contains the same ideas in a different register: worked examples, financial intuition, and applications. The fastest entry point to the papers is [The Topology of Risk: A Primer](https://doi.org/10.5281/zenodo.20642983) (13 pages, no prior mathematics required).
