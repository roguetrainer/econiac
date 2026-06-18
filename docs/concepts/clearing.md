# Clearing, Netting, and the Topology of Obligation

> *"The gold at the centre of the system is not gold at all —
> it is the promise of orderly netting."*
> — Perry Mehrling, *The New Lombard Street*

Whether a network of financial obligations clears bilaterally, through a
central counterparty, or not at all is not just an operational detail.
It changes the *topology* of the network — which determines what kinds
of failure cascade are even possible.

---

## The three levels

| Arrangement | Topology | Cohomology | What it eliminates |
| --- | --- | --- | --- |
| No netting (gross exposures) | Bare graph | — | nothing |
| Bilateral netting | Fewer edges | Reduces $\beta_1$ | Edge-level loops |
| CCP / multilateral clearing | Filled triangles | Annihilates $H^1$ generators | Loop as topological class |

The distinction matters because *topology determines what can go wrong*, not just
*how much* is at risk.

---

## Level 0: gross exposures

Start with four institutions $A, B, C, D$ and a set of gross obligations:

```text
A → B  (A owes B)
B → C  (B owes C)
C → A  (C owes A)
C → D
D → A
```

$A \to B$ means A has a gross obligation to B. The arrow runs from obligor to
creditor. These are **gross** exposures before any netting. This is the natural
setting for repo and securities-lending markets, where rehypothecation means gross
positions can be many times net positions — and a net position of zero can still
sit inside a large funding loop that matters for systemic risk.

The network has $n=4$ nodes, $m=5$ edges, $c=1$ component, so
$\beta_1 = m - n + c = 2$: exactly two independent funding loops.

---

## The Eisenberg-Noe model: clearing as an H⁰ problem

The Eisenberg-Noe (2001) model asks: if some institutions cannot pay in full,
what is the *clearing payment vector* $p^*$ — the actual payments made, respecting
seniority and pro-rata rules?

The answer is a fixed point:

$$p_i^* = \min\!\left(\bar{p}_i,\; e_i + \sum_j \frac{L_{ji}}{\bar{p}_j} p_j^*\right)$$

where $\bar{p}_i$ is the total liability of institution $i$ and $e_i$ is its
external asset value.

**In cohomological language, this is an H⁰ computation.** The clearing vector
$p^*$ is the unique global section of the payments sheaf on the *1-skeleton*
(the graph of bilateral edges). EN uses only the edges — the bilateral
exposure matrix $L_{ij}$. When $p^*$ exists and is unique, the payments sheaf
has a consistent global section: $H^0$ is well-posed. EN's main theorem
(existence and uniqueness of $p^*$) is exactly a statement about $H^0$.

**What EN correctly captures:** direct, first-order contagion. If $A$ defaults,
$B$ loses exactly the unpaid portion of $L_{AB}$. The cascade propagates along
edges.

**What EN cannot see:**

| Failure mode | Why EN misses it |
| --- | --- |
| Funding freeze via shared repo counterparty | No bilateral edge between MMF and distant dealer — indirect channel is a triangle, not an edge |
| Basis blow-out (credit/rates correlation) | H¹ of the joint sheaf; no bilateral clearing can unwind it |
| AIG's CDS book: collateral calls created by the very protection it sold | Conflict cycle across four institutions — H²; EN's iteration does not converge uniquely |

EN is the right instrument for its level. The problem in 2008 was applying an
$H^0$ tool to a system whose failure was $H^1$ (funding loops) and $H^2$
(cross-desk correlation inconsistency).

---

## Bilateral netting: removing edges

A **bilateral netting agreement** (e.g. an ISDA Master Agreement between $A$ and $B$)
allows $A$ and $B$ to replace their gross claims on each other with a single net
position:

$$L_{AB}^{\mathrm{net}} = \max(L_{AB} - L_{BA},\; 0), \qquad L_{BA}^{\mathrm{net}} = \max(L_{BA} - L_{AB},\; 0)$$

If both owe each other comparable amounts, $L_{AB}^{\mathrm{net}} \approx 0$: the
edge effectively disappears from the graph.

**Topological effect: removes an edge, reduces $\beta_1$.**

If $A$ and $C$ net bilaterally in the example above, the edge $C \to A$
disappears. Loop 1 ($A \to B \to C \to A$) collapses — it can no longer be
traversed. $\beta_1$ drops from 2 to 1.

But bilateral netting between $A$ and $C$ does nothing about:

- The $C \to D \to A$ loop (Loop 2) — it remains
- Any triangle involving $B$ — still present
- Any default correlation between $B$ and $C$ — invisible to netting

Bilateral netting reduces edge count. It does not change the *dimension* of the
simplicial complex. The highest non-degenerate cells are still edges (1-simplices).
No triangle is filled in.

---

## When do we fill in a triangle?

A triangle $A$-$B$-$C$ is **filled in** (added as a 2-simplex) when
$A$, $B$, and $C$ enter a genuinely trilateral arrangement — one where the
commitment runs to the *triple*, not just to each bilateral pair.

**The test:** does closing out $A$'s position automatically and simultaneously
affect $B$'s and $C$'s positions, without any further bilateral negotiation?
If yes, the triangle is filled in.

Three examples:

### 1. CCP novation

A central counterparty (CCP) novates three bilateral trades: $A$-$B$, $B$-$C$,
$C$-$A$. After novation, the CCP is buyer to every seller and seller to every
buyer. The three bilateral edges are replaced by three edges to a central node
(the CCP), and the *original triangle* $A$-$B$-$C$ is filled in as a 2-simplex.

The loop $A \to B \to C \to A$, which was a free generator of $H^1$, is now the
*boundary* of the filled triangle. A boundary is not a free cycle — it bounds
something. So the loop is annihilated as an $H^1$ generator. $\beta_1$ drops by 1.

This is stronger than bilateral netting: CCP novation moves the cycle from $H^1$
into the *image of the boundary map* $\partial_2$. Bilateral netting removes an
edge (changes the graph); CCP novation changes the *homological class* of the cycle.

Concretely: CCP novation gives multilateral default protection that bilateral
netting does not. If $A$ defaults under bilateral netting, $B$ and $C$ have
separate close-out processes that may not coordinate. Under CCP novation, a
single waterfall (initial margin, default fund, CCP equity) covers all three
simultaneously.

### 2. Trilateral repo agreement

Three dealers agree to a tri-party repo with a custodian bank as agent. The custodian
holds the collateral and manages substitution rights for all three simultaneously.
This fills in the triangle: the custodian's simultaneous management of all three legs
makes it a 2-simplex, not three separate bilateral 1-simplices.

### 3. Cross-margin agreement across three products

An exchange allows margin offsets across equity futures ($A$), equity options ($B$),
and index futures ($C$). The offset is computed at the portfolio level — not pairwise.
This fills in the product-type triangle and annihilates the basis-risk loop as an
$H^1$ generator.

---

## When do we fill in a tetrahedron?

A tetrahedron $A$-$B$-$C$-$D$ is **filled in** (added as a 3-simplex) when
all four institutions enter a genuinely quadrilateral arrangement — one where
the commitment runs to the *quadruple* simultaneously.

In practice, this almost never happens in finance. Four-party simultaneous
commitments are rare: CCPs clear pairs or triples; cross-margining covers
product pairs or triples at a single venue. This is precisely why $H^2$
obstructions persist in the financial system — because the resolution instruments
(bilateral netting, CCPs) only go up to the triangle level.

**The hollow tetrahedron is the default state of any four-institution network.**
The four triangular faces $(A,B,C)$, $(A,B,D)$, $(A,C,D)$, $(B,C,D)$ each carry
their own pricing or risk residual — an $H^1$ class computed on that face. Genuine
$H^2$ is a statement about whether those four face-level residuals are mutually
consistent. If they were all computed from a single model, they automatically satisfy

```text
c_BCD − c_ACD + c_ABD − c_ABC = 0
```

This is the boundary identity $\delta^2 \circ \delta^1 = 0$ (Pentagon identity in
its simplest form). A single source, however many triangles, cannot generate $H^2$.

The $H^2$ obstruction appears when the four face residuals come from *more than one
source* — different desks, different models, different jurisdictions — that each
produce internally consistent triangles but whose triangle-level residuals fail to
close around the tetrahedron.

This is precisely the structure of the 2008 correlation desk failure: each desk's
own triangle looked fine; the four overlapping triangles did not close.

---

## The rule of thumb

| Fill in when... | Cohomological effect |
| --- | --- |
| $k+1$ parties enter a simultaneous joint arrangement | Adds a $k$-simplex |
| Bilateral netting (2-party) | Removes an edge (reduces $m$, reduces $\beta_1$) |
| CCP novation (3-party simultaneous) | Fills a triangle (annihilates $H^1$ generator) |
| Hypothetical 4-party simultaneous resolution | Fills a tetrahedron (would annihilate $H^2$ generator) |
| Government bailout of AIG / Lehman | The only actual $H^2$ resolution in 2008 — no private 4-party mechanism existed |

The absence of a filling is not just an operational gap — it is a structural
constraint on what resolution mechanisms exist. An $H^2$ obstruction cannot be
resolved by any combination of bilateral agreements or CCPs. It requires an
instrument that acts at the tetrahedron level: a central bank, a resolution
authority, or (in Paper 426's framework) a $K^{\text{top}}$ charge that prices
the irresolvability before the fact.

---

## What the topology tells the regulator

| Question | Classical answer | Topological answer |
| --- | --- | --- |
| Is this institution safe? | Net exposure, VaR | $H^0$: consistent global section of payments sheaf |
| Is this funding loop dangerous? | Count the cycles | $H^1$: $\beta_1$ independent funding loops; netting or CCP reduces this |
| Is this crisis resolvable bilaterally? | Workout, ISDA close-out | $H^2 = 0$? If not, no bilateral resolution exists |
| What would it take to resolve it? | Government intervention | Fill the tetrahedron: act at the 4-party level |

The 2008 crisis was a sequence of $H^1$ loops (repo, commercial paper)
cascading into an $H^2$ obstruction (AIG, Lehman prime brokerage) that
only a government-level instrument could resolve.
No amount of bilateral netting or CCP novation of triangles resolves an $H^2$
obstruction — because the obstruction lives one dimension higher than any
available instrument.

---

## Further reading

- Eisenberg, L. & Noe, T.H. (2001). Systemic Risk in Financial Systems. *Management Science* 47(2).
- Buckley (2026). Paper 397: Systemic Risk as H². doi:10.5281/zenodo.20642908
- Buckley (2026). Paper 426: The Cohomological Regulator. doi:TBD
- Buckley (2026). Paper 430: The Topology of Intermediation. doi:10.5281/zenodo.20694463
- Mehrling, P. (2011). *The New Lombard Street*. Princeton University Press.
