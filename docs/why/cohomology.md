# Bilateral · Triangular · Systemic Risk

> *"Just because I don't know what the connection is doesn't mean there isn't one."*
> — Douglas Adams, *The Long Dark Tea-Time of the Soul*

Financial risk has three structural levels. Existing tools address the first two.
EconIAC computes all three.

| Level | Name | What it is | Instruments | Who manages |
| --- | --- | --- | --- | --- |
| $H^0$ | **Bilateral risk** | Consistency of bilateral prices | Forwards, swaps | Trading desks |
| $H^1$ | **Triangular risk** | Unhedgeable residual; convexity, basis, XVA | Options, swaptions | XVA desks |
| $H^2$ | **Systemic risk** | Mutual inconsistency of triangular risks; cascade | CCPs, central banks | Regulators, CRO |

A risk is hedgeable with bilateral instruments if and only if its $H^1$ class is trivial.
The 2008 crisis was an $H^2$ event. No regulator was computing $H^2$.

**Start here:** [The Topology of Risk — a plain-language primer](https://doi.org/10.5281/zenodo.20642983)
(13 pages, no prior mathematics required).

---

## Why does EconIAC measure inconsistency topologically?

*You have a network of banks, lenders, or trading venues. Each holds a local piece
of information: a capital ratio, a funding assessment, a price. The question is not
"how far are these from the truth?" The question is: "do they fit together at all?"
Those are different questions, and only the second one predicts crises.*

---

## The residual paradigm and what it misses

Standard econometric and risk models work like this:

1. Specify a model M with parameters θ
2. Fit θ to minimise the sum of squared residuals: Σ (data − M(data; θ))²
3. Report R², standard errors, residuals as diagnostics
4. Call the residual "noise" — measurement error or model misspecification

This works well for isolated agents. It fails for networked systems in a specific way.

Consider a repo market under stress. MMF funds assess dealer A as solvent and roll
their repo. LDI pension funds assess the same dealer A as insolvent and withdraw.
Both may be *individually* consistent with the data available to them. A model fitted
to aggregate data — total repo rolled, average haircut — may have R²=1. But the
*network* is in an irreconcilable state: two agents acting on mutually inconsistent
assessments of the same counterparty, simultaneously.

No improvement in model fit resolves this. The inconsistency is not measurement error.
It is a structural property of the network state at that moment — and it is what causes
the cascade.

---

## The topological alternative: H¹ cohomology

EconIAC measures network inconsistency using **H¹ cohomology** of a cellular sheaf on
the network graph.

The construction has three components:

**Stalk**: at each node (bank, dealer, venue), a vector space holding the local
information — the capital ratio, the funding assessment, the asset price.

**Restriction map**: on each edge (bilateral exposure, lending relationship, arbitrage
link), a linear map specifying *how much the information at one node should agree with
the information at the adjacent node*, weighted by the strength of the relationship.

**Coboundary operator δ₀**: maps the section (the collection of all local values) to
the space of edge disagreements. δ₀ applied to a globally consistent section gives
zero. Applied to an inconsistent section it gives a non-zero vector of disagreements.

The **H¹ signal** is then:

```
H¹ = ‖L_F · s‖² / ‖s‖²
```

where L_F = δ₀ᵀδ₀ is the sheaf Laplacian and s is the section.

- **H¹ = 0**: the local pieces fit together into a globally consistent picture.
  The network is in a reconcilable state.
- **H¹ > 0**: the local pieces cannot be globally reconciled. The network is in a
  state of topological inconsistency. The magnitude measures how far it is from
  reconcilability.

---

## Worked example: independent vs. dependent funding loops

Paper 426 measures network fragility with a related but distinct
quantity: $\beta_1$, the number of *independent funding loops*, defined
as the first Betti number of the exposure graph,
$\beta_1 = m - n + c$ (edges minus nodes plus connected components).
"Independent" is doing real work in that phrase, and it does not mean
what it sounds like on first read — it is not about whether two loops
look different, or share edges, or share no edges. It is about whether
one loop can be written as a sum of others plus a *boundary* (a trivial
back-and-forth contribution that adds nothing new).

Take four institutions $A, B, C, D$ with obligations:

```text
A → B,  B → C,  C → A,  C → D,  D → A
```

Here $n=4$ nodes, $m=5$ edges, $c=1$ component, so
$\beta_1 = 5 - 4 + 1 = 2$: the network has exactly two independent
funding loops, not five (the number of edges) and not one (the number
of "obvious" triangles).

Three loops are visible by eye:

- **Loop 1** ($A\to B\to C\to A$): the triangle through $B$.
- **Loop 2** ($C\to D\to A\to C$, i.e. traversing $C\to A$ backwards):
  the triangle through $D$.
- **Loop 3** ($A\to B\to C\to D\to A$): the quadrilateral around the
  outside.

$\beta_1=2$ says only two of these three are independent — and indeed,
Loop 3 is exactly **Loop 1 + Loop 2**: travelling $A\to B\to C$ (shared
with Loop 1), then $C \to D \to A$ (shared with Loop 2), is the same
net obligation pattern as doing Loop 1 and Loop 2 separately and
cancelling the shared $C\to A$ / $A\to C$ leg. Loop 3 is therefore
**dependent** on Loops 1 and 2 — it adds no new systemic fragility
beyond what Loops 1 and 2 already represent, even though it is a
perfectly real, traceable cycle of obligations. A regulator who counted
"how many cycles can I find by inspection" would see three and
overstate the network's fragility; $\beta_1$ correctly reports two.

This is the homological content of "financial redundancy": a loop adds
systemic fragility only if it cannot be re-expressed as a combination
of other loops already counted. Loops 1 and 2, by contrast, share no
edges and have no such relation — both are needed to generate the full
cycle space, so both count.

Direction matters only through sign, not through which way arrows
point. Loop 2 above was written traversing $C\to A$ *backwards*
(against its stated direction, $A$ owes $C$) precisely because the
loop closes that way — in the signed, ℝ-coefficient chain complex this
is legitimate: the edge contributes with a $-1$ coefficient instead of
$+1$, and the cycle still closes (its boundary is still zero). A loop
that runs against an edge's stated direction is not a different kind of
object from one that runs with it, and counts toward $\beta_1$ in
exactly the same way — see
[β₁](../glossary.md#β₁-independent-funding-loops) for why edge
orientation is a sign convention, not a separate cycle-counting rule.

---

## What makes this different from a residual

The conceptual gap is subtle but load-bearing:

| | Residual | H¹ cohomology |
| --- | --- | --- |
| **Requires a model** | Yes — residual is (data − model prediction) | No — only requires the network and a consistency relation |
| **Node or edge property** | Node property (individual deviation) | Edge property (bilateral disagreement) |
| **Zero means** | Perfect model fit | Globally reconcilable state |
| **Non-zero means** | Model misspecification or noise | Irreconcilable local assessments |
| **Symmetric?** | Yes — overfit and underfit penalised equally | No — H¹ is a topological invariant |
| **Predictive?** | Contemporaneous | Leads the cascade by 1–3 periods |

The last row is the key result (Theorem 1 of Paper 335):

> **In any Gibbs-lifted contagion model, H¹ peaks strictly before the cascade peaks.**

The intuition: the Gibbs lift creates a smooth sigmoid around the hard threshold. The
bilateral inconsistency — agents beginning to disagree on the same counterparty's
creditworthiness — starts accumulating as soon as the active region of the sigmoid is
entered, before any individual threshold is individually breached. H¹ detects this
accumulation; the cascade count only fires when the threshold is crossed.

In the CHZ fire-sale model (Paper 332) and the sovereign repo model (Paper 333), the
empirical lead time is 2–3 periods.

---

## The residual-obstruction separation

There is a stronger result (Theorem 2 of Paper 335):

> **A model can achieve R²=1 and H¹ ≠ 0 simultaneously.**

Proof by construction: build a repo market model that correctly predicts total funding
rolled (aggregate prediction = data, zero residual), but assigns the rolled funding
across the dealer-lender bilateral pairs in a way that is inconsistent with the
collateral coverage ratios. The residual is zero; H¹ is non-zero.

This means H¹ is not computable from a model's residuals. It measures something
the standard toolkit cannot see — and something that is directly relevant to whether
a cascade is imminent.

---

## Why this is universal

The same H¹ signal appears in three apparently unrelated systems in EconIAC:

| System | Graph | Section | H¹ measures |
| --- | --- | --- | --- |
| CHZ interbank contagion (Paper 332) | Interbank exposure | Capital ratios | Bilateral solvency disagreement |
| Sovereign repo run (Paper 333) | Dealer-lender bipartite | Funding ratios | Roll probability inconsistency |
| FMO energy transfer (Paper 325) | Chromophore Fano graph | Energy efficiency | Broken topological symmetry |

These are not analogies. The computation is identical on three different graphs.

The reason is Theorem 3 of Paper 335: the H¹ signal depends only on the graph topology
and the restriction maps, not on the specific dynamics. Fire sales, repo runs, and
quantum energy transfer are different mechanisms, but they all produce the same
topological signal because they all involve the same structural question: *can the
local pieces be globally reconciled?*

---

## What H¹ requires — and what it does not

**Requires**:
- The bilateral exposure matrix (which agent is exposed to which, at what weight)
- The current section values (capital ratios, funding ratios, prices)
- A consistency relation on each edge (how much adjacent nodes should agree)

**Does not require**:
- A model of the true state
- Assumptions about the distribution of errors
- Parameter estimation
- A prediction of what the section *should* be

This is what makes H¹ a genuinely different instrument from model-based risk measures.
VaR, Expected Shortfall, and eigenvalue-based fragility measures (Acemoglu et al. 2015)
all require a model. H¹ requires only the network and the current state.

---

## Regulatory implications

If H¹ is a reliable 2–3 period leading indicator for cascades, and if it requires
only bilateral exposure data (already reported to regulators via EMIR, FSB-LEI, and
ECB repo statistics), then:

1. A real-time H¹ monitor is computationally feasible — it is a matrix-vector product
   on the bilateral exposure matrix, updated at each reporting period.

2. The FSB's current early-warning toolkit (based on haircut levels, concentration
   ratios, and VaR) could be supplemented with an H¹ signal that detects bilateral
   inconsistency before any individual threshold is breached.

3. Optimal haircut calibration can target H¹ = 0 rather than minimising systemic
   loss after the fact — the policy gradient ∂H¹/∂h gives the regulator the cheapest
   intervention to restore global reconcilability.

---

## Further reading

- Robinson, M. (2014). *Topological Signal Processing*. Springer.
  The foundational treatment of H¹ as a distributed inconsistency measure.

- Hansen, J. & Ghrist, R. (2021). Opinion Dynamics on Discourse Sheaves.
  *SIAM J. Appl. Math.* 81(5), 2033–2060.
  The direct mathematical predecessor — agents with local opinions on a graph.

- Acemoglu, D., Ozdaglar, A. & Tahbaz-Salehi, A. (2015). Systemic Risk and Stability
  in Financial Networks. *AER* 105(2), 564–608.
  The eigenvalue-based fragility measure that H¹ supersedes for early-warning purposes.

- Buckley (2026). Paper 335: Topological Inconsistency. doi:TBD
- Buckley (2026). Paper 332: CHZ Fire Sales. doi:TBD
- Buckley (2026). Paper 333: Sovereign Repo Run. doi:TBD
- Buckley (2026). Paper 325: FMO Topological Heat Engine. doi:10.5281/zenodo.20400638
