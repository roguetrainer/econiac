# Why does Econiac use abstract mathematics?

econiac is built on concepts borrowed from physics and pure mathematics that most economists and quants will not have encountered before. This is not gratuitous abstraction — each tool solves a specific problem that the standard toolkit cannot.

| Concept | The problem it solves | Read |
| --- | --- | --- |
| **Thermodynamics** | How do heterogeneous agents aggregate? The Gibbs distribution is the only honest answer. | [Why thermodynamics?](thermodynamics.md) |
| **Topology** | Double-entry bookkeeping is ∂²=0 — a theorem, not a convention. Models that violate it are wrong by construction. | [Why topology?](topology.md) |
| **Differential geometry** | Exchange rates multiply. Multiplicative composition along paths is geometry. Curvature is arbitrage. | [Why geometry?](geometry.md) |
| **Stock-flow consistency** | Every pound exists on two balance sheets. Crises are stock problems; models without stocks cannot detect them. | [Why stock-flow consistency?](stock_flow_consistency.md) |
| **Gauge theory** | The unit of account is a choice. Models that give different answers in euros vs. dollars have a bug. Gauge invariance fixes it. | [Why gauge theory?](gauge_theory.md) |
| **Non-associative algebra** | Policy interventions don't associate: (A then B) then C ≠ A then (B then C) when transmission is state-dependent. | [Why non-associative algebra?](non_associative_algebra.md) |
| **Combinator library** | Conservation is the substrate, not the point. The Pacioli Combinator Library gives you the components — flows, switches, mixers — that determine what a financial model actually does. | [Why a combinator library?](combinators.md) |
| **Connections and curvature** | The Pacioli identity (∂²=0) says every financial claim has a matching counter-claim — money can be created by banks, but only by simultaneously creating a liability. Flatness is the stronger claim that the path doesn't matter. Only ∂²=0 is always true. Non-flat connections handle discrepancies, float, and arbitrage without abandoning it. | [Why connections and curvature?](curvature.md) |
| **What is money a claim on?** | Under commodity money the answer is gold. Under credit money it is a loan portfolio. Under fiat money it is the continued functioning of the monetary network itself — a topological property, not a balance-sheet entry. The three regimes are three gauge fixings of the unit of account. | [What is money a claim on?](money.md) |
| **Sheaves** | Local information may not fit together globally. Sheaves measure that gap — in contagion networks, market signals, neural networks, and biological energy transfer. | [Why sheaves?](sheaves.md) |
| **Sheaves for contagion** | Why the contagion library uses sheaf Laplacians, Gibbs lifts, and adjoint operators instead of just writing down the fire-sale equations directly. | [Why these abstractions for contagion?](abstractions_for_contagion.md) |
| **Topological inconsistency** | Residuals measure deviation from a model. H¹ cohomology measures whether local data is mutually reconcilable at all — without any model. H² measures whether the H¹ residuals on the faces of a tetrahedron are mutually consistent with each other. These are different instruments, and only they predict cascades. | [Why measure inconsistency topologically?](cohomology.md) |
| **Universality across domains** | The three pillars of Econiac — ∂²=0, Gibbs dynamics, H¹ cohomology — appear identically in neuroscience (Friston free energy), ecology, climate tipping points, and metabolism. Econiac is the economics instance of a universal framework called Thermology. | [Why does Econiac generalise beyond economics?](universality.md) |

Each page is written for an economist or quant — it starts from your existing problem, not from the mathematics.
