# Why does Econiac use topology?

*Topology sounds like abstract mathematics with no practical content. In fact, you have been using it for seven centuries without knowing its name.*

---

## Double-entry bookkeeping is a topological theorem

In 1494, Luca Pacioli wrote down the rules of double-entry bookkeeping: every transaction has a debit and a credit; the books must balance. Accountants treat this as a convention — a discipline imposed on record-keepers to prevent errors. It is not. It is a theorem.

The theorem is this: in any directed graph of flows, the boundary of a boundary is zero.

```
∂² = 0
```

If money flows from A to B (an edge in the graph), and from B to C (another edge), then the net flow *out of* the path A→B→C is zero: what leaves A arrives at C, and B is merely a waypoint. The debit-credit balance is not a rule that accountants enforce — it is a consequence of the mathematics of directed graphs. You cannot violate it any more than you can have a road that starts somewhere but goes nowhere.

This is topology: the study of properties that are preserved under continuous deformation. The property ∂²=0 is preserved under any rearrangement of the flow network. It is the reason accounting works.

## What breaks when you ignore it

Standard macroeconomic models — DSGE models in particular — routinely violate stock-flow consistency. A household sector can accumulate net financial assets without a corresponding net liability appearing somewhere else. This is not a modelling choice; it is an error. The model is describing a flow network where ∂² ≠ 0 — a mathematical impossibility in the real economy.

The consequences are real. Godley and Lavoie (2007) showed that standard New Keynesian models implicitly assume the government sector absorbs unlimited net liabilities with no feedback — a violation of the budget constraint that only shows up in multi-period simulations. The 2008 crisis was partly a failure of models that violated stock-flow consistency to detect the accumulating imbalances in the household and financial sectors.

econiac enforces ∂²=0 by construction. It is not a constraint you add — it is the definition of the Pacioli manifold. A model that violates it is a type error, caught before runtime.

## Becker's theorem: the demand law is topological too

In 1962, Gary Becker published a short paper in the *Journal of Political Economy* with a claim that most economists found unsettling: **downward-sloping demand does not require rational agents**. An agent who spends their income at random — no preferences, no utility maximisation, no cognition — still produces a demand curve with negative slope.

The argument is geometric. The budget set is a simplex: a triangle in (q₁, q₂) space whose vertices are at (M/p₁, 0), (0, M/p₂), and (0, 0). A uniform random agent samples from this triangle. The expected quantity of good 1 is the centroid of the triangle along the q₁ axis:

$$\mathbb{E}[q_1] = \frac{M}{3p_1}$$

Raise p₁ and the triangle shrinks along the q₁ axis. The centroid moves left. The demand curve slopes downward — with no utility function, no preferences, no rational calculation. The **budget constraint is the only assumption**.

This is a topological statement: the demand law is an invariant of the budget simplex. It is preserved under any deformation of the flow network that preserves the budget constraint. Rationality is sufficient for the demand law but not necessary. The simplex geometry is what is necessary.

EconIAC makes this precise. The Gibbs ensemble on the budget simplex has β as its temperature. At β→0, agents are pure noise and the demand law holds by Becker's geometric argument. At β→∞, agents are fully rational and the demand law holds by utility maximisation. At every β in between, the demand law holds — because it is a topological invariant, not a behavioural one.

The β parameter tells you *how sharply* the demand law holds, not *whether* it holds. That sharpness is calibratable from data — from the observed cross-sectional variance of household expenditure shares. This is what the [budget simplex tutorial](../tutorials/budget_simplex_becker.ipynb) demonstrates.

Becker's insight, in retrospect, was that he had found a topological invariant of consumer theory without having the language to say so. EconIAC provides that language.

## Homology: what the topology tells you

Once you accept that the economy is a directed graph with ∂²=0, topology gives you three diagnostic numbers for free:

- **H₀** (zeroth homology): the number of disconnected components. If H₀ > 1, you have isolated sectors with no financial connection to the rest of the economy — a model pathology or a real structural feature worth investigating.

- **H₁** (first homology): the number of independent financial cycles. Each independent loop in the flow graph corresponds to a circuit that can carry a persistent imbalance — a credit cycle, a carry trade, a rollover risk. H₁ counts how many such circuits exist in your model.

- **H₂** (second homology): obstructions to global consistency. A non-zero H₂ means there is a local consistency condition that cannot be satisfied globally — the economic equivalent of a Möbius strip.

These numbers are computed automatically by econiac. They tell you things about your model's structure that inspection of the equations alone cannot.

## Topology is not geometry

A common confusion: topology is not the same as geometry. Geometry cares about distances and angles; topology cares only about connectivity — which things are connected to which. Two flow networks with completely different exchange rates, sector sizes, and time horizons can have identical topology (same H₀, H₁, H₂) and therefore the same structural vulnerabilities.

This matters for stress testing. A stress test that changes exchange rates and sector sizes while preserving the topology is testing a *quantitative* shock. A stress test that changes the topology — adds or removes financial connections, opens or closes circuits — is testing a *structural* shock. econiac distinguishes these; most frameworks do not.

## The short version

You need topology because:

1. Double-entry bookkeeping is ∂²=0 — a topological theorem, not an accounting convention. econiac enforces it automatically.
2. Models that violate stock-flow consistency are topologically inconsistent. econiac makes such models type errors.
3. The homology groups H₀, H₁, H₂ give you structural diagnostics — disconnected sectors, financial cycles, global inconsistencies — that you cannot read off from the equations alone.

The topology of the Pacioli manifold is the reason the books balance. Everything else in econiac is built on top of it.
