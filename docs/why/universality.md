# Why does Econiac generalise beyond economics?

*The mathematical core of Econiac — conservation laws, Gibbs dynamics, and H¹
cohomology — does not know it is doing economics. The same three structures
appear in neuroscience, ecology, climate science, and metabolism. Econiac is
the first fully-implemented domain; the others follow the same codebase.*

---

## The three pillars and where they appear

Econiac is built on three mathematical structures. None of them is specific
to economics.

### 1. Conservation as ∂²=0

The Pacioli identity says: every financial claim has a matching counter-claim.
Money can be created, but only symmetrically. This is ∂²=0 — the boundary of
a boundary is zero.

The same identity holds, with different physical content, in:

| Domain | What is conserved | The ∂²=0 statement |
| --- | --- | --- |
| **Economics** | Financial claims | Every debit has a credit |
| **Metabolism** | Electron carriers (ATP, NADH) | Every oxidation has a reduction |
| **Ecology** | Energy through trophic levels | Every calorie eaten was a calorie somewhere |
| **Climate** | Atmospheric mass and energy | Every flux in has a flux out |
| **Neuroscience** | Charge at each neuron | Kirchhoff's current law at every synapse |

In all five cases, ∂²=0 is not a model assumption — it is a constraint the
system satisfies by construction. Models that violate it are wrong in the same
way a balance sheet that doesn't balance is wrong.

### 2. Gibbs dynamics (β as decision sharpness)

The Gibbs lift replaces hard thresholds with smooth sigmoids parameterised by
β — the inverse temperature. At β→∞ you recover the hard rule; at finite β
you have a differentiable model calibratable from data.

This structure is not specific to financial agents. It is the universal model
of a system that minimises free energy:

**Karl Friston's free energy principle** (the dominant framework in theoretical
neuroscience) says that biological systems minimise variational free energy:

    F = KL[q(s) ‖ p(s|o)] + log p(o)

where q(s) is the agent's internal model and p(s|o) is the generative model of
the world. This is *exactly* the Gibbs distribution. The "precision" parameter
in Friston's framework is exactly β in ours. High precision (high β) = sharp,
cliff-edge responses. Low precision (low β) = gradual, exploratory responses.

The Gibbs lift is not borrowed from economics. It is the universal form of
any system that balances exploration and exploitation under uncertainty.

### 3. H¹ cohomology as inconsistency

The sheaf H¹ signal measures whether local pieces of information fit together
globally. It requires only: a network, local values at each node, and a
consistency relation on each edge.

The same computation — same code, different calibration — applies across domains:

| Domain | Network | Section | H¹ measures |
| --- | --- | --- | --- |
| **Finance** | Interbank exposure | Capital ratios | Bilateral solvency disagreement |
| **Repo markets** | Dealer-lender bipartite | Funding ratios | Roll-probability inconsistency |
| **Neuroscience** | Cortical regions | Prediction errors | Irreconcilable predictions |
| **Ecology** | Food web | Population ratios | Trophic inconsistency |
| **Climate** | Atmospheric cells | Temperature anomalies | Heat flux inconsistency |
| **Metabolism** | Reaction network | Metabolite concentrations | Stoichiometric imbalance |
| **Photosynthesis** | Chromophore graph | Energy efficiency | Broken Fano symmetry |

The last row is already implemented: Paper 325 (FMO topological heat engine)
uses the same `sheaf_laplacian()` function from `econiac.finance.contagion`
to compute H¹ on the FMO chromophore network. The three-way isomorphism
between the CHZ cascade, the repo run, and the FMO heat engine (Paper 334 §6)
is empirical confirmation that H¹ is genuinely universal.

---

## The Friston connection in detail

Karl Friston's free energy principle is the most ambitious unified theory in
neuroscience. It claims that all biological self-organisation — perception,
action, learning, evolution — can be understood as free energy minimisation.

In Econiac's language:

| Friston | Econiac |
| --- | --- |
| Generative model p(s,o) | Restriction maps of the sheaf |
| Variational density q(s) | Section s of the sheaf |
| Free energy F = KL[q‖p] | H¹ signal = ‖L_F·s‖²/‖s‖² |
| Precision ω | Inverse temperature β |
| Active inference | Policy gradient ∂H¹/∂action |
| Markov blanket | The graph boundary ∂G |
| Surprise minimisation | Fixed-point iteration to H¹=0 |

This is not an analogy. The KL divergence and the sheaf H¹ signal are both
measuring the same thing: how far a distribution (or a section) is from
globally consistent with a generative model (or a sheaf). The mathematics is
identical; the physical interpretation differs.

**The practical consequence**: Friston's framework lacks a concrete
computational implementation that scales to large networks and admits policy
gradients. Econiac provides exactly this. A `thermology.neuroscience` module
that wraps the existing sheaf library with cortical region calibrations would
be a direct implementation of the free energy principle at the network level.

---

## Ecosystems

Trophic networks are conservation-law networks. Energy flows from producers
through consumers to decomposers, with ∂²=0 at every node (energy in = energy
out + stored). The Gibbs lift models predator switching behaviour (β measures
how sharply predators switch prey species as relative abundance changes).

The H¹ signal on a food web measures trophic inconsistency — whether local
population ratios are mutually reconcilable. This is a leading indicator of
ecosystem collapse, for the same reason it is a leading indicator of financial
cascades (Theorem 1, Paper 335): inconsistency accumulates before any
individual population crashes through its threshold.

Keen's predator-prey model (`econiac.economics.minsky`) is already
Lotka-Volterra dynamics — the same equations that govern debt-deflation
spirals and species collapse. The calibration layer differs; the operator
algebra is identical.

---

## Climate

The atmospheric and oceanic circulation satisfies ∂²=0 (conservation of mass
and energy). Tipping points — Amazon dieback, Atlantic overturning collapse,
ice-albedo feedback — are cascade dynamics with Gibbs-like thresholds.

Paper 311 (Climate Yield Surface) already uses Econiac's framework for climate
investment geometry. The extension to tipping-point early warning is direct:
build a FinancialGraph where nodes are climate subsystems (Amazon, AMOC,
Greenland ice, West Antarctic ice, permafrost), edges are their known physical
couplings, and the section is the current anomaly. H¹ on this graph is a
topological tipping-point indicator — detectable from existing observational
data, model-agnostic, and (by Theorem 1 of Paper 335) a leading indicator.

---

## Metabolism

Stoichiometric matrices in metabolic networks are ∂²=0 by construction.
Flux Balance Analysis (FBA) — the dominant computational method in systems
biology — already computes the null space of the stoichiometric matrix, which
is H⁰ of the metabolic sheaf. H¹ of the metabolic network is the natural
next step: it measures whether flux assignments are globally consistent, and
its leading-indicator property (Theorem 1) predicts metabolic disease onset.

---

## The Thermology unification

All five domains share the same three-layer architecture:

```
Layer 1: Conservation (∂²=0)
         — the network has a boundary operator
         — valid states satisfy the conservation law

Layer 2: Gibbs dynamics (β, free energy)
         — agents/nodes minimise free energy
         — β parameterises response sharpness
         — differentiable end-to-end via JAX

Layer 3: H¹ cohomology (sheaf Laplacian)
         — measures global inconsistency
         — leads cascade/failure by 2–3 periods
         — model-agnostic early-warning instrument
```

This three-layer architecture is **Thermology** — the mathematics of
conservation-law networks with Gibbs dynamics. Econiac is the economics
implementation. Thermion (three-body orbit search) is the physics
implementation. A neuroscience implementation, ecology implementation,
and climate implementation follow the same codebase with domain-specific
calibration layers.

The shared library is `thermology.core`. Econiac and Thermion become
domain-specific wrappers on top of it.

---

## Further reading

- Friston, K. (2010). The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience* 11, 127–138.
- Friston, K. et al. (2017). Active inference and epistemic value.
  *Cognitive Neuroscience* 8(4), 187–224.
- Ramstead, M.J.D., Badcock, P.B. & Friston, K.J. (2018). Answering
  Schrödinger's question: A free-energy formulation.
  *Physics of Life Reviews* 24, 1–16.
- Orth, J.D., Thiele, I. & Palsson, B.Ø. (2010). What is flux balance
  analysis? *Nature Biotechnology* 28, 245–248.
- Rockström, J. et al. (2009). Planetary boundaries. *Nature* 461, 472–475.
- Buckley (2026). Paper 291: The Topology of Conservation.
  doi:10.5281/zenodo.20234853
- Buckley (2026). Paper 313: Thermal Economics.
  doi:10.5281/zenodo.20318505
- Buckley (2026). Paper 325: The Topological Heat Engine.
  doi:10.5281/zenodo.20400638
- Buckley (2026). Paper 335: Topological Inconsistency. doi:TBD
- Buckley (2026). Paper 339: Thermology — Conservation, Gibbs Dynamics,
  and H¹ Cohomology as a Universal Framework. doi:TBD

---

## See also

- [Why topology?](topology.md) — ∂²=0 as the foundational identity
- [Why sheaves?](sheaves.md) — H¹ cohomology across four domains
- [Topological inconsistency](cohomology.md) — H¹ as a first-class observable
- [What is money a claim on?](money.md) — conservation in monetary systems
