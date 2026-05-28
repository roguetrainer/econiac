# Why does Econiac use sheaves?

*Sheaves sound like advanced algebraic topology. In practice they solve one
ubiquitous problem: local information that may not fit together globally.
That problem appears in at least four distinct places in Econiac.*

---

## The one problem sheaves solve

Imagine you have a network of agents — banks, markets, proteins, neurons —
and each agent holds a piece of local information: a capital ratio, a price,
an energy level. The question a sheaf answers is:

> **"Do all the local pieces of information fit together into a consistent
> global picture?"**

If they do, the system is in a coherent state. If they do not, there is a
measurable obstruction — a "gap" between what different parts of the network
believe. That gap is what the sheaf cohomology group H¹ measures.

This is not an abstract concern. Every use case below is a situation where
local consistency fails in an economically or scientifically important way,
and where detecting the failure early matters.

---

## The four uses of sheaves in Econiac

### 1. Contagion early-warning (current)

**File**: `econiac/finance/contagion/sheaf.py`

**The question**: Do different parts of the interbank or repo network agree
on which dealers are solvent / which repos are safe?

**The section**: Capital ratios (Paper 332) or funding ratios (Paper 333).

**What H¹ measures**: The extent to which bilateral assessments of solvency
are inconsistent across the network. When lenders disagree on the creditworthiness
of the same borrower — MMFs rolling while LDI funds withdraw — the H¹ signal
rises before any individual threshold is breached.

**Lead time**: 2–3 periods before distress in the CHZ and repo models.

**See also**: [Why these abstractions for contagion?](abstractions_for_contagion.md)

---

### 2. Sheaf neural networks (planned)

**The question**: Can a neural network's weights be made globally consistent
with a geometric constraint — a symmetry, a conservation law, a gauge invariance?

A standard neural network assigns weights to edges of a computation graph.
A sheaf neural network assigns a *vector space* (stalk) to each node and a
*linear map* (restriction map) to each edge. Gradient descent then optimises
not just the weights but the consistency of the stalks across the graph.

**What H¹ measures**: Whether the network's representations are globally
consistent with the imposed geometric structure. A network that has learned
a good representation of a symmetric physical system should have H¹ = 0
(the learned stalks form a globally consistent section).

**Why this matters for Econiac**: Financial networks have natural geometric
structures — gauge invariance of the unit of account, SFC conservation,
interest rate term structure constraints. A sheaf NN trained on such a network
should respect these structures, and H¹ tells us whether it does.

**Reference**: Hansen & Ghrist (2020), *Sheaf Neural Networks*.
arXiv:2012.06333.

---

### 3. Topological signal processing on market data (planned)

**The question**: Is a market microstructure signal globally consistent across
trading venues, time zones, and asset classes?

A price signal observed on exchange A and exchange B should, under the
no-arbitrage condition, be consistent: they should agree (up to bid-ask spread
and latency) on the same underlying value. When they do not, the inconsistency
is a trading signal — or a sign of market stress.

The sheaf encodes the expected consistency relationship between observations
at different nodes (venues, time-stamps, asset classes). H¹ detects when
the market is generating inconsistent signals — which precedes liquidity
crises in the same way it precedes solvency crises in the contagion model.

**Reference**: Robinson, M. (2014). *Topological Signal Processing*. Springer.

---

### 4. Biological network coherence (Papers 324, 325, 328)

**The question**: Is energy, information, or mechanical force being transferred
coherently across a biological network?

In the FMO complex (Paper 325), the chromophores form a graph. Energy
transferred along the Fano line (the "broken" edge connecting chromophores
1 and 7) creates a measurable H¹ obstruction. When the Fano symmetry is
intact, H¹ = 0 and efficiency is zero. When it is broken (Tyr16 coupling),
H¹ ≠ 0 and the heat engine runs.

The same construction appears in:

- Ribosomal decoding (Paper 324): tRNA accommodation as a sheaf consistency
  problem on the A-site network
- Microtubule CST (Paper 328): tubulin conformational states as sections
  of a sheaf on the protofilament lattice

**What H¹ measures in biology**: A topological obstruction to coherent global
function. H¹ ≠ 0 means the network cannot assign consistent local states
everywhere simultaneously — which is exactly the condition for directed,
asymmetric energy or information transfer.

---

## The shared mathematics

All four uses share the same computational core, now in `contagion/sheaf.py`:

```python
graph       = FinancialGraph (or BiologicalGraph, MarketGraph, ...)
section     = local information vector (capital ratios, energy levels, prices)
L_F         = sheaf_laplacian(graph, section)
signal      = h1_signal(L_F, section)       # scalar: 0 = consistent
decomp      = harmonic_decomposition(...)   # which nodes contribute most
```

The function names are application-agnostic. `FinancialGraph` is the current
concrete implementation; a `BiologicalGraph` and `MarketGraph` will follow
the same interface.

---

## Why one library, not four

The temptation is to write separate sheaf code for each application. The
reason not to:

1. **The mathematics is identical.** The coboundary operator, the Laplacian,
   the H¹ dimension count, and the harmonic decomposition are the same
   computation regardless of whether the nodes are banks, chromophores, or
   trading venues.

2. **Cross-model comparison becomes trivial.** `compare_h1_series(ts_finance,
   ts_biology)` in `sheaf.py` measures whether two H¹ time series are
   structurally isomorphic. Paper 334 §6 uses this to prove the three-way
   isomorphism between the CHZ cascade, the repo run, and the FMO heat engine.
   That result requires a single shared library.

3. **Bugs are fixed once.** The `harmonic_decomposition()` function had a
   subtle sign convention issue in an earlier draft. Fixing it in one place
   fixes it for all four applications.

---

## The deeper reason: sheaves are the right language for consistency

The recurring theme across all four uses is:

> Local information + a notion of "how local pieces should relate" + a
> measurement of how far reality departs from that relation.

That is precisely what a sheaf is: a functor from the category of open sets
of a space (or faces of a simplicial complex, or nodes and edges of a graph)
to the category of vector spaces, together with restriction maps saying how
the local pieces relate.

In all four Econiac applications, the "right relation" is determined by the
conservation law:

- Finance: Pacioli ∂²=0 (double-entry accounting)
- Biology: energy conservation on the chromophore graph
- Signal processing: no-arbitrage / no-energy-creation

The sheaf is what makes these conservation laws geometric rather than merely
algebraic — it gives them a spatial structure that H¹ can measure.

---

## Further reading

- Robinson, M. (2014). *Topological Signal Processing*. Springer.
- Hansen, J. & Ghrist, R. (2021). Opinion Dynamics on Discourse Sheaves.
  *SIAM J. Appl. Math.* 81(5), 2033–2060.
- Curry, J. (2014). Sheaves, Cosheaves, and Applications. arXiv:1303.3255.
- Ghrist, R. (2014). *Elementary Applied Topology*. CreateSpace. Ch. 6.
- Buckley (2026). Paper 325: FMO topological heat engine.
  doi:10.5281/zenodo.20400638
- Buckley (2026). Paper 334: Contagion operator algebra. doi:TBD
