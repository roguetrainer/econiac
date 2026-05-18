# econiac: Strategy, Priorities, and Next Steps

*Last updated: May 2026*

---

## What econiac is

econiac is the Python implementation of the Adelic Simplicial Architecture (ASA)
economic and financial frameworks. It is named after MONIAC (1949) — Bill Phillips's
hydraulic computer that modelled the economy as a conserved flow system. econiac does
the same with differential geometry, the Maslov-Gibbs partition function, and JAX.

The mathematical foundation is the **Pacioli manifold**: the directed graph of
institutional money flows where the conservation law ∂²=0 (every debit has a credit)
holds by construction. Exchange rates, yield curves, survival probabilities, and
attribution weights are all **connections** on this manifold. Curvature measures
path-dependence: triangular arbitrage, default correlation, and policy non-commutativity
are all curvature phenomena.

---

## The paradigm shift: why existing frameworks are insufficient

### Traditional ABMs (Mesa, Repast, NetLogo)

Traditional agent-based models are object-oriented: an agent is a software object,
and the economy advances by running a for-loop over millions of objects executing
discrete if/then rules.

**The problem**: object-oriented logic is inherently non-differentiable. You cannot
use autograd to compute the gradient of a macroscopic outcome with respect to
micro-parameters. Calibration requires gradient descent; traditional ABMs make this
impossible.

**The nuance**: Mesa etc. are not obsolete — they are the wrong tool for FGT
specifically. They remain useful for qualitative exploration and prototyping agent
logic. The workflow is: prototype in Mesa, re-implement the calibrated version in JAX.

### Digital twin frameworks (Anylogic, Simul8, commercial platforms)

Commercial digital twin platforms are built on Discrete Event Simulation (DES):
rigid, rules-based engines designed to track physical widgets through a factory.
They have no concept of continuous gauge transformations, no homology, and no
ability to compute a derivative. Forcing FGT into a DES platform strips away the
geometry that makes the theory work.

### System dynamics (Stella, Vensim, Minsky)

Stella and Vensim use visual stock-flow diagrams — pedagogically powerful, but
proprietary and non-differentiable. Keen's Minsky uses Godley tables (stock-flow
consistent balance sheets) which are *directly* the Pacioli manifold conceptually.
The gap: Minsky is non-differentiable and has no gauge structure.

**Opportunity**: `pysd` reads Stella/Vensim files and executes them in Python.
A `pysd` backend that re-executes models through the JAX/Pacioli engine would
make every existing Stella model in academia calibratable by gradient descent —
immediate adoption path.

### QuantLib

QuantLib is the industry standard for derivative pricing and risk management.
It is excellent at computing risk-free discount curves, option prices, and
risk sensitivities. It is not a gauge theory framework.

**The interface**: QuantLib computes curves; econiac wraps them as connections
on the Pacioli manifold. XVA (CVA, DVA, FVA, KVA, MVA) computed as curvature
integrals (Paper 299). The bridge is a thin adapter, not a replacement.

### Why JAX/PyTorch is the actual ABM framework

- **Native GPU acceleration**: tensor operations parallelize perfectly; 10 million
  agents via matrix multiplication is trivially fast vs. looping in Java
- **Autograd**: every transaction in FGT is a continuous algebraic operation
  (or smoothed via MGE); the framework automatically builds a computational graph
- **Calibration**: run the economy forward N steps, observe outcomes, backpropagate
  error to initial conditions — impossible with traditional ABMs
- **The mental model**: economy as RNN/GNN; Pacioli manifold as sparse adjacency
  matrix; wealth as tensor; timestep as forward pass

---

## Build order

### Phase 1: Core (build first — everything depends on this)

**`econiac.core`** — ~500 lines of JAX. Proves the framework works before any
application code.

1. `pacioli.py` — Pacioli manifold: directed graph, ∂²=0 enforcement, homology
   groups H_0, H_1, H_2
2. `mge.py` — Maslov-Gibbs partition function Z(β), `choose(β, ...)` combinator,
   β-schedule primitives
3. `connections.py` — parallel transport, holonomy, curvature F = dA + A∧A
4. `geometry.py` — four TIR geometry types: Abelian, Fano, G₂, Catalan

**Completion criterion**: a Godley table (3-sector SFC model) that enforces ∂²=0
by construction and is differentiable end-to-end.

### Phase 2: Finance (highest value, clearest adoption path)

**`econiac.finance`** — QuantLib bridge + FGT bundle implementations

1. `quantlib.py` — adapter: QuantLib curves → Pacioli connections
2. `curves.py` — yield curves as temporal connections; HJM flatness condition
   (Paper 296)
3. `fx.py` — exchange rates as connection curvature; triangular arbitrage =
   non-zero holonomy (Paper 295)
4. `credit.py` — survival probabilities as parallel transport; CVA as curvature
   integral (Papers 298-299)

**Completion criterion**: CVA surface computed via curvature integral on the
Pacioli manifold, matching QuantLib's CVA output on a benchmark portfolio.

### Phase 3: Economics / system dynamics (large community, political alignment)

**`econiac.economics`** — differentiable stock-flow engine

1. `sfc.py` — stock-flow consistency engine; Godley-Lavoie SFC models as Pacioli
   manifold instances (Papers 291, 300)
2. `minsky.py` — Minsky-compatible stock-flow DSL; drop-in replacement for Keen's
   Minsky that is differentiable and gauge-consistent
3. `agents.py` — tensor-based agent layer; agents as indices not objects; DABM
   forward pass (Paper 305)
4. `pysd_backend.py` — execute Stella/Vensim models through JAX/Pacioli engine;
   makes every existing system dynamics model differentiable

**Completion criterion**: Keen predator-prey (Lotka-Volterra debt dynamics) model
running in JAX, calibrated to data by gradient descent, matching Minsky output.

### Phase 4: Routing and attribution

**`econiac.routing`** — TIR framework and thermal Shapley values

1. `tir.py` — full TIR framework; four geometry types; β-scheduling (Paper 294)
2. `attribution.py` — thermal Shapley values; latent bottleneck index Λ_i
   (Paper 293)

### Phase 5: PCL DSL

**`econiac.pcl`** — Pacioli Combinator Library (Paper 306)

1. `combinators.py` — `choose`, `sequence`, `parallel`, `fold`; type system
   enforcing Pacioli conservation; JAX compilation

---

## Application priority ranking

| Application | Value | Difficulty | Adoption risk | Priority |
|-------------|-------|------------|---------------|----------|
| QuantLib bridge (finance) | Very high | Medium | Low (clear interface) | **1** |
| Differentiable SFC / Minsky replacement | High | Medium | Medium (community exists) | **2** |
| Tensor-based DABM | High | Medium | Medium | **3** |
| Stella/pysd JAX backend | Medium | Low | Low (pysd is open) | **4** |
| Multi-agent LLM routing (Paper 307) | High | Low | Low (LangGraph exists) | **5** |

---

## Papers implemented by each module

| Module | Papers |
|--------|--------|
| `core/pacioli.py` | 291 (Topology of Conservation) |
| `core/mge.py` | 201 (MGE), 289 (Temperature of Rationality) |
| `core/connections.py` | 291, 300 (EGT) |
| `finance/curves.py` | 296 (Term Structure Bundles) |
| `finance/fx.py` | 295 (Currency Bundles) |
| `finance/credit.py` | 298 (Credit Bundles), 299 (XVA) |
| `finance/quantlib.py` | 299 (XVA as curvature) |
| `economics/sfc.py` | 291, 300 (EGT), 302 (IFRS) |
| `economics/agents.py` | 305 (DABM) |
| `economics/minsky.py` | 300 (EGT), 305 (DABM) |
| `routing/tir.py` | 294 (TIR) |
| `routing/attribution.py` | 293 (Thermal Attribution) |
| `pcl/combinators.py` | 306 (PCL) |

---

## What econiac is NOT

- Not a replacement for QuantLib — it wraps QuantLib
- Not a replacement for Mesa for qualitative ABM exploration
- Not a DES framework — no event queues, no discrete rules
- Not a standalone statistics package — use scipy/statsmodels for that

---

## Open questions / decisions needed

1. **JAX vs PyTorch**: JAX is preferred (functional, JIT, no mutable state matches
   the gauge theory formalism); PyTorch is more familiar to ML practitioners.
   Recommendation: JAX-first with a PyTorch compatibility layer.

2. **QuantLib Python bindings**: QuantLib-Python (`QuantLib` on PyPI) provides
   SWIG-generated bindings — functional but not differentiable. Consider whether
   to wrap the Python bindings directly or interface at the C++ level via pybind11
   for better performance.

3. **Graph backend**: the Pacioli manifold adjacency matrix could use `scipy.sparse`,
   `jax.experimental.sparse`, or `torch_geometric`. Decision pending Phase 1 build.

4. **pysd integration**: `pysd` is MIT licensed and actively maintained. A PR to
   pysd adding a JAX execution backend would be the fastest adoption path for the
   system dynamics community.
