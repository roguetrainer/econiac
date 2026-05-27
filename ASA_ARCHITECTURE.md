# ASA Library Architecture

**Date:** 2026-05-27  
**Status:** Design phase — ASA core not yet started as code

---

## The naming question

The core library needs a name that:
1. Is domain-agnostic (not "econ", not "physics")
2. Signals the thermodynamic / geometric character
3. Works as a Python package name
4. Can anchor a family of wrappers

Previous session proposed **`thermion`** — from *thermi-* (heat) + *-on* (carrier/particle suffix, like phonon, magnon). It signals thermodynamic routing without claiming a domain. **`asa`** is the research programme name (Adelic Simplicial Architecture); the Python package name can differ.

Alternatively: keep `asa` as the package name since the research brand is established. `import asa` is clean.

**Decision deferred** — but the architecture below uses `asa` as placeholder. Find-replace to `thermion` if needed.

---

## The wrapper family

The pattern is already visible across the papers:

| Wrapper | Domain | Core modules used | Status |
|---|---|---|---|
| **`econiac`** | Economics, finance, SFC | `asa.ensemble`, `.manifold`, `.sheaf`, `.connections`, `.routing` | v0.0.3 exists |
| **`asa-ai`** (unnamed) | ML/AI, Fano-RAG, VoT, G₂ BM | `asa.ensemble`, `.geometry`, `.algebra` | Papers 213, 214, 317 |
| **`asa-stats`** (unnamed) | Bayesian inference, MCMC | `asa.ensemble`, `.routing` | Closest to core; may just be `asa` + docs |
| **`asa-qm`** (unnamed) | Quantum hardware, RPU, FTCs | `asa.algebra`, `.geometry`, `.sheaf` | Papers 199, 205, 206 |
| **`asa-chem`** (unnamed) | FeMo-cofactor, FMO, biology | `asa.algebra`, `.sheaf`, `.geometry` | Papers 318, 319, 325 |
| **`asa-phys`** (unnamed) | Condensed matter, Hopfions, choreography | `asa.sheaf`, `.algebra`, `.manifold` | Papers 285, 323, 329 |

The statistics case is the most interesting edge case: Bayesian inference is Gibbs sampling, which is thermodynamics. `asa-stats` might just be `asa` with a different documentation target — the machinery is nearly identical to the core.

**Key principle (from previous session):** the core must be well-engineered — tested, versioned, stable API — *before* the wrapper proliferation begins. A shaky foundation becomes harder to fix once four things depend on it.

---

## Layer diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           asa  (core)                               │
│                                                                     │
│  algebra/     Octonions, G₂, Fano plane, Jordan J₃(𝕆),            │
│               731-calculus, exceptional algebras                    │
│                                                                     │
│  ensemble/    Maslov-Gibbs: Z(β), gibbs_weights, free energy F(β)  │
│               β-schedule (BOIL→SNAP), MGE operator                 │
│                                                                     │
│  geometry/    Admissibility: Abelian, Fano, G₂, Catalan            │
│               Fisher metric, Fano boundary criterion                │
│                                                                     │
│  manifold/    Pacioli manifold: ∂²=0, incidence matrix B,          │
│               homology H₀/H₁, BalanceSheet, GodleyTable            │
│               (domain-agnostic — any conserved flow network)        │
│                                                                     │
│  sheaf/       Stalks + restriction maps on graphs                   │
│               Sheaf Laplacian Δ_F, cohomology H¹                   │
│               NSD diffusion: diagonal / general / orthogonal        │
│               learn_restriction_maps() via jax.grad                 │
│                                                                     │
│  connections/ Gauge connections: log-rates, parallel transport,     │
│               holonomy, curvature (Wilson loops)                    │
│                                                                     │
│  routing/     TIR: Gibbs routing, thermal attribution, PCL DSL      │
│                                                                     │
│  calculus/    Non-associative calculus: octonion derivatives,       │
│               right-division norm, chain rule, ODEs (Phase 1)       │
└─────────────────────────────────────────────────────────────────────┘
     ▲              ▲              ▲              ▲              ▲
     │              │              │              │              │
┌────┴────┐  ┌──────┴─────┐  ┌───┴────┐  ┌─────┴───┐  ┌──────┴──────┐
│ econiac │  │  asa-stats │  │ asa-ai │  │ asa-qm  │  │  asa-chem   │
│         │  │            │  │        │  │         │  │  / asa-phys │
│ econ    │  │ inference  │  │fano-rag│  │ rpu     │  │  bio / phys │
│ finance │  │ estimation │  │ vot    │  │ ftc     │  │  choreo     │
│ sfc     │  │ testing    │  │ g2-bm  │  │ 731-isa │  │  hopfions   │
└─────────┘  └────────────┘  └────────┘  └─────────┘  └─────────────┘
```

---

## What currently lives in econiac that belongs in asa

| Current path | Moves to | Contains |
|---|---|---|
| `econiac/core/ensemble.py` | `asa/ensemble/` | Z(β), gibbs_weights — pure thermodynamics |
| `econiac/core/geometry.py` | `asa/geometry/` | Fano/G₂ admissibility — pure algebra |
| `econiac/core/manifold.py` | `asa/manifold/` | ∂²=0, incidence matrix — pure topology |
| `econiac/core/connections.py` | `asa/connections/` | Holonomy, curvature — pure differential geometry |
| `econiac/routing/tir.py` | `asa/routing/` | TIR routing — domain-agnostic |
| `econiac/routing/attribution.py` | `asa/routing/` | Thermal attribution — domain-agnostic |
| `econiac/pcl/combinators.py` | `asa/routing/pcl.py` | PCL DSL — a flow-network language |
| *(new)* | `asa/sheaf/` | Sheaf Laplacian + NSD + `learn_restriction_maps` |
| *(new)* | `asa/algebra/` | Octonions, G₂ generators (from x323 experiments) |
| *(new)* | `asa/calculus/` | Non-associative calculus (Phase 1 roadmap) |

## What stays in econiac

| Module | Reason |
|---|---|
| `economics/sfc.py` | Sector names (HH/Firms/Banks/Gov/RoW), GDP accounting conventions |
| `economics/minsky.py` | Keen dynamics, Minsky software compatibility |
| `economics/agents.py` | Household/firm/bank parametrisation |
| `economics/climate_yield.py` | Climate-yield surface, doomsday isocurves |
| `economics/supply_chain.py` | RST supply chain, tariff routing |
| `finance/fx.py` | FX rates, triangular arbitrage |
| `finance/curves.py` | Yield curves, HJM/LMM |
| `finance/credit.py` | XVA, credit spreads |

---

## The sheaf module: asa/sheaf/

**Why sheaf belongs in asa core, not econiac:**

The sheaf Laplacian is the same object in three completely different domains:

| Domain | Stalks | Restriction maps | H¹=0 means | `learn_restriction_maps` learns |
|---|---|---|---|---|
| Economics | Sector balance sheets (ℝ^assets) | Godley table flow entries | Conservation (∂²=0) | Godley entries from Flow of Funds data |
| Biology (FMO/FeMo) | Chromophore excitation amplitudes | Fano coupling coefficients | Coherent energy transfer | Coupling topology from spectroscopy |
| Choreography (3-body) | Phase space tangent planes (ℝ⁴) | Linearised monodromy | Orbit periodicity | Monodromy matrix → Maslov index |

The Cayley-transform (orthogonal) variant is the right default: it preserves the symplectic structure in physics, the Pacioli conservation in economics, and gives numerically stable gradients everywhere.

---

## asa/sheaf/ — full implementation

```python
# asa/sheaf/__init__.py
from .laplacian import SheafGraph, sheaf_laplacian_action, cohomology_gap
from .learn import learn_restriction_maps, cayley_map
from .nsd import nsd_diffuse

__all__ = [
    "SheafGraph",
    "sheaf_laplacian_action",
    "cohomology_gap", 
    "learn_restriction_maps",
    "cayley_map",
    "nsd_diffuse",
]
```

```python
# asa/sheaf/laplacian.py
"""
Sheaf Laplacian on graphs.

A sheaf F on graph G assigns:
  - A stalk F(v) = ℝ^d at each node v
  - A restriction map F_{v←e}: F(v) → F(e) = ℝ^d at each incident (node, edge) pair

The sheaf Laplacian Δ_F acts on sections x = (x_v)_{v∈V}:
  (Δ_F x)_v = Σ_{e: v∈∂e} F_{v←e}ᵀ (F_{v←e} x_v − F_{u←e} x_u)

where u is the other endpoint of edge e.

Key special cases:
  - Standard graph Laplacian: F_{v←e} = 1 for all v, e (scalar stalks, identity maps)
  - Pacioli manifold: stalks = balance sheets, maps = Godley entries, Δ_F = 0 ↔ ∂²=0
  - 3-body monodromy: stalks = phase-space tangent planes, maps = linearised flow

References:
  Hansen & Ghrist (2019) Toward a Spectral Theory of Cellular Sheaves. arXiv:1808.01513
  Bodnar et al. (2022) Neural Sheaf Diffusion. NeurIPS 2022.
  Borgi et al. (2026) Sheaf Neural Networks as Message Passing. github.com/alessioborgi/sheaf-mpnn
  Buckley (2026) Topology of Conservation. doi:10.5281/zenodo.20234853
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import jit


class SheafGraph(NamedTuple):
    """
    A graph equipped with a sheaf.

    Attributes:
        edge_src:  (n_edges,) int32 — source node index per edge
        edge_dst:  (n_edges,) int32 — destination node index per edge
        stalks:    (n_nodes, d, f) — stalk features; d=stalk_dim, f=feature_dim
        maps_src:  (n_edges, d, d) — restriction map F_{src←e} per edge
        maps_dst:  (n_edges, d, d) — restriction map F_{dst←e} per edge

    Conservation / cohomology:
        H¹(F) = 0  iff  sheaf_laplacian_action(graph) = 0
        For Pacioli manifold: this is ∂²=0 (double-entry bookkeeping).
        For 3-body orbit: this is the periodicity condition.
    """
    edge_src: jax.Array   # (n_edges,)   int
    edge_dst: jax.Array   # (n_edges,)   int
    stalks:   jax.Array   # (n_nodes, d, f)
    maps_src: jax.Array   # (n_edges, d, d)
    maps_dst: jax.Array   # (n_edges, d, d)


@jit
def sheaf_laplacian_action(graph: SheafGraph) -> jax.Array:
    """
    Compute (Δ_F x) for all nodes.

    (Δ_F x)_v = Σ_{e: v∈∂e} F_{v←e}ᵀ (F_{v←e} x_v − F_{u←e} x_u)

    Returns: (n_nodes, d, f) — the Laplacian applied to the stalk features.

    Complexity: O(n_edges × d² × f). On A100: ~microseconds for economic models.
    """
    n_nodes = graph.stalks.shape[0]
    src, dst = graph.edge_src, graph.edge_dst

    x_src = graph.stalks[src]          # (n_edges, d, f)
    x_dst = graph.stalks[dst]          # (n_edges, d, f)
    F_src = graph.maps_src             # (n_edges, d, d)
    F_dst = graph.maps_dst             # (n_edges, d, d)

    # Coboundary per edge: F_{dst←e} x_dst − F_{src←e} x_src
    Fx_src = jnp.einsum('eij,ejf->eif', F_src, x_src)  # (n_edges, d, f)
    Fx_dst = jnp.einsum('eij,ejf->eif', F_dst, x_dst)  # (n_edges, d, f)
    coboundary = Fx_dst - Fx_src                        # (n_edges, d, f)

    # Laplacian: scatter F_{v←e}ᵀ coboundary back to nodes
    # dst nodes receive  +F_{dst←e}ᵀ coboundary
    # src nodes receive  −F_{src←e}ᵀ coboundary
    FT_cob_dst = jnp.einsum('eji,ejf->eif', F_dst, coboundary)   # (n_edges, d, f)
    FT_cob_src = jnp.einsum('eji,ejf->eif', F_src, coboundary)   # (n_edges, d, f)

    lap = jnp.zeros_like(graph.stalks)
    lap = lap.at[dst].add(FT_cob_dst)
    lap = lap.at[src].add(-FT_cob_src)
    return lap


@jit
def cohomology_gap(graph: SheafGraph) -> jax.Array:
    """
    Scalar measure of sheaf cohomology: ||Δ_F x|| / (||x|| + ε).

    = 0  iff  H¹(F) = 0  (sheaf is globally consistent)

    Interpretation by domain:
      Economics:    0 ↔ all accounts balance (∂²=0 on Pacioli manifold)
      Biology:      0 ↔ coherent energy transfer (no dissipation leakage)
      Choreography: 0 ↔ periodic orbit closed (choreography condition)
    """
    lap = sheaf_laplacian_action(graph)
    return jnp.linalg.norm(lap) / (jnp.linalg.norm(graph.stalks) + 1e-12)
```

```python
# asa/sheaf/learn.py
"""
Learning restriction maps from observed stalk dynamics.

Uses jax.grad through sheaf_laplacian_action — the sheaf version of
calibrate_beta() in econiac. Instead of calibrating a scalar β, this
calibrates the full geometric structure (the restriction maps).

Three variants (following Borgi et al. 2026):
  diagonal    — F_{v←e} = diag(w), O(d) params/edge. Fast, limited.
  general     — F_{v←e} = arbitrary d×d matrix. Most expressive.
  orthogonal  — F_{v←e} = Cayley map (I−A)(I+A)⁻¹. Numerically stable.
                Preserves: symplectic structure (choreography),
                           Pacioli conservation (economics),
                           unitary evolution (quantum).

The orthogonal variant is the right default for ASA applications because
the underlying geometry in all three domains is symplectic / unitary.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, jvp
import optax
from typing import Literal
from .laplacian import SheafGraph, sheaf_laplacian_action


def cayley_map(A: jax.Array) -> jax.Array:
    """
    Cayley transform: A (skew-symmetric d×d) → O(d) orthogonal matrix.
    Q = (I − A)(I + A)⁻¹

    Properties:
      - det(Q) = 1  (special orthogonal)
      - Numerically stable: no exponential map instabilities
      - Differentiable everywhere (no branch cuts)
      - Same construction as OrthogonalNSDConv in Borgi et al. (2026)

    Corresponds to parallel transport on SO(d) — appropriate for:
      Pacioli conservation (SO(n_sectors) gauge rotations)
      Symplectic monodromy (Sp(2n) ⊂ SO(4n) for n degrees of freedom)
    """
    d = A.shape[0]
    I = jnp.eye(d)
    A_skew = A - A.T          # enforce skew-symmetry
    return jnp.linalg.solve(I + A_skew, I - A_skew)


def learn_restriction_maps(
    edge_src: jax.Array,          # (n_edges,) int
    edge_dst: jax.Array,          # (n_edges,) int
    stalk_observations: jax.Array,# (n_timesteps, n_nodes, d, f)
    stalk_dim: int,
    variant: Literal["diagonal", "general", "orthogonal"] = "orthogonal",
    n_steps: int = 1000,
    learning_rate: float = 1e-3,
    verbose: bool = False,
) -> SheafGraph:
    """
    Learn restriction maps F_{v←e} that minimise the cohomology gap
    across all observed stalk configurations.

    Loss = mean over timesteps of cohomology_gap(graph_at_t)²

    The learned maps are the geometric structure that best explains the
    observed dynamics. Domain interpretations:
      econiac:      maps ≈ Godley table entries (flow fractions between sectors)
      biology:      maps ≈ Fano coupling coefficients (energy transfer amplitudes)
      choreography: maps ≈ monodromy matrix (phase-space transport per period)

    Args:
        edge_src, edge_dst: Graph topology (fixed, not learned)
        stalk_observations: Time series of observed stalk features
        stalk_dim: d (stalk vector space dimension)
        variant: Parametrisation of restriction maps
        n_steps: Gradient descent iterations
        learning_rate: Adam learning rate

    Returns:
        SheafGraph with the final-time stalk observations and learned maps.
    """
    n_edges = len(edge_src)
    f = stalk_observations.shape[-1]

    # Initialise raw parameters (before applying variant transform)
    if variant == "diagonal":
        # d params per edge per direction (diagonal entries)
        raw = jnp.zeros((n_edges, 2, stalk_dim))  # 2 = src/dst
    elif variant == "general":
        raw = jnp.zeros((n_edges, 2, stalk_dim, stalk_dim))
    else:  # orthogonal (default)
        # Skew-symmetric matrix A; Cayley(A) gives the orthogonal map
        raw = jnp.zeros((n_edges, 2, stalk_dim, stalk_dim))

    def raw_to_maps(raw):
        """Apply variant-specific transform to raw parameters."""
        if variant == "diagonal":
            # F = diag(softplus(w)) — positive diagonal entries
            w = jax.nn.softplus(raw)  # (n_edges, 2, d)
            maps = jax.vmap(jax.vmap(jnp.diag))(w)  # (n_edges, 2, d, d)
        elif variant == "general":
            maps = raw  # unconstrained
        else:  # orthogonal
            # Apply Cayley map per edge per direction
            maps = jax.vmap(jax.vmap(cayley_map))(raw)  # (n_edges, 2, d, d)
        return maps[:, 0, :, :], maps[:, 1, :, :]   # maps_src, maps_dst

    def loss_fn(raw, stalks):
        maps_src, maps_dst = raw_to_maps(raw)
        graph = SheafGraph(
            edge_src=edge_src, edge_dst=edge_dst,
            stalks=stalks,
            maps_src=maps_src, maps_dst=maps_dst,
        )
        return cohomology_gap(graph) ** 2

    # Mean loss over all timesteps
    def total_loss(raw):
        losses = jax.vmap(loss_fn, in_axes=(None, 0))(raw, stalk_observations)
        return jnp.mean(losses)

    # Adam optimiser via optax
    optimiser = optax.adam(learning_rate)
    opt_state = optimiser.init(raw)

    @jit
    def step(raw, opt_state):
        loss, grads = jax.value_and_grad(total_loss)(raw)
        updates, opt_state = optimiser.update(grads, opt_state)
        raw = optax.apply_updates(raw, updates)
        return raw, opt_state, loss

    for i in range(n_steps):
        raw, opt_state, loss = step(raw, opt_state)
        if verbose and i % 100 == 0:
            print(f"  step {i:4d}: cohomology_gap² = {loss:.6e}")

    maps_src, maps_dst = raw_to_maps(raw)
    return SheafGraph(
        edge_src=edge_src, edge_dst=edge_dst,
        stalks=stalk_observations[-1],   # final-time stalks
        maps_src=maps_src, maps_dst=maps_dst,
    )


def cohomology_gap(graph: SheafGraph) -> jax.Array:
    """Re-exported here for convenience in loss functions."""
    from .laplacian import cohomology_gap as _cg
    return _cg(graph)
```

```python
# asa/sheaf/nsd.py
"""
Neural Sheaf Diffusion (NSD) — learnable diffusion on a sheaf.

x_{t+1} = x_t − α Δ_F x_t

At fixed restriction maps F: standard sheaf diffusion (Hansen-Ghrist).
With learnable F: Neural Sheaf Diffusion (Bodnar et al. 2022, Borgi et al. 2026).

This module provides the forward pass only. For training, use learn.py.
"""

import jax
import jax.numpy as jnp
from jax import jit
from .laplacian import SheafGraph, sheaf_laplacian_action


@jit
def nsd_diffuse(graph: SheafGraph, alpha: float = 1.0, n_steps: int = 1) -> jax.Array:
    """
    Apply n_steps of Neural Sheaf Diffusion.

    x_{t+1} = x_t − α Δ_F x_t

    Args:
        graph: SheafGraph with stalks as initial features
        alpha: Diffusion step size (should satisfy 0 < α < 2/λ_max for stability)
        n_steps: Number of diffusion steps

    Returns:
        (n_nodes, d, f) — diffused stalk features

    Connection to Gibbs ensemble:
        At α=1/β, NSD is one Newton step of the Gibbs energy minimisation.
        The fixed point of NSD is the minimum of the sheaf free energy F(β).
        This connects sheaf diffusion to the Maslov-Gibbs ensemble in asa.ensemble.
    """
    stalks = graph.stalks
    for _ in range(n_steps):
        g = SheafGraph(
            edge_src=graph.edge_src, edge_dst=graph.edge_dst,
            stalks=stalks,
            maps_src=graph.maps_src, maps_dst=graph.maps_dst,
        )
        stalks = stalks - alpha * sheaf_laplacian_action(g)
    return stalks
```

---

## Migration path

**Phase 0 (now):** Design complete. Write `asa/sheaf/` as new code — zero migration risk.

**Phase 1:** Create `asa` package repo. Implement `asa/sheaf/` from the spec above. Add to `econiac` as optional dependency: `from asa.sheaf import learn_restriction_maps`.

**Phase 2:** Move `econiac/core/ensemble.py` → `asa/ensemble/`. It has no economics-domain concepts — pure thermodynamics. Update `econiac` import. Run tests.

**Phase 3:** Move geometry, manifold, connections one at a time.

**Phase 4:** Move routing, pcl.

**Phase 5:** econiac is a pure domain wrapper. ASA core is standalone. asa-ai, asa-stats, asa-chem can each depend on `asa` directly.

**Engineering note (from previous session):** the core must be well-engineered — typed, tested, stable API, versioned releases — before any second wrapper begins. One poorly-designed abstraction in `asa.ensemble` causes rework in five libraries simultaneously.

---

## How econiac uses asa/sheaf (after migration)

```python
# econiac/economics/sfc.py  (econiac wrapper — knows sector names)
from asa.sheaf import SheafGraph, learn_restriction_maps
from asa.manifold import PacioliManifold

class SFCModel:
    def calibrate_from_flow_of_funds(self, fof_data):
        """
        Learn Godley table entries from quarterly Flow of Funds data.
        
        The restriction maps ARE the Godley entries:
          maps_src[e] = fraction of sector-src flow that goes through edge e
          maps_dst[e] = corresponding receipt fraction at sector-dst
        
        econiac provides: sector graph topology + economic meaning of stalks
        asa.sheaf provides: the learnable geometry (Cayley maps + jax.grad)
        """
        stalk_obs = self._fof_to_stalks(fof_data)  # econiac domain logic
        return learn_restriction_maps(
            edge_src=self.sector_graph.src,
            edge_dst=self.sector_graph.dst,
            stalk_observations=stalk_obs,
            stalk_dim=self.n_sectors,
            variant="orthogonal",   # preserves Pacioli conservation
            n_steps=2000,
        )
```

## How asa-phys uses asa/sheaf (future)

```python
# asa_phys/choreography/maslov.py
from asa.sheaf import SheafGraph, learn_restriction_maps

def maslov_index_from_orbit(trajectory, T):
    """
    Learn monodromy matrix from Yoshida trajectory, extract Maslov index.
    
    stalks: phase-space tangent vectors at each body (d=4 for 2D planar)
    maps:   linearised flow maps (monodromy per orbit segment)
    Maslov index = negative eigenvalue count of sheaf Laplacian
    """
    stalk_obs = trajectory_to_stalks(trajectory)  # (n_timesteps, 3, 4, 1)
    graph = learn_restriction_maps(
        edge_src=jnp.array([0, 1, 2]),
        edge_dst=jnp.array([1, 2, 0]),
        stalk_observations=stalk_obs,
        stalk_dim=4,
        variant="orthogonal",   # monodromy is symplectic → SO(4) restriction
    )
    lap_eigenvalues = jnp.linalg.eigvalsh(
        _sheaf_laplacian_matrix(graph)  # explicit matrix form for small graphs
    )
    return int(jnp.sum(lap_eigenvalues < 0))   # Maslov index
```
