"""
sheaf.py
========
Cellular sheaf cohomology on financial networks — H¹ early-warning signal.

A cellular sheaf assigns:
  - a vector space (stalk) to each node and edge of a graph
  - a linear map (restriction map) from each node stalk to each adjacent edge stalk

Consistency: a *section* s assigns values s_i to each node such that the
restriction maps agree on every edge. Sections that are consistent everywhere
are *global sections* (ker δ₀). The obstruction to global consistency is
measured by the first sheaf cohomology group H¹ = ker δ₁ / im δ₀.

Financial interpretation
------------------------
The stalk at each node is a one-dimensional space carrying the agent's
"health ratio":
  - Fire-sale model (Paper 332): capital ratio γᵢ = Eᵢ / RWAᵢ
  - Repo model (Paper 333):      funding ratio fᵢ = rolled_funding / repo_out
  - FMO model (Paper 325):       energy transfer efficiency ηᵢ

The restriction map on edge (i→j) is the bilateral exposure weight:
    F_{i, e} = exposure_ij / total_assets_i

Consistency section s is globally consistent iff:
    F_{j,e} · s_j = F_{i,e} · s_i   for all edges e = (i→j)

H¹ signal
---------
    signal(t) = ‖L_F · s(t)‖² / ‖s(t)‖²

Zero iff s is a harmonic section (globally consistent valuation).
Large iff there are inconsistent cycles — agents disagree on the network's
solvency/funding state. This disagreement precedes defaults because it
reflects stress that has not yet been absorbed into observable breaches.

Empirically (x332e, 20 seeds): H¹ signal peaks 2-3 periods before the
cascade distress count peaks. The same topology appears in the FMO Fano
line (Paper 325) and the FeMo-cofactor G₂ Casimir (Paper 318) — all three
are broken-symmetry H¹ obstructions on different graphs.

Three-way isomorphism (Paper 334 §6):
    CHZ fire-sale cascade  ←→  Sovereign repo run  ←→  FMO Fano breakdown
    H¹ on interbank graph  ←→  H¹ on dealer-lender ←→  H¹ on Fano graph
    capital ratio section  ←→  funding ratio section←→  energy ratio section

References:
    Robinson, M. (2014). Topological Signal Processing. Springer.
    Hansen, J. & Ghrist, R. (2021). Opinion Dynamics on Discourse Sheaves.
        SIAM J. Appl. Math. 81(5), 2033-2060. doi:10.1137/20M1341088
    Buckley (2026). Paper 325: FMO topological heat engine.
        doi:10.5281/zenodo.20400638
    Buckley (2026). Paper 332: CHZ fire sales. doi:TBD
    Buckley (2026). Paper 333: Sovereign repo run. doi:TBD
    Buckley (2026). Paper 334: Contagion operator algebra. doi:TBD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence

import numpy as np
import jax.numpy as jnp
import jax


# ---------------------------------------------------------------------------
# Edge and graph types
# ---------------------------------------------------------------------------

@dataclass
class WeightedEdge:
    """
    A directed edge in the financial network.

    source:  index of the source node (lender / bank)
    target:  index of the target node (borrower / counterparty)
    weight:  bilateral exposure amount (not normalised)
    """
    source: int
    target: int
    weight: float


@dataclass
class FinancialGraph:
    """
    Weighted directed graph representing bilateral exposures.

    n_nodes:    number of agents (banks, dealers, lenders)
    edges:      list of WeightedEdge
    node_assets: (n_nodes,) — total assets per agent (for normalising
                  restriction maps: F_{i,e} = exposure_ij / assets_i)

    Constructed from an exposure matrix via ``from_matrix()``.
    """
    n_nodes:     int
    edges:       list[WeightedEdge]
    node_assets: np.ndarray   # (n_nodes,)

    @classmethod
    def from_matrix(
        cls,
        A:           np.ndarray,   # (n, n) — A[i,j] = exposure of i to j
        node_assets: np.ndarray,   # (n,) — total assets per node
        threshold:   float = 0.01, # edges with weight < threshold × max are dropped
    ) -> "FinancialGraph":
        """
        Build a FinancialGraph from a bilateral exposure matrix.

        Args:
            A:           (n, n) exposure matrix; A[i,j] = amount i is owed by j
                          (or amount i lends to j — convention is consistent if used
                          uniformly within one model)
            node_assets: (n,) total assets per node (for restriction map normalisation)
            threshold:   edges with A[i,j] < threshold × max(A) are dropped

        Returns:
            FinancialGraph
        """
        n = A.shape[0]
        cutoff = float(np.max(A)) * threshold
        edges = [
            WeightedEdge(source=i, target=j, weight=float(A[i, j]))
            for i in range(n) for j in range(n)
            if i != j and A[i, j] > cutoff
        ]
        return cls(n_nodes=n, edges=edges, node_assets=np.array(node_assets))

    @classmethod
    def from_bipartite(
        cls,
        B:              np.ndarray,   # (n_lenders, n_dealers) — lending matrix
        lender_assets:  np.ndarray,   # (n_lenders,)
        dealer_assets:  np.ndarray,   # (n_dealers,)
        threshold:      float = 0.01,
    ) -> "FinancialGraph":
        """
        Build a FinancialGraph from a bipartite lender-dealer matrix.

        Used for the repo market (Paper 333): lenders are nodes 0..n_lenders-1,
        dealers are nodes n_lenders..n_lenders+n_dealers-1.

        Args:
            B:              (n_lenders, n_dealers) — B[i,j] = lending of lender i to dealer j
            lender_assets:  (n_lenders,) total assets of each lender
            dealer_assets:  (n_dealers,) total assets of each dealer
            threshold:      edge weight cutoff

        Returns:
            FinancialGraph on n_lenders + n_dealers nodes
        """
        n_l, n_d = B.shape
        n = n_l + n_d
        node_assets = np.concatenate([lender_assets, dealer_assets])
        cutoff = float(np.max(B)) * threshold if np.max(B) > 0 else 0.0
        edges = [
            WeightedEdge(source=i, target=n_l + j, weight=float(B[i, j]))
            for i in range(n_l) for j in range(n_d)
            if B[i, j] > cutoff
        ]
        return cls(n_nodes=n, edges=edges, node_assets=node_assets)


# ---------------------------------------------------------------------------
# Sheaf Laplacian
# ---------------------------------------------------------------------------

class SheafLaplacianResult(NamedTuple):
    """Output of sheaf_laplacian()."""
    L_F:         np.ndarray   # (n_nodes, n_nodes) — symmetric PSD sheaf Laplacian
    delta0:      np.ndarray   # (n_edges, n_nodes) — coboundary operator
    eigenvalues: np.ndarray   # (n_nodes,) — eigenvalues of L_F in ascending order
    h1_dim:      int          # dim H¹ = number of near-zero eigenvalues
    n_edges:     int          # number of edges included


def sheaf_laplacian(
    graph:        FinancialGraph,
    section:      np.ndarray,   # (n_nodes,) — current health ratios (γ, f, η, …)
    zero_tol:     float = 1e-6, # eigenvalue threshold for H¹ dimension count
) -> SheafLaplacianResult:
    """
    Build the cellular sheaf Laplacian on a financial network.

    Construction
    ------------
    Stalk at each node i: ℝ (one-dimensional; the health ratio sᵢ).

    Restriction map on directed edge e = (i → j):
        F_{i,e} = w_{ij} / (assets_i + ε)   (source side)
        F_{j,e} = w_{ij} / (assets_j + ε)   (target side)

    Coboundary δ₀ : C⁰(G; F) → C¹(G; F):
        (δ₀ s)[e=(i,j)] = F_{j,e} · s_j − F_{i,e} · s_i

    This is zero for edge e iff the two endpoints agree on the relative
    importance of their bilateral exposure (consistent valuation).

    Sheaf Laplacian:
        L_F = δ₀ᵀ δ₀

    L_F is symmetric positive semidefinite. Its null space is the space of
    globally consistent sections (harmonic sections). dim ker(L_F) = dim H⁰.
    The H¹ obstruction is measured by the *near-zero but positive* eigenvalues
    of L_F — they indicate directions in which the section is almost consistent
    but not quite.

    Args:
        graph:    FinancialGraph with nodes, edges, and node_assets
        section:  (n_nodes,) health ratio at each agent
        zero_tol: threshold below which eigenvalues count toward h1_dim

    Returns:
        SheafLaplacianResult
    """
    n = graph.n_nodes
    m = len(graph.edges)
    eps = 1e-8

    if m == 0:
        return SheafLaplacianResult(
            L_F         = np.zeros((n, n)),
            delta0      = np.zeros((0, n)),
            eigenvalues = np.zeros(n),
            h1_dim      = n,
            n_edges     = 0,
        )

    delta0 = np.zeros((m, n))
    for k, e in enumerate(graph.edges):
        i, j, w = e.source, e.target, e.weight
        f_i = w / (graph.node_assets[i] + eps)
        f_j = w / (graph.node_assets[j] + eps)
        delta0[k, i] = -f_i
        delta0[k, j] =  f_j

    L_F  = delta0.T @ delta0                          # (n, n)
    eigs = np.linalg.eigvalsh(L_F)                    # ascending order
    h1   = int(np.sum(eigs < zero_tol))

    return SheafLaplacianResult(
        L_F         = L_F,
        delta0      = delta0,
        eigenvalues = eigs,
        h1_dim      = h1,
        n_edges     = m,
    )


# ---------------------------------------------------------------------------
# H¹ signal — the scalar early-warning indicator
# ---------------------------------------------------------------------------

def h1_signal(
    L_F:     np.ndarray,   # (n_nodes, n_nodes)
    section: np.ndarray,   # (n_nodes,)
) -> float:
    """
    Cohomological stress signal: ‖L_F · s‖² / ‖s‖².

    Interpretation
    --------------
    Zero   iff s is a harmonic section (globally consistent valuation).
    Large  iff the section s has large components in the directions of
            maximal curvature — i.e., agents strongly disagree on relative
            solvency/funding across bilateral exposures.

    WARNING (2026-05-29): this Rayleigh-quotient form does NOT lead the cascade
    and barely fires for near-flat health-ratio sections — it is dominated by
    the harmonic (constant) component. Empirically (x332e_h1_diagnostic.py) its
    mean lead is +0.2 periods at magnitude ~1e-4. Use h1_obstruction_signal()
    instead, which leads by ~1 period and fires at O(1). Retained only for
    backward compatibility / comparison; do NOT use as the early-warning signal.

    Args:
        L_F:     sheaf Laplacian from sheaf_laplacian()
        section: (n_nodes,) health ratios (capital ratios, funding ratios, …)

    Returns:
        scalar ≥ 0
    """
    s   = np.array(section, dtype=float)
    Ls  = L_F @ s
    num = float(np.dot(Ls, Ls))
    den = float(np.dot(s, s)) + 1e-10
    return num / den


def h1_obstruction_signal(
    L_F:      np.ndarray,   # (n_nodes, n_nodes) sheaf Laplacian
    section:  np.ndarray,   # (n_nodes,) health-ratio section
    zero_tol: float = 1e-6,
) -> float:
    """
    Cohomological obstruction signal: ‖s − P_ker s‖ / ‖s‖,
    where P_ker is the orthogonal projector onto ker(L_F) (the harmonic =
    globally-consistent sections). This is the fraction of the section that
    CANNOT be reconciled to a consistent global valuation — the genuine H¹
    obstruction.

    Why this and not the Rayleigh quotient ‖L_F s‖²/‖s‖²
    ----------------------------------------------------
    The Rayleigh quotient (the previous ``h1_signal``) is dominated by the
    near-CONSTANT component of the section. For health-ratio data (capital
    ratios, funding ratios) the section is nearly flat across agents, and a
    constant vector is harmonic (in ker L_F), so the Rayleigh quotient is tiny
    (~1e-4 in the CHZ model) and tracks the VARIANCE of s, not its cohomological
    inconsistency. Empirically (x332e_h1_diagnostic.py, 15 seeds) the Rayleigh
    signal does NOT lead the cascade (mean lead +0.2 periods) and barely fires,
    whereas this obstruction signal leads by ~1 period (max 2) and fires at
    O(1). It is the correct early-warning observable.

    Interpretation
    --------------
    0    : s lies entirely in ker(L_F) — globally consistent (no obstruction).
    →1   : s is almost entirely obstructed — agents hold mutually irreconcilable
           bilateral valuations around cycles in the network.

    Args:
        L_F:      sheaf Laplacian from sheaf_laplacian()
        section:  (n_nodes,) health ratios
        zero_tol: eigenvalue threshold defining the harmonic (kernel) subspace

    Returns:
        scalar in [0, 1]
    """
    s = np.array(section, dtype=float)
    eigs, vecs = np.linalg.eigh(L_F)             # ascending
    harm = eigs < zero_tol
    if np.any(harm):
        Vh = vecs[:, harm]                        # (n, k) kernel basis
        s_harm = Vh @ (Vh.T @ s)                  # projection onto ker(L_F)
    else:
        s_harm = np.zeros_like(s)
    obstruction = s - s_harm
    return float(np.linalg.norm(obstruction) / (np.linalg.norm(s) + 1e-12))


def h1_signal_normalised(
    L_F:     np.ndarray,
    section: np.ndarray,
) -> float:
    """
    Normalised H¹ signal: ‖L_F · s‖ / (‖L_F‖_F · ‖s‖ + ε).

    Divides by the Frobenius norm of L_F, making the signal comparable
    across networks of different sizes and connectivity densities.
    Used for cross-model comparison in Paper 334 §6.

    Args:
        L_F:     sheaf Laplacian
        section: health ratio section

    Returns:
        normalised scalar ∈ [0, 1]
    """
    s    = np.array(section, dtype=float)
    Ls   = L_F @ s
    num  = float(np.linalg.norm(Ls))
    den  = float(np.linalg.norm(L_F, 'fro')) * float(np.linalg.norm(s)) + 1e-10
    return num / den


# ---------------------------------------------------------------------------
# Harmonic decomposition — which direction is the obstruction?
# ---------------------------------------------------------------------------

class HarmonicDecomposition(NamedTuple):
    """
    Decomposition of the section into harmonic and non-harmonic parts.

    harmonic:      the projection of s onto ker(L_F)  (consistent part)
    obstruction:   s − harmonic  (the inconsistency)
    node_contrib:  (n_nodes,) — per-node contribution to the obstruction
    top_nodes:     indices of the top-5 most obstructed nodes
    """
    harmonic:    np.ndarray   # (n_nodes,)
    obstruction: np.ndarray   # (n_nodes,)
    node_contrib: np.ndarray  # (n_nodes,) ≥ 0
    top_nodes:   np.ndarray   # (≤5,) indices


def harmonic_decomposition(
    result:  SheafLaplacianResult,
    section: np.ndarray,
) -> HarmonicDecomposition:
    """
    Decompose the section into harmonic (consistent) and obstructed parts.

    The harmonic part is the projection onto ker(L_F) — the space of
    globally consistent sections. The obstruction is what remains.

    Uses the eigendecomposition of L_F: the harmonic subspace is spanned
    by eigenvectors with eigenvalue < zero_tol.

    Args:
        result:  SheafLaplacianResult from sheaf_laplacian()
        section: (n_nodes,) health ratio section

    Returns:
        HarmonicDecomposition
    """
    s = np.array(section, dtype=float)

    # Full eigendecomposition
    eigs, vecs = np.linalg.eigh(result.L_F)   # ascending order

    # Harmonic subspace: eigenvectors with eigenvalue ≈ 0
    harmonic_mask = eigs < 1e-6
    if np.any(harmonic_mask):
        V_harm = vecs[:, harmonic_mask]               # (n, k)
        s_harm = V_harm @ (V_harm.T @ s)              # projection
    else:
        s_harm = np.zeros_like(s)

    obstruction  = s - s_harm
    node_contrib = np.abs(obstruction)
    top_nodes    = np.argsort(node_contrib)[::-1][:5]

    return HarmonicDecomposition(
        harmonic     = s_harm,
        obstruction  = obstruction,
        node_contrib = node_contrib,
        top_nodes    = top_nodes,
    )


# ---------------------------------------------------------------------------
# Time-series tracking — sheaf state at each simulation period
# ---------------------------------------------------------------------------

class SheafPeriod(NamedTuple):
    """Sheaf diagnostics at one simulation period."""
    t:              int
    h1_signal:      float     # ‖L_F s‖² / ‖s‖²
    h1_signal_norm: float     # normalised version
    h1_dim:         int       # dim H¹ (near-zero eigenvalue count)
    n_distressed:   int       # agents below threshold
    section:        np.ndarray   # (n_nodes,) health ratios
    top_nodes:      np.ndarray   # top-5 obstructed nodes


class SheafTimeSeries(NamedTuple):
    """Full sheaf time series across the cascade simulation."""
    periods:         list[SheafPeriod]
    peak_h1_t:       int      # period of peak H¹ signal
    peak_distress_t: int      # period of peak distress count
    lead_time:       int      # peak_distress_t − peak_h1_t (positive = H¹ leads)
    mean_h1:         float
    max_h1:          float

    def signals(self) -> np.ndarray:
        """(T,) array of H¹ signals across all periods."""
        return np.array([p.h1_signal for p in self.periods])

    def distress_counts(self) -> np.ndarray:
        """(T,) array of distressed agent counts."""
        return np.array([p.n_distressed for p in self.periods])

    def h1_dims(self) -> np.ndarray:
        """(T,) array of H¹ dimensions."""
        return np.array([p.h1_dim for p in self.periods])


def sheaf_h1_signal(
    graph:       FinancialGraph,
    sections:    np.ndarray,     # (T, n_nodes) — health ratios over T periods
    thresholds:  np.ndarray,     # (n_nodes,) — distress threshold per agent
    zero_tol:    float = 1e-6,
) -> SheafTimeSeries:
    """
    Compute the sheaf H¹ early-warning signal across a cascade simulation.

    This is the primary function called by Papers 332, 333, and any future
    contagion model. It consumes the time-series of health ratios from the
    simulation and returns the H¹ signal at each period, along with the
    lead time (how many periods H¹ precedes the distress peak).

    Args:
        graph:      FinancialGraph — the bilateral exposure network
        sections:   (T, n_nodes) — health ratio (γ, f, η) at each period
        thresholds: (n_nodes,) — each agent is "distressed" if section < threshold
        zero_tol:   eigenvalue threshold for H¹ dimension count

    Returns:
        SheafTimeSeries with per-period diagnostics and summary statistics

    Example (fire-sale model):
        graph    = FinancialGraph.from_matrix(A_interbank, total_assets)
        sections = np.array([[gamma_t0], [gamma_t1], ...])  # (T, n_banks)
        ts       = sheaf_h1_signal(graph, sections, thresholds=np.full(n, gamma_min))

    Example (repo model):
        graph    = FinancialGraph.from_bipartite(B_lender_dealer, l_assets, d_assets)
        sections = np.array([funding_ratios_t])...
        ts       = sheaf_h1_signal(graph, sections, thresholds=np.full(n, f_min))
    """
    T = sections.shape[0]
    periods = []

    for t in range(T):
        s_t    = sections[t]
        lap    = sheaf_laplacian(graph, s_t, zero_tol=zero_tol)
        sig    = h1_signal(lap.L_F, s_t)
        sig_n  = h1_signal_normalised(lap.L_F, s_t)
        n_dist = int(np.sum(s_t < thresholds))
        decomp = harmonic_decomposition(lap, s_t)

        periods.append(SheafPeriod(
            t              = t,
            h1_signal      = sig,
            h1_signal_norm = sig_n,
            h1_dim         = lap.h1_dim,
            n_distressed   = n_dist,
            section        = s_t,
            top_nodes      = decomp.top_nodes,
        ))

    signals  = np.array([p.h1_signal   for p in periods])
    distress = np.array([p.n_distressed for p in periods])

    peak_h1_t       = int(np.argmax(signals))
    peak_distress_t = int(np.argmax(distress))
    lead_time       = peak_distress_t - peak_h1_t

    return SheafTimeSeries(
        periods         = periods,
        peak_h1_t       = peak_h1_t,
        peak_distress_t = peak_distress_t,
        lead_time       = lead_time,
        mean_h1         = float(np.mean(signals)),
        max_h1          = float(np.max(signals)),
    )


# ---------------------------------------------------------------------------
# Cross-model isomorphism — compare H¹ signals across different graphs
# ---------------------------------------------------------------------------

class IsomorphismResult(NamedTuple):
    """
    Result of comparing two SheafTimeSeries for structural similarity.

    Measures whether two financial networks (or a financial network and
    a quantum network like FMO) share the same H¹ obstruction structure.

    correlation: Pearson r between the two normalised H¹ signal series
    lead_offset: period offset that maximises the correlation
    is_isomorphic: True if correlation > 0.8 (empirical threshold)
    """
    correlation:    float
    lead_offset:    int
    is_isomorphic:  bool
    label_a:        str
    label_b:        str


def compare_h1_series(
    ts_a:    SheafTimeSeries,
    ts_b:    SheafTimeSeries,
    label_a: str = "Model A",
    label_b: str = "Model B",
    r_threshold: float = 0.80,
) -> IsomorphismResult:
    """
    Compare two H¹ time series for structural isomorphism.

    Pads the shorter series with its final value and computes the
    cross-correlation at all lags. The lag with maximum correlation
    is the lead_offset.

    Used in Paper 334 §6 to demonstrate the three-way isomorphism:
        CHZ cascade H¹  ←→  Repo run H¹  ←→  FMO Fano H¹

    Args:
        ts_a, ts_b:   two SheafTimeSeries (may be different lengths)
        label_a/b:    human-readable labels for reporting
        r_threshold:  Pearson r above which we call the series isomorphic

    Returns:
        IsomorphismResult
    """
    sig_a = ts_a.signals()
    sig_b = ts_b.signals()

    # Normalise to zero mean, unit variance
    def normalise(x):
        mu, sd = x.mean(), x.std() + 1e-10
        return (x - mu) / sd

    # Pad to equal length
    T = max(len(sig_a), len(sig_b))
    a = np.pad(normalise(sig_a), (0, T - len(sig_a)), mode='edge')
    b = np.pad(normalise(sig_b), (0, T - len(sig_b)), mode='edge')

    # Cross-correlation
    xcorr  = np.correlate(a, b, mode='full')
    lags   = np.arange(-(T - 1), T)
    best_i = int(np.argmax(xcorr))
    best_r = float(xcorr[best_i] / T)  # normalised by length
    lead   = int(lags[best_i])

    return IsomorphismResult(
        correlation   = best_r,
        lead_offset   = lead,
        is_isomorphic = best_r > r_threshold,
        label_a       = label_a,
        label_b       = label_b,
    )


# ---------------------------------------------------------------------------
# Convenience: gravity-model interbank network (from x332e, generalised)
# ---------------------------------------------------------------------------

def gravity_network(
    total_lending:    np.ndarray,   # (n,) — BL_i: total amount each node lends
    total_borrowing:  np.ndarray,   # (n,) — BB_i: total amount each node borrows
    total_assets:     np.ndarray,   # (n,) — for restriction map normalisation
    noise_scale:      float = 0.01,
    threshold:        float = 0.01,
    rng:              Optional[np.random.Generator] = None,
) -> FinancialGraph:
    """
    Construct a random bilateral exposure network via the gravity model.

    A_{ij} ∝ BL_i × BB_j  (i ≠ j), scaled so row sums = BL_i.
    Small exponential noise is added to prevent perfect rank-1 structure.

    This replicates the ``build_interbank_matrix()`` logic from x332e,
    generalised to arbitrary node populations and extracted into the library
    so that both x332e and x333e use the same construction.

    Args:
        total_lending:   (n,) how much each node lends in total (BL_i)
        total_borrowing: (n,) how much each node borrows in total (BB_i)
        total_assets:    (n,) total assets per node (for restriction maps)
        noise_scale:     scale of exponential noise relative to max(A)
        threshold:       drop edges below threshold × max(A)
        rng:             numpy random generator (default: new default_rng())

    Returns:
        FinancialGraph
    """
    if rng is None:
        rng = np.random.default_rng()

    n   = len(total_lending)
    BL  = np.array(total_lending,   dtype=float)
    BB  = np.array(total_borrowing, dtype=float)

    # Gravity model
    outer = np.outer(BL, BB)
    np.fill_diagonal(outer, 0.0)

    # Scale rows so row sums = BL_i
    row_sums = outer.sum(axis=1, keepdims=True) + 1e-10
    A = outer / row_sums * BL[:, None]

    # Add small noise
    noise = rng.exponential(noise_scale, (n, n))
    np.fill_diagonal(noise, 0.0)
    A = A + noise

    # Re-normalise
    row_sums = A.sum(axis=1, keepdims=True) + 1e-10
    A = A / row_sums * BL[:, None]

    return FinancialGraph.from_matrix(A, total_assets, threshold=threshold)
