"""
h2.py
=====
H² sheaf cohomology — systemic cascade obstruction on financial networks.

H⁰, H¹, H² form the complete cohomological hierarchy for financial risk
(Papers 396–398, doi:10.5281/zenodo.20635479 / 20642908 / 20642983):

    H⁰  bilateral solvency      ker(δ⁰)          can each agent meet obligations?
    H¹  triangular consistency  ker(δ¹) / im(δ⁰)  are bilateral exposures coherent?
    H²  systemic cascade        coker(δ¹)          is there any consistent resolution?

This module implements H². The central object is the *coboundary operator
on triangles*:

    δ¹ : C¹(G; ℱ) → C²(G; ℱ)

On a graph G, C² is indexed by triangles (3-cycles). For each oriented
triangle (i→j→k→i), the coboundary is the failure of the exposure weights
to satisfy a consistency relation around the cycle:

    (δ¹ f)[i,j,k] = F_{ij} · f_{jk} + F_{jk} · f_{ki} + F_{ki} · f_{ij} - f_{ij} - f_{jk} - f_{ki}

(In the one-dimensional stalk model, this simplifies to the signed sum of
edge values around the triangle — non-zero iff the triangle is inconsistent.)

H² = ker(δ²) / im(δ¹)  (trivially, since C³ = 0 for a graph)
   = coker(δ¹)
   = the space of triangle inconsistencies that cannot be resolved by
     any reassignment of edge values (1-cochains).

Financial interpretation (Paper 397)
--------------------------------------
A non-zero H² means there exists a cycle of bilateral exposures
whose inconsistency CANNOT be resolved by any bilateral renegotiation.
This is the "unhedgeable residual" — the portion of systemic stress
that persists no matter what bilateral netting or haircut adjustments
are made. It is the topological signature of an H² event (2008 crisis).

Concretely:
    dim H²(G, ℱ) > 0  ⟺  the network has entered a regime where no
                           consistent global valuation exists, and bilateral
                           interventions cannot prevent cascade.

    dim H² = 0         ⟺  the unhedgeable residual is zero; a consistent
                           global resolution exists in principle.

Connection to H¹
----------------
H² is the obstruction to *lifting* a consistent solution from H¹ to a
global section. When H¹ = 0 (no triangular inconsistency), H² is also 0.
H² can be non-zero only when H¹ ≠ 0 — i.e., H² is a *refinement* of H¹,
detecting whether the triangular inconsistency is globally irresolvable.

In practice:
    - H¹ signal rises as stress accumulates (1–2 periods before cascade)
    - H² becomes non-zero at the tipping point when no resolution exists
    - H² > 0 is the "point of no return" for systemic intervention

References
----------
Buckley (2026). Systemic Risk as H²: Cohomological Stress Testing.
    doi:10.5281/zenodo.20642908  (Paper 397)
Buckley (2026). The Topology of Risk: Primer on Cohomology.
    doi:10.5281/zenodo.20642983  (Paper 398)
Buckley (2026). The 6j Symbol as H¹.
    doi:10.5281/zenodo.20635479  (Paper 396)
Curry, J. (2014). Sheaves, Cosheaves and Applications. arXiv:1303.3255
Robinson, M. (2014). Topological Signal Processing. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np

from econiac.finance.contagion.sheaf import (
    FinancialGraph,
    WeightedEdge,
    sheaf_laplacian,
    SheafLaplacianResult,
    h1_obstruction_signal,
)


# ---------------------------------------------------------------------------
# Triangle enumeration
# ---------------------------------------------------------------------------

def find_triangles(graph: FinancialGraph) -> list[tuple[int, int, int]]:
    """
    Find all directed triangles (3-cycles) in the financial graph.

    A triangle is an ordered triple (i, j, k) such that edges i→j, j→k,
    and k→i all exist (have non-zero weight after thresholding).

    Returns:
        List of (i, j, k) triples. Each undirected triangle appears as
        up to 2 directed orientations.
    """
    # Build adjacency set for O(1) edge lookup
    edge_set = {(e.source, e.target) for e in graph.edges}
    triangles = []
    nodes = list(range(graph.n_nodes))

    for i in nodes:
        for j in nodes:
            if i == j: continue
            if (i, j) not in edge_set: continue
            for k in nodes:
                if k == i or k == j: continue
                if (j, k) in edge_set and (k, i) in edge_set:
                    triangles.append((i, j, k))

    return triangles


# ---------------------------------------------------------------------------
# Coboundary operator δ¹
# ---------------------------------------------------------------------------

def delta1_matrix(
    graph:     FinancialGraph,
    triangles: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    Build the coboundary operator δ¹ : C¹(G; ℱ) → C²(G; ℱ).

    C¹ is indexed by directed edges (n_edges dimensions).
    C² is indexed by directed triangles (n_triangles dimensions).

    For triangle (i→j→k→i), the coboundary collects the three participating
    edge values with signs +1 (forward direction of triangle orientation).

    Returns:
        (n_triangles, n_edges) matrix. Entry [t, e] is ±1 if edge e is in
        triangle t with the corresponding orientation, 0 otherwise.
    """
    n_edges     = len(graph.edges)
    n_triangles = len(triangles)

    if n_triangles == 0 or n_edges == 0:
        return np.zeros((max(n_triangles, 1), max(n_edges, 1)))

    # Edge index for fast lookup
    edge_index = {(e.source, e.target): k for k, e in enumerate(graph.edges)}

    D1 = np.zeros((n_triangles, n_edges))

    for t, (i, j, k) in enumerate(triangles):
        # Three edges of the triangle (i→j), (j→k), (k→i)
        for src, tgt in [(i, j), (j, k), (k, i)]:
            if (src, tgt) in edge_index:
                D1[t, edge_index[(src, tgt)]] = +1.0
            # Reverse edge gets -1 if present
            if (tgt, src) in edge_index:
                D1[t, edge_index[(tgt, src)]] = -1.0

    return D1


# ---------------------------------------------------------------------------
# H² computation
# ---------------------------------------------------------------------------

class H2Result(NamedTuple):
    """
    Result of H² cohomology computation on a financial network.

    h2_dim:          dim H²(G, ℱ) = number of irresolvable triangle
                     inconsistencies (the "unhedgeable residual" dimension).
    h2_signal:       scalar ≥ 0 measuring the magnitude of H² obstruction.
                     Zero iff h2_dim = 0 (no systemic cascade obstruction).
    n_triangles:     number of directed 3-cycles found in the network.
    unhedgeable:     (n_triangles,) — residual inconsistency per triangle
                     after best-fit bilateral netting (im δ¹ subtracted).
    top_triangles:   indices of the 5 most obstructed triangles.
    is_h2_event:     True if h2_signal > h2_threshold (systemic cascade
                     regime; no bilateral intervention can resolve cascade).
    h2_threshold:    the threshold used for is_h2_event classification.
    h1_signal:       the H¹ obstruction signal (for comparison / reference).
    """
    h2_dim:        int
    h2_signal:     float
    n_triangles:   int
    unhedgeable:   np.ndarray   # (n_triangles,)
    top_triangles: np.ndarray   # (≤5,) indices
    is_h2_event:   bool
    h2_threshold:  float
    h1_signal:     float


def h2_obstruction(
    graph:         FinancialGraph,
    section:       np.ndarray,   # (n_nodes,) health ratios
    h2_threshold:  float = 0.10, # H² signal above this → H² event
    zero_tol:      float = 1e-6,
) -> H2Result:
    """
    Compute the H² systemic cascade obstruction for a financial network.

    Algorithm
    ---------
    1. Find all directed triangles in the graph.
    2. Build the edge inconsistency vector b ∈ C¹(G; ℱ):
           b[e=(i→j)] = F_{j,e}·s_j − F_{i,e}·s_i
       (the coboundary δ⁰s — same as computed by sheaf_laplacian).
    3. Build δ¹ (n_triangles × n_edges).
    4. Project b onto im(δ¹)ᵀ = the space of edge-inconsistencies that
       CAN be resolved by triangle-level renegotiation. The residual
       b − P_{im(δ¹)ᵀ} b  is the unhedgeable residual — it lives in H².
    5. dim H² = rank of the unhedgeable residual subspace.
       H² signal = ‖unhedgeable residual‖ / ‖b‖ ∈ [0, 1].

    Interpretation
    --------------
    H² signal = 0  : all triangular inconsistency can be resolved bilaterally.
                     No systemic cascade obstruction.
    H² signal > 0  : part of the stress is irresolvable at the bilateral level.
    H² signal ≥ h2_threshold : H² event — the network has crossed the tipping
                     point into irresolvable systemic cascade.

    Args:
        graph:         FinancialGraph with bilateral exposures
        section:       (n_nodes,) health ratios (capital ratios, funding ratios)
        h2_threshold:  H² signal above which is_h2_event = True (default 0.10)
        zero_tol:      threshold for near-zero singular values (numerical rank)

    Returns:
        H2Result
    """
    n = graph.n_nodes
    eps = 1e-8

    # ── Step 1: H¹ signal for reference ─────────────────────────────────────
    lap = sheaf_laplacian(graph, section, zero_tol=zero_tol)
    h1_sig = h1_obstruction_signal(lap.L_F, section, zero_tol=zero_tol)

    # ── Step 2: Edge inconsistency vector b = δ⁰s ───────────────────────────
    m = len(graph.edges)
    if m == 0:
        return H2Result(
            h2_dim=0, h2_signal=0.0, n_triangles=0,
            unhedgeable=np.array([]), top_triangles=np.array([]),
            is_h2_event=False, h2_threshold=h2_threshold, h1_signal=h1_sig,
        )

    b = lap.delta0 @ np.array(section, dtype=float)  # (n_edges,)

    # ── Step 3: Find triangles and build δ¹ ──────────────────────────────────
    triangles = find_triangles(graph)
    n_tri = len(triangles)

    if n_tri == 0:
        # No triangles → H² is trivially zero (no 3-cycles to be inconsistent)
        return H2Result(
            h2_dim=0, h2_signal=0.0, n_triangles=0,
            unhedgeable=np.array([]),
            top_triangles=np.array([], dtype=int),
            is_h2_event=False, h2_threshold=h2_threshold, h1_signal=h1_sig,
        )

    D1 = delta1_matrix(graph, triangles)  # (n_tri, n_edges)

    # ── Step 4: Project b onto im(δ¹)ᵀ and compute residual ─────────────────
    # im(δ¹)ᵀ = column space of D1.T = row space of D1
    # Projection of b onto row space of D1:
    #   P b = D1.T (D1 D1.T)^+ D1 b
    # Residual (unhedgeable component):
    #   r = b - P b  ∈ ker(D1) = the part of b invisible to triangles

    # Use SVD of D1 for numerical stability
    U, s_vals, Vt = np.linalg.svd(D1, full_matrices=False)
    # Numerical rank
    rank = int(np.sum(s_vals > zero_tol))

    if rank == 0:
        # D1 is numerically zero — no triangle constraints active
        r = b.copy()
    else:
        # Projection onto row space of D1 (= col space of D1.T)
        Vt_r = Vt[:rank, :]            # (rank, n_edges) — row space basis
        b_proj = Vt_r.T @ (Vt_r @ b)  # projection of b onto row space
        r = b - b_proj                 # residual in ker(D1)

    # ── Step 5: H² dimension and signal ─────────────────────────────────────
    # Map residual back to triangles: unhedgeable inconsistency per triangle
    # = D1 @ r  (how much each triangle "sees" of the unhedgeable residual)
    unhedgeable_tri = D1 @ r           # (n_tri,)

    b_norm = float(np.linalg.norm(b))
    r_norm = float(np.linalg.norm(r))
    h2_sig = r_norm / (b_norm + eps)   # fraction of inconsistency that is irresolvable

    # H² dimension: rank of the unhedgeable subspace
    # = n_edges - rank(D1) - rank(L_F kernel)
    # Simplified: count significant components of r
    r_svd = np.linalg.svd(r.reshape(-1, 1), compute_uv=False)
    h2_dim = int(np.sum(r_svd > zero_tol))

    top_tri = np.argsort(np.abs(unhedgeable_tri))[::-1][:5]

    return H2Result(
        h2_dim        = h2_dim,
        h2_signal     = h2_sig,
        n_triangles   = n_tri,
        unhedgeable   = unhedgeable_tri,
        top_triangles = top_tri,
        is_h2_event   = h2_sig >= h2_threshold,
        h2_threshold  = h2_threshold,
        h1_signal     = h1_sig,
    )


# ---------------------------------------------------------------------------
# H⁰ — globally consistent section (explicit)
# ---------------------------------------------------------------------------

def h0_section(
    graph:    FinancialGraph,
    section:  np.ndarray,   # (n_nodes,)
    zero_tol: float = 1e-6,
) -> np.ndarray:
    """
    Return the H⁰ component of the section: its projection onto ker(L_F).

    H⁰(G, ℱ) = ker(δ⁰) = the space of globally consistent sections
    (agents whose bilateral exposure ratios are mutually consistent across
    the entire network, with no triangular inconsistency).

    The returned vector is the "consensual valuation" — the nearest globally
    consistent section to the observed health-ratio vector.

    Args:
        graph:    FinancialGraph
        section:  (n_nodes,) observed health ratios
        zero_tol: eigenvalue threshold defining ker(L_F)

    Returns:
        (n_nodes,) — the harmonic (globally consistent) part of section.
        The residual section − h0_section() is the H¹ obstruction.
    """
    lap = sheaf_laplacian(graph, section, zero_tol=zero_tol)
    s = np.array(section, dtype=float)
    eigs, vecs = np.linalg.eigh(lap.L_F)
    harm_mask = eigs < zero_tol
    if not np.any(harm_mask):
        return np.zeros_like(s)
    V_harm = vecs[:, harm_mask]
    return V_harm @ (V_harm.T @ s)


# ---------------------------------------------------------------------------
# Full cohomological report H⁰ / H¹ / H²
# ---------------------------------------------------------------------------

class CohomologyReport(NamedTuple):
    """
    Complete H⁰/H¹/H² cohomological risk report for a financial network.

    h0:           (n_nodes,) globally consistent section (bilateral solvency)
    h1_signal:    scalar H¹ obstruction (triangular inconsistency magnitude)
    h1_dim:       dim H¹ (number of independent inconsistency cycles)
    h2_signal:    scalar H² obstruction (systemic cascade magnitude)
    h2_dim:       dim H² (unhedgeable residual dimension)
    is_h2_event:  True if the network is in irresolvable systemic cascade
    risk_level:   'bilateral' | 'triangular' | 'systemic' — the highest
                  active risk level
    summary:      one-line human-readable risk assessment
    """
    h0:          np.ndarray
    h1_signal:   float
    h1_dim:      int
    h2_signal:   float
    h2_dim:      int
    is_h2_event: bool
    risk_level:  str
    summary:     str


def cohomology_report(
    graph:         FinancialGraph,
    section:       np.ndarray,
    h1_threshold:  float = 0.05,
    h2_threshold:  float = 0.10,
    zero_tol:      float = 1e-6,
) -> CohomologyReport:
    """
    Compute the full H⁰/H¹/H² cohomological risk report.

    This is the primary public API for systemic risk assessment using
    sheaf cohomology. It returns a structured report covering all three
    risk levels simultaneously.

    Args:
        graph:          FinancialGraph with bilateral exposures
        section:        (n_nodes,) health ratios (capital ratios etc.)
        h1_threshold:   H¹ signal above which triangular risk is flagged
        h2_threshold:   H² signal above which systemic cascade is flagged
        zero_tol:       eigenvalue threshold for harmonic subspace

    Returns:
        CohomologyReport with H⁰ section, H¹/H² signals, and risk level

    Example:
        graph  = FinancialGraph.from_matrix(A_interbank, total_assets)
        report = cohomology_report(graph, capital_ratios)
        print(report.summary)
        # → "SYSTEMIC: H² event — unhedgeable cascade (H²=0.18, H¹=0.43)"
    """
    h0 = h0_section(graph, section, zero_tol=zero_tol)

    lap = sheaf_laplacian(graph, section, zero_tol=zero_tol)
    h1_sig = h1_obstruction_signal(lap.L_F, section, zero_tol=zero_tol)
    h1_dim = lap.h1_dim

    h2_res = h2_obstruction(graph, section, h2_threshold=h2_threshold,
                             zero_tol=zero_tol)

    # Risk level classification
    if h2_res.is_h2_event:
        risk_level = 'systemic'
        summary = (
            f"SYSTEMIC: H² event — unhedgeable cascade "
            f"(H²={h2_res.h2_signal:.2f}, H¹={h1_sig:.2f})"
        )
    elif h1_sig >= h1_threshold:
        risk_level = 'triangular'
        summary = (
            f"TRIANGULAR: H¹ obstruction — inconsistent bilateral "
            f"exposures (H¹={h1_sig:.2f}, H²={h2_res.h2_signal:.2f})"
        )
    else:
        risk_level = 'bilateral'
        summary = (
            f"BILATERAL: no systemic obstruction "
            f"(H¹={h1_sig:.2f}, H²={h2_res.h2_signal:.2f})"
        )

    return CohomologyReport(
        h0          = h0,
        h1_signal   = h1_sig,
        h1_dim      = h1_dim,
        h2_signal   = h2_res.h2_signal,
        h2_dim      = h2_res.h2_dim,
        is_h2_event = h2_res.is_h2_event,
        risk_level  = risk_level,
        summary     = summary,
    )
