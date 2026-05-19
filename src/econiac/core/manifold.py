"""
Pacioli manifold — the directed graph of institutional money flows.

The Pacioli manifold is a directed graph (V, E) where:
  - Nodes V are accounts/sectors (households, firms, banks, government, ...)
  - Edges E are flows (wages, taxes, loans, dividends, ...)
  - The boundary operator ∂: C₁ → C₀ maps each edge to (head - tail)
  - The fundamental invariant ∂²=0 encodes double-entry bookkeeping

∂²=0 means: every flow that enters a node also exits somewhere.
This is not a constraint we impose — it is the definition of the manifold.
A model that violates it is a type error.

The incidence matrix B (n_nodes × n_edges) encodes ∂:
  B[i, e] = +1  if edge e flows INTO node i
  B[i, e] = -1  if edge e flows OUT OF node i
  B[i, e] =  0  otherwise

Stock-flow consistency: B @ flows = 0 (net flow at every node is zero).

Homology:
  H₀ = ker(∂₀) / im(∂₁)  — connected components (disconnected sectors)
  H₁ = ker(∂₁) / im(∂₂)  — independent financial cycles

References:
    Buckley (2026) Topology of Conservation.  doi:10.5281/zenodo.20234853
    Buckley (2026) Economic Gauge Theory.     doi:10.5281/zenodo.20259495
    Godley & Lavoie (2007) Monetary Economics.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# BalanceSheet — stocks (positions at a point in time)
# ---------------------------------------------------------------------------

@dataclass
class BalanceSheet:
    """
    A sectoral balance sheet matrix.

    positions[i, j] = net position of sector i in instrument j.
    Positive = asset, negative = liability.

    The stock-flow consistency condition:
        positions.sum(axis=0) ≈ 0
    Every asset is someone else's liability.
    """
    positions: jax.Array        # shape (n_sectors, n_instruments)
    sectors: list[str]
    instruments: list[str]

    def __post_init__(self):
        n_s = len(self.sectors)
        n_i = len(self.instruments)
        if self.positions.shape != (n_s, n_i):
            raise ValueError(
                f"positions shape {self.positions.shape} != "
                f"({n_s} sectors, {n_i} instruments)"
            )

    @property
    def n_sectors(self) -> int:
        return len(self.sectors)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    def column_sums(self) -> jax.Array:
        """Net position per instrument across all sectors. Should be ≈ 0."""
        return self.positions.sum(axis=0)

    def is_consistent(self, atol: float = 1e-6) -> bool:
        """True iff ∂²=0: every asset has a corresponding liability."""
        return bool(jnp.allclose(self.column_sums(), jnp.zeros(self.n_instruments), atol=atol))

    def net_worth(self) -> jax.Array:
        """Net financial worth of each sector (row sums)."""
        return self.positions.sum(axis=1)

    def __repr__(self) -> str:
        consistent = "✓ consistent" if self.is_consistent() else "✗ INCONSISTENT"
        return (
            f"BalanceSheet({self.n_sectors} sectors × {self.n_instruments} instruments, "
            f"{consistent})"
        )


# ---------------------------------------------------------------------------
# GodleyTable — flows (changes per period)
# ---------------------------------------------------------------------------

@dataclass
class GodleyTable:
    """
    A Godley table: the flow analogue of the BalanceSheet.

    flows[i, j] = net flow into sector i from instrument j in one period.
    Column sums must be zero: every source of funds has a use of funds.

    Updating a BalanceSheet by one period:
        new_positions = old_positions + flows
    """
    flows: jax.Array            # shape (n_sectors, n_instruments)
    sectors: list[str]
    instruments: list[str]

    def __post_init__(self):
        n_s = len(self.sectors)
        n_i = len(self.instruments)
        if self.flows.shape != (n_s, n_i):
            raise ValueError(
                f"flows shape {self.flows.shape} != ({n_s}, {n_i})"
            )

    def column_sums(self) -> jax.Array:
        return self.flows.sum(axis=0)

    def is_consistent(self, atol: float = 1e-6) -> bool:
        return bool(jnp.allclose(self.column_sums(), jnp.zeros(len(self.instruments)), atol=atol))

    def apply(self, balance_sheet: BalanceSheet) -> BalanceSheet:
        """Advance the balance sheet by one period."""
        return BalanceSheet(
            positions=balance_sheet.positions + self.flows,
            sectors=balance_sheet.sectors,
            instruments=balance_sheet.instruments,
        )


# ---------------------------------------------------------------------------
# PacioliManifold — the directed graph with ∂²=0
# ---------------------------------------------------------------------------

class HomologyGroups(NamedTuple):
    """Homology groups of the Pacioli manifold."""
    H0_rank: int    # number of connected components
    H1_rank: int    # number of independent financial cycles
    betti_0: int    # alias for H0_rank
    betti_1: int    # alias for H1_rank


@dataclass
class PacioliManifold:
    """
    The Pacioli manifold: a directed graph of institutional money flows.

    Encodes the flow network as an incidence matrix B (n_nodes × n_edges):
        B[i, e] = +1  if edge e flows into node i
        B[i, e] = -1  if edge e flows out of node i
        B[i, e] =  0  otherwise

    The fundamental invariant ∂²=0 is B @ B.T having zero off-diagonal
    structure consistent with the chain complex — enforced at construction.

    For most financial applications use PacioliManifold.from_edges() or
    PacioliManifold.from_godley_table().
    """
    incidence: jax.Array        # shape (n_nodes, n_edges)
    nodes: list[str]
    edges: list[str]

    def __post_init__(self):
        n, e = len(self.nodes), len(self.edges)
        if self.incidence.shape != (n, e):
            raise ValueError(
                f"incidence shape {self.incidence.shape} != ({n} nodes, {e} edges)"
            )
        # Each column must sum to zero: every edge has exactly one source and one sink
        col_sums = self.incidence.sum(axis=0)
        if not jnp.allclose(col_sums, jnp.zeros(e), atol=1e-6):
            raise ValueError(
                "Incidence matrix columns must sum to zero (∂²=0 violated). "
                f"Column sums: {col_sums}"
            )

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @staticmethod
    def from_edges(
        nodes: list[str],
        edges: list[tuple[str, str, str]],
    ) -> 'PacioliManifold':
        """
        Construct from a list of (name, source_node, target_node) triples.

        Example:
            PacioliManifold.from_edges(
                nodes=['households', 'firms', 'banks'],
                edges=[
                    ('wages',    'firms',      'households'),
                    ('deposits', 'households', 'banks'),
                    ('loans',    'banks',      'firms'),
                ]
            )
        """
        node_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)
        n_edges = len(edges)
        B = np.zeros((n_nodes, n_edges), dtype=float)
        edge_names = []
        for e_idx, (name, src, tgt) in enumerate(edges):
            B[node_idx[tgt], e_idx] = +1.0   # flows INTO target
            B[node_idx[src], e_idx] = -1.0   # flows OUT OF source
            edge_names.append(name)
        return PacioliManifold(
            incidence=jnp.array(B),
            nodes=nodes,
            edges=edge_names,
        )

    @staticmethod
    def from_godley_table(balance_sheet: BalanceSheet) -> 'PacioliManifold':
        """
        Derive the Pacioli manifold from a BalanceSheet.

        Each instrument becomes an edge connecting the sector with the
        largest positive position (asset holder) to the sector with the
        largest negative position (liability issuer). For multi-sector
        instruments this is an approximation — use from_edges() for
        precise bilateral flow specification.
        """
        nodes = balance_sheet.sectors
        edges = []
        edge_names = []
        n_s = len(nodes)
        n_i = balance_sheet.n_instruments
        B = np.zeros((n_s, n_i), dtype=float)

        for j, instrument in enumerate(balance_sheet.instruments):
            col = np.array(balance_sheet.positions[:, j])
            # Asset holder: max positive position
            # Liability issuer: most negative position
            asset_holder = int(np.argmax(col))
            liability_issuer = int(np.argmin(col))
            if asset_holder != liability_issuer:
                B[asset_holder, j] = +1.0
                B[liability_issuer, j] = -1.0
            edge_names.append(instrument)

        return PacioliManifold(
            incidence=jnp.array(B),
            nodes=nodes,
            edges=edge_names,
        )

    def boundary(self, flows: jax.Array) -> jax.Array:
        """
        Apply the boundary operator ∂: C₁ → C₀.

        Returns net flow at each node. Should be zero for stock-flow
        consistent flows (∂(flows) = 0).

        Args:
            flows: shape (n_edges,) — flow magnitude on each edge

        Returns:
            shape (n_nodes,) — net inflow at each node
        """
        return self.incidence @ flows

    def is_consistent(self, flows: jax.Array, atol: float = 1e-6) -> bool:
        """True iff the given flows satisfy ∂(flows) = 0."""
        return bool(jnp.allclose(self.boundary(flows), jnp.zeros(self.n_nodes), atol=atol))

    def homology(self) -> HomologyGroups:
        """
        Compute the Betti numbers (H₀, H₁) of the manifold.

        Uses the rank-nullity theorem on the incidence matrix:
            H₀ rank = n_nodes - rank(B)       connected components
            H₁ rank = n_edges - rank(B)       independent cycles
        """
        B = np.array(self.incidence)
        rank = np.linalg.matrix_rank(B)
        h0 = self.n_nodes - rank
        h1 = self.n_edges - rank
        return HomologyGroups(H0_rank=h0, H1_rank=h1, betti_0=h0, betti_1=h1)

    def laplacian(self) -> jax.Array:
        """
        Graph Laplacian L = B @ B.T.

        Eigenvalues of L encode the spectral geometry of the flow network.
        Zero eigenvalues correspond to connected components (H₀).
        """
        return self.incidence @ self.incidence.T

    def __repr__(self) -> str:
        h = self.homology()
        return (
            f"PacioliManifold({self.n_nodes} nodes, {self.n_edges} edges, "
            f"H₀={h.H0_rank}, H₁={h.H1_rank})"
        )


# ---------------------------------------------------------------------------
# Mechanism 2: CurvedBalanceSheet — non-flat Pacioli manifold
# ---------------------------------------------------------------------------

@dataclass
class CurvedBalanceSheet:
    """
    A balance sheet on a curved Pacioli manifold.

    Extends BalanceSheet with a curvature field F (field strength tensor)
    representing the failure of flatness:

        ∂²(bs) = F(bs)    instead of 0

    F is a skew-symmetric (n_instruments,) vector of curvature values —
    one per instrument column. When F=0 everywhere, this reduces to a
    flat BalanceSheet and the standard PCL type system applies.

    Physical interpretation:
    - F=0: no arbitrage; the manifold is flat.
    - F≠0: genuine arbitrage or risk premium at this point; the manifold
      has non-zero field strength. The magnitude ||F|| is the arbitrage
      profit per unit of capital deployed on a round trip.

    Relation to gauge theory:
        F = dA + A∧A
    where A is the connection (exchange-rate gauge field). Non-zero F
    means parallel transport around a closed loop returns a different
    value — the holonomy is non-trivial.

    Args:
        positions:   shape (n_sectors, n_instruments)
        sectors:     list of sector names
        instruments: list of instrument names
        curvature:   shape (n_instruments,) — field strength per instrument;
                     zero recovers flat BalanceSheet behaviour
    """
    positions:   jax.Array
    sectors:     list[str]
    instruments: list[str]
    curvature:   jax.Array      # shape (n_instruments,); zero = flat

    def __post_init__(self):
        n_s = len(self.sectors)
        n_i = len(self.instruments)
        if self.positions.shape != (n_s, n_i):
            raise ValueError(
                f"positions shape {self.positions.shape} != ({n_s}, {n_i})"
            )
        if self.curvature.shape != (n_i,):
            raise ValueError(
                f"curvature shape {self.curvature.shape} != ({n_i},)"
            )

    def column_sums(self) -> jax.Array:
        """Net position per instrument. Equals curvature F when non-flat."""
        return self.positions.sum(axis=0)

    def is_consistent(self, atol: float = 1e-6) -> bool:
        """True iff column sums equal the curvature field (generalised ∂²=0)."""
        return bool(jnp.allclose(self.column_sums(), self.curvature, atol=atol))

    def is_flat(self, atol: float = 1e-6) -> bool:
        """True iff the curvature field is zero (standard PCL flatness)."""
        return bool(jnp.allclose(self.curvature, jnp.zeros_like(self.curvature), atol=atol))

    def net_worth(self) -> jax.Array:
        """Net financial worth of each sector (row sums)."""
        return self.positions.sum(axis=1)

    def to_flat(self) -> 'BalanceSheet':
        """
        Project to a flat BalanceSheet by zeroing the curvature.

        The projection adjusts the last sector's positions so that column
        sums are exactly zero. Use when you need PCL combinators (which
        require flat inputs) but want to work near a curved manifold.
        """
        adjustment = self.column_sums()  # = F
        new_positions = self.positions.at[-1].add(-adjustment)
        return BalanceSheet(
            positions=new_positions,
            sectors=self.sectors,
            instruments=self.instruments,
        )

    def __repr__(self) -> str:
        flat = "flat" if self.is_flat() else f"curved ||F||={float(jnp.linalg.norm(self.curvature)):.4g}"
        return (
            f"CurvedBalanceSheet({len(self.sectors)} sectors × "
            f"{len(self.instruments)} instruments, {flat})"
        )


def holonomy(
    computation: 'Computation',
    curved_bs: CurvedBalanceSheet,
) -> jax.Array:
    """
    Compute the holonomy of a computation on a curved balance sheet.

    Holonomy measures path-dependence: the residual after applying
    `computation` and returning to the start. On a flat manifold,
    holonomy is zero. On a curved manifold, it equals the integrated
    field strength around the loop — the arbitrage profit.

    Args:
        computation: a PCL Computation (BalanceSheet → BalanceSheet)
        curved_bs:   the base point on the curved manifold

    Returns:
        shape (n_instruments,) — holonomy per instrument;
        zero means no arbitrage; non-zero magnitude = arbitrage profit
    """
    from econiac.pcl.combinators import Computation  # avoid circular import
    flat_in  = curved_bs.to_flat()
    flat_out = computation(flat_in)
    # Holonomy = column sums of result - column sums of input (both should be 0 on flat)
    # On curved manifold: measures how much the curvature F is exposed by this path
    return flat_out.column_sums() - flat_in.column_sums() + curved_bs.curvature


# ---------------------------------------------------------------------------
# Mechanism 1: Residual sector utilities
# ---------------------------------------------------------------------------

RESIDUAL_SECTOR = "_residual"
FLOAT_SECTOR    = "_float"


def add_residual_sector(bs: BalanceSheet) -> BalanceSheet:
    """
    Extend a BalanceSheet with a `_residual` sector for discrepancy accounting.

    The residual sector absorbs any imbalance in the existing positions,
    making the extended balance sheet exactly flat (∂²=0). Its initial
    positions are set to -column_sums(bs) so that the total column sums
    are zero.

    Usage:
        bs_raw = ...  # national accounts data (may not balance exactly)
        bs     = add_residual_sector(bs_raw)
        assert bs.is_consistent()  # guaranteed

    The residual sector's net position is observable: large values indicate
    either data quality issues or model misspecification.

    Args:
        bs: the original (potentially inconsistent) balance sheet

    Returns:
        a new BalanceSheet with one extra sector ('_residual') that
        exactly absorbs any imbalance
    """
    imbalance    = bs.column_sums()                     # shape (n_instruments,)
    residual_row = (-imbalance).reshape(1, bs.n_instruments)
    new_positions = jnp.concatenate([bs.positions, residual_row], axis=0)
    return BalanceSheet(
        positions=new_positions,
        sectors=bs.sectors + [RESIDUAL_SECTOR],
        instruments=bs.instruments,
    )


def add_float_sector(bs: BalanceSheet) -> BalanceSheet:
    """
    Extend a BalanceSheet with a `_float` sector for in-transit payments.

    The float sector acts as a temporary holding account for payments that
    have been sent but not yet received. Its initial balance is zero.

    Typical usage:
        # Payment sent (sender's books updated immediately):
        send    = flow("payer", "_float", "deposits", amount)
        # Payment received (receiver's books updated at settlement):
        receive = flow("_float", "payee",  "deposits", amount)
        # sequence(send, receive) has zero net float — the float clears.

    Args:
        bs: the original balance sheet

    Returns:
        a new BalanceSheet with one extra sector ('_float') initialised to zero
    """
    float_row    = jnp.zeros((1, bs.n_instruments))
    new_positions = jnp.concatenate([bs.positions, float_row], axis=0)
    return BalanceSheet(
        positions=new_positions,
        sectors=bs.sectors + [FLOAT_SECTOR],
        instruments=bs.instruments,
    )


def residual_magnitude(bs: BalanceSheet) -> float:
    """
    Return the L2 norm of the `_residual` sector's positions.

    Zero means perfect conservation; large values indicate data discrepancy
    or model misspecification. Use as a diagnostic or calibration loss term.

    Returns 0.0 if the balance sheet has no `_residual` sector.
    """
    if RESIDUAL_SECTOR not in bs.sectors:
        return 0.0
    idx = bs.sectors.index(RESIDUAL_SECTOR)
    return float(jnp.linalg.norm(bs.positions[idx]))


# ---------------------------------------------------------------------------
# Convenience constructors for common economic structures
# ---------------------------------------------------------------------------

def three_sector_sfc(
    household_deposits: float = 100.0,
    firm_loans: float = 80.0,
) -> tuple[BalanceSheet, PacioliManifold]:
    """
    Minimal three-sector stock-flow consistent model.

    Sectors:  households, firms, banks
    Instruments: deposits, loans

    The three flows form one closed financial cycle:
        wages:    firms → households
        deposits: households → banks
        loans:    banks → firms

    H₀ = 1 (connected), H₁ = 1 (one independent cycle).

    Returns a (BalanceSheet, PacioliManifold) pair ready for simulation.
    """
    sectors = ['households', 'firms', 'banks']
    instruments = ['deposits', 'loans']

    positions = jnp.array([
        # deposits            loans
        [ household_deposits,  0.0       ],  # households: asset (deposits)
        [ 0.0,                -firm_loans],  # firms:      liability (loans)
        [-household_deposits,  firm_loans],  # banks:      liability (deposits), asset (loans)
    ])

    bs = BalanceSheet(positions=positions, sectors=sectors, instruments=instruments)

    manifold = PacioliManifold.from_edges(
        nodes=sectors,
        edges=[
            ('wages',    'firms',      'households'),
            ('deposits', 'households', 'banks'     ),
            ('loans',    'banks',      'firms'     ),
        ],
    )

    return bs, manifold
