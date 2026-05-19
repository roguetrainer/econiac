"""
Connections on the Pacioli manifold: parallel transport, holonomy, curvature.

The gauge group is (ℝ₊, ×): all quantities are positive scalars (exchange rates,
discount factors, survival probabilities). Working in log-space maps this to (ℝ, +),
making the algebra linear.

A connection A assigns a log-rate A[i,j] to each ordered pair of nodes (i,j):
    A[i,j] = log(rate from node i to node j)

Parallel transport along a path i₀→i₁→⋯→iₖ:
    T = exp(A[i₀,i₁] + A[i₁,i₂] + ⋯ + A[iₖ₋₁,iₖ])

Holonomy of a closed loop (Wilson loop):
    Hol(γ) = exp(sum of A around γ)
    Hol = 1 ↔ no arbitrage ↔ flat connection on that loop

Curvature of an elementary triangle (i,j,k):
    F[i,j,k] = A[i,j] + A[j,k] - A[i,k]
    F = 0 everywhere ↔ connection is flat ↔ no arbitrage

References:
    Buckley (2026) Currency Bundles.       doi:10.5281/zenodo.20242355
    Buckley (2026) Economic Gauge Theory.  doi:10.5281/zenodo.20259495
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Connection — log-rates on a complete graph of nodes
# ---------------------------------------------------------------------------

@dataclass
class Connection:
    """
    A connection on the Pacioli manifold.

    log_rates[i, j] = log(transport factor from node i to node j).

    For exchange rates: log_rates[i,j] = log(S_ij) where S_ij is units of
    currency j per unit of currency i.

    For discount factors: log_rates[i,j] = -r_ij * dt (negative because
    transporting money forward in time costs interest).

    Antisymmetry: log_rates[i,j] = -log_rates[j,i] (reciprocal rates).
    This is enforced weakly — use Connection.from_rates() or
    Connection.from_symmetric() to guarantee it.
    """
    log_rates: jax.Array    # shape (n_nodes, n_nodes); diagonal = 0
    nodes: list[str]

    def __post_init__(self):
        n = len(self.nodes)
        if self.log_rates.shape != (n, n):
            raise ValueError(
                f"log_rates shape {self.log_rates.shape} != ({n}, {n})"
            )

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def rates(self) -> jax.Array:
        """Transport factors in multiplicative form: rates[i,j] = exp(log_rates[i,j])."""
        return jnp.exp(self.log_rates)

    def is_antisymmetric(self, atol: float = 1e-6) -> bool:
        """True iff log_rates[i,j] = -log_rates[j,i] (reciprocal rates)."""
        return bool(jnp.allclose(self.log_rates + self.log_rates.T,
                                  jnp.zeros_like(self.log_rates), atol=atol))

    @staticmethod
    def from_rates(rates: jax.Array, nodes: list[str]) -> 'Connection':
        """
        Construct from a matrix of positive transport factors.

        rates[i,j] = transport factor from node i to node j (e.g. spot FX rate).
        Diagonal entries are ignored (set to 1 internally).
        """
        n = len(nodes)
        if rates.shape != (n, n):
            raise ValueError(f"rates shape {rates.shape} != ({n}, {n})")
        log_r = jnp.log(rates)
        log_r = log_r.at[jnp.arange(n), jnp.arange(n)].set(0.0)
        return Connection(log_rates=log_r, nodes=nodes)

    @staticmethod
    def from_symmetric(rates: jax.Array, nodes: list[str]) -> 'Connection':
        """
        Construct an antisymmetric connection from the upper-triangle of rates.

        rates[i,j] for i<j gives the forward rate; rates[j,i] is set to 1/rates[i,j].
        This guarantees log_rates[i,j] = -log_rates[j,i].
        """
        n = len(nodes)
        log_r = jnp.log(rates)
        log_r = log_r.at[jnp.arange(n), jnp.arange(n)].set(0.0)
        # Antisymmetrise: take upper triangle, reflect with sign flip
        upper = jnp.triu(log_r, k=1)
        log_antisym = upper - upper.T
        return Connection(log_rates=log_antisym, nodes=nodes)

    def __repr__(self) -> str:
        flat = "flat" if self.is_flat() else "curved"
        return (
            f"Connection({self.n_nodes} nodes, {flat}, "
            f"antisymmetric={self.is_antisymmetric()})"
        )

    # Delegate flat-check here for repr; defined properly below
    def is_flat(self, atol: float = 1e-8) -> bool:
        return bool(jnp.allclose(curvature(self),
                                  jnp.zeros((self.n_nodes, self.n_nodes, self.n_nodes)),
                                  atol=atol))


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------

def parallel_transport(connection: Connection, path: Sequence[int]) -> jax.Array:
    """
    Parallel transport along a path of node indices.

    Args:
        connection: the connection on the manifold
        path: sequence of node indices [i₀, i₁, ..., iₖ]

    Returns:
        Scalar transport factor exp(Σ A[iₗ, iₗ₊₁]) ∈ ℝ₊.

    Example:
        transport = parallel_transport(conn, [0, 1, 2])
        # = exp(A[0,1] + A[1,2])
    """
    path = list(path)
    if len(path) < 2:
        return jnp.ones(())
    log_total = sum(
        connection.log_rates[path[k], path[k + 1]]
        for k in range(len(path) - 1)
    )
    return jnp.exp(log_total)


def wilson_loop(connection: Connection, loop: Sequence[int]) -> jax.Array:
    """
    Wilson loop: holonomy of a closed path.

    Args:
        connection: the connection
        loop: sequence of node indices forming a closed path; the last node
              need not equal the first — closure is automatic.

    Returns:
        Holonomy scalar Hol(γ) ∈ ℝ₊.
        Hol = 1  ↔  no arbitrage around this loop.
        Hol > 1  ↔  profit by going around the loop.
        Hol < 1  ↔  loss (going the other way yields profit).
    """
    loop = list(loop)
    closed = loop + [loop[0]]   # close the loop
    return parallel_transport(connection, closed)


def log_holonomy(connection: Connection, loop: Sequence[int]) -> jax.Array:
    """
    Log-holonomy of a closed path.

    Returns log(Hol(γ)) = sum of log-rates around the loop.
    Zero iff no arbitrage on this loop.
    """
    loop = list(loop)
    closed = loop + [loop[0]]
    log_total = sum(
        connection.log_rates[closed[k], closed[k + 1]]
        for k in range(len(closed) - 1)
    )
    return log_total


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def curvature(connection: Connection) -> jax.Array:
    """
    Curvature 2-form of the connection.

    F[i, j, k] = A[i,j] + A[j,k] - A[i,k]

    In multiplicative notation: F=0 iff S_ij · S_jk = S_ik (triangle consistency).

    Returns:
        Array of shape (n, n, n). F[i,i,k] = F[i,j,i] = 0 by construction.
        F = 0 everywhere ↔ connection is flat ↔ no triangular arbitrage.
    """
    A = connection.log_rates   # (n, n)
    # F[i,j,k] = A[i,j] + A[j,k] - A[i,k]
    # broadcast: A[i,j] → (n,n,1), A[j,k] → (1,n,n), A[i,k] → (n,1,n)
    Aij = A[:, :, None]        # (n, n, 1)
    Ajk = A[None, :, :]        # (1, n, n)
    Aik = A[:, None, :]        # (n, 1, n)
    return Aij + Ajk - Aik    # (n, n, n)


def is_flat(connection: Connection, atol: float = 1e-8) -> bool:
    """
    True iff the connection is flat: F = 0 everywhere.

    Flat connection ↔ no triangular arbitrage ↔ rates are consistent
    across all triangles in the node graph.
    """
    F = curvature(connection)
    return bool(jnp.allclose(F, jnp.zeros_like(F), atol=atol))


def max_curvature(connection: Connection) -> jax.Array:
    """
    Maximum absolute curvature across all triangles.

    Zero for a flat connection. Large values indicate strong arbitrage.
    """
    return jnp.max(jnp.abs(curvature(connection)))


def curvature_matrix(connection: Connection) -> jax.Array:
    """
    Reduce the (n,n,n) curvature tensor to an (n,n) antisymmetric matrix
    by summing over the middle index.

    C[i,k] = Σⱼ F[i,j,k]

    Useful for visualising which node pairs have the most curvature.
    """
    return curvature(connection).sum(axis=1)   # sum over j


# ---------------------------------------------------------------------------
# Gauge transformations
# ---------------------------------------------------------------------------

def gauge_transform(connection: Connection, log_lambda: jax.Array) -> Connection:
    """
    Apply a gauge transformation with parameter log_lambda.

    In multiplicative form: S_ij ↦ λ_i · S_ij · λ_j⁻¹
    In log form:            A_ij ↦ A_ij + log_λ_i - log_λ_j

    Args:
        connection: the connection to transform
        log_lambda: shape (n_nodes,) — log of the gauge parameter λ_i

    Returns:
        New Connection with transformed log_rates.
        Curvature is gauge-invariant (unchanged by construction).
    """
    n = connection.n_nodes
    # A'[i,j] = A[i,j] + log_lambda[i] - log_lambda[j]
    diff = log_lambda[:, None] - log_lambda[None, :]   # (n, n)
    new_log_rates = connection.log_rates + diff
    new_log_rates = new_log_rates.at[jnp.arange(n), jnp.arange(n)].set(0.0)
    return Connection(log_rates=new_log_rates, nodes=connection.nodes)


def flat_gauge(connection: Connection) -> Connection:
    """
    Find the gauge in which the connection appears as flat as possible.

    For a genuinely flat connection, this returns a connection whose
    log_rates[i,j] = log_lambda[i] - log_lambda[j] for some vector log_lambda.
    For a curved connection, this minimises the Frobenius norm of the off-diagonal
    residual by choosing log_lambda = first eigenvector of the graph Laplacian
    weighted by the log-rates.

    Useful for finding the 'risk-neutral measure' gauge in finance.
    """
    # The optimal log_lambda minimises ||A - (λ1ᵀ - 1λᵀ)||²
    # Solution: log_lambda = row-mean of A (i.e. average log-rate out of each node)
    log_lambda = connection.log_rates.mean(axis=1)
    return gauge_transform(connection, log_lambda - log_lambda.mean())


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def fx_connection(
    currencies: list[str],
    spot_rates: jax.Array,
) -> Connection:
    """
    Build a currency connection from a spot-rate matrix.

    spot_rates[i,j] = units of currencies[j] per unit of currencies[i].
    Upper triangle is used; lower triangle is set to 1/upper (antisymmetry).

    Example:
        conn = fx_connection(
            currencies=['USD', 'EUR', 'GBP'],
            spot_rates=jnp.array([
                [1.0,  0.91, 0.79],
                [1.10, 1.0,  0.87],
                [1.27, 1.15, 1.0 ],
            ])
        )
    """
    return Connection.from_symmetric(spot_rates, currencies)


def discount_connection(
    nodes: list[str],
    rates: jax.Array,
    dt: float = 1.0,
) -> Connection:
    """
    Build a temporal connection from a vector of interest rates.

    Each node i has rate rates[i]. Transporting value from node i to node j
    for time dt gives a factor exp((rates[j] - rates[i]) * dt).

    This connection is always flat (F=0): the log-rates are an exact 1-form
    derived from the scalar potential r_i. Curvature arises only when rates
    differ across paths — use fx_connection() for that.
    """
    n = len(nodes)
    if rates.shape != (n,):
        raise ValueError(f"rates shape {rates.shape} != ({n},)")
    # A[i,j] = (rates[j] - rates[i]) * dt
    log_r = (rates[None, :] - rates[:, None]) * dt
    log_r = log_r.at[jnp.arange(n), jnp.arange(n)].set(0.0)
    return Connection(log_rates=log_r, nodes=nodes)
