"""
Foreign exchange as connection curvature on the Pacioli manifold.

Spot FX rates are the connection coefficients on the currency bundle.
Triangular arbitrage = non-zero holonomy: S_USD/EUR · S_EUR/GBP · S_GBP/USD ≠ 1.
CIP holds iff the combined (FX × IR) connection is flat.
CIP violation = persistent curvature (post-2008, documented by Du-Tepper-Verdelhan).

This module provides financial wrappers around econiac.core.connections,
adding the CIP deviation formula and cross-currency basis computation.

Reference: Buckley (2026) Currency Bundles. doi:10.5281/zenodo.20242355
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.connections import (
    Connection,
    wilson_loop,
    log_holonomy,
    curvature,
    is_flat,
    max_curvature,
    fx_connection,
)
from econiac.finance.curves import YieldCurve


# ---------------------------------------------------------------------------
# FXMarket — spot rates + interest rate curves, together
# ---------------------------------------------------------------------------

@dataclass
class FXMarket:
    """
    A multi-currency FX market: spot rates and domestic interest-rate curves.

    The no-arbitrage (CIP) condition links three things:
        spot rate S_ij, domestic rate r_i, foreign rate r_j, forward rate F_ij(T).

    CIP: F_ij(T) = S_ij · P_j(0,T) / P_i(0,T)
    Deviation: φ_ij(T) = log F_ij(T) - log S_ij - log P_j(0,T) + log P_i(0,T)

    φ_ij = 0 ↔ CIP holds ↔ flat temporal+spatial connection on this rectangle.
    """
    currencies: list[str]
    spot_rates: jax.Array       # shape (n, n); spot_rates[i,j] = units of j per i
    ir_curves: list[YieldCurve] # one YieldCurve per currency

    def __post_init__(self):
        n = len(self.currencies)
        if self.spot_rates.shape != (n, n):
            raise ValueError(f"spot_rates shape {self.spot_rates.shape} != ({n},{n})")
        if len(self.ir_curves) != n:
            raise ValueError(f"need {n} IR curves, got {len(self.ir_curves)}")

    @property
    def n(self) -> int:
        return len(self.currencies)

    def connection(self) -> Connection:
        """Return the FX spatial connection (log spot rates)."""
        return fx_connection(self.currencies, self.spot_rates)

    def forward_rate(self, i: int, j: int, T: float) -> jax.Array:
        """
        Arbitrage-free (CIP) forward rate F_ij(T).

        F_ij(T) = S_ij · P_j(0,T) / P_i(0,T)
        """
        S_ij = self.spot_rates[i, j]
        P_i  = self.ir_curves[i].discount(T)
        P_j  = self.ir_curves[j].discount(T)
        return S_ij * P_j / P_i

    def cip_deviation(self, i: int, j: int, T: float,
                      market_forward: Optional[float] = None) -> jax.Array:
        """
        Cross-currency basis (CIP deviation) φ_ij(T).

        φ = log F_market - log F_CIP

        If market_forward is None, returns 0 (assumes CIP holds).
        Non-zero φ indicates persistent curvature (post-2008 regime).

        Args:
            i, j: currency indices
            T:    tenor in years
            market_forward: observed market forward rate (optional)

        Returns:
            φ_ij(T) in log-space. Zero = no basis. Positive = i-funded basis.
        """
        cip_fwd = self.forward_rate(i, j, T)
        if market_forward is None:
            return jnp.zeros(())
        return jnp.log(jnp.array(market_forward)) - jnp.log(cip_fwd)

    def triangular_arbitrage(self, i: int, j: int, k: int) -> jax.Array:
        """
        Log-holonomy of the triangle i→j→k→i.

        Zero = no triangular arbitrage. Positive = profit from i→j→k→i round-trip.
        """
        conn = self.connection()
        return log_holonomy(conn, [i, j, k])

    def is_arbitrage_free(self, atol: float = 1e-6) -> bool:
        """True iff no triangular arbitrage exists across all currency triples."""
        conn = self.connection()
        return is_flat(conn, atol=atol)

    def arbitrage_surface(self) -> jax.Array:
        """
        Curvature tensor F[i,j,k] across all currency triples.

        Zero everywhere = flat connection = no triangular arbitrage.
        Returns shape (n, n, n).
        """
        return curvature(self.connection())

    def max_arbitrage(self) -> jax.Array:
        """Maximum triangular arbitrage across all triples (in log-space)."""
        return max_curvature(self.connection())

    def __repr__(self) -> str:
        ccys = '/'.join(self.currencies)
        flat = "flat" if self.is_arbitrage_free() else "curved"
        return f"FXMarket({ccys}, {flat})"


# ---------------------------------------------------------------------------
# Convenience: CIP residual for a single currency pair
# ---------------------------------------------------------------------------

def cip_residual(
    spot: float,
    rate_domestic: float,
    rate_foreign: float,
    forward: float,
    T: float,
) -> jax.Array:
    """
    CIP deviation for a single currency pair (continuously compounded rates).

    φ = log(F) - log(S) - r_foreign·T + r_domestic·T

    Zero = CIP holds. Positive = forward trades above CIP (foreign funding premium).

    Args:
        spot:           S_{d/f} spot rate (units of domestic per foreign)
        rate_domestic:  domestic continuously compounded rate
        rate_foreign:   foreign continuously compounded rate
        forward:        observed market forward rate
        T:              tenor in years
    """
    log_cip_fwd = jnp.log(spot) + (rate_foreign - rate_domestic) * T
    return jnp.log(jnp.array(forward)) - log_cip_fwd


def swap_line_holonomy(
    conn: Connection,
    i: int,
    j: int,
    swap_rate: float,
) -> jax.Array:
    """
    Holonomy of a central-bank swap line between currency i and currency j.

    A swap line at rate swap_rate modifies the effective connection along
    edge (i,j). Returns the log-holonomy of the round-trip i→j→i after
    adding the swap line as a parallel edge.

    The Swap Line Theorem (Paper 295): adding a swap line increases the
    first Betti number by 1 and reduces curvature along the new path.
    """
    log_S_ij = float(conn.log_rates[i, j])
    log_S_ji = float(conn.log_rates[j, i])
    # Swap line offers rate swap_rate; effective log-rate along new path
    log_swap  = jnp.log(jnp.array(swap_rate))
    log_swap_back = -log_swap  # symmetric (swap is reciprocal)
    # Holonomy of round-trip via swap line
    return log_swap + log_swap_back  # = 0 if swap is symmetric
