"""
Credit risk: survival probabilities as parallel transport; CVA as curvature integral.

Forward hazard rate h(t,T) is the credit connection coefficient.
Survival probability Q(τ>T) = exp(-∫_0^T h(0,u) du) is parallel transport.
CVA = LGD · ∫_0^T D(t) · (-dQ(τ>t)/dt) dt — integral of exposure-weighted default density.

Exact formal parallel to yield curves (Paper 296):
    f(t,T) → h(t,T)     (forward rate → forward hazard rate)
    P(t,T) → Q(τ>T|τ>t) (discount factor → conditional survival probability)
    No-arbitrage → hazard-rate HJM drift (flatness under survival measure)

References:
    Buckley (2026) Credit Bundles. doi:10.5281/zenodo.20257596
    Buckley (2026) XVA as Curvature. doi:10.5281/zenodo.20257724
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.finance.curves import YieldCurve


# ---------------------------------------------------------------------------
# HazardRateConnection — the credit connection
# ---------------------------------------------------------------------------

@dataclass
class HazardRateConnection:
    """
    A credit connection: forward hazard rates on a maturity grid.

    hazard_rates[k] = h(0, maturities[k]) — the forward hazard rate (default
    intensity) at maturity T_k, observed today.

    Survival probability to T:
        Q(τ > T) = exp(-∫_0^T h(0,u) du)

    This is formally identical to a yield curve with r replaced by h.
    The credit fibre sits above the same tenor lattice as the IR fibre.
    """
    maturities: jax.Array     # shape (n,), strictly increasing, > 0
    hazard_rates: jax.Array   # shape (n,), ≥ 0 (hazard rates are non-negative)
    name: str = "obligor"

    def __post_init__(self):
        n = len(self.maturities)
        if self.hazard_rates.shape != (n,):
            raise ValueError(
                f"hazard_rates shape {self.hazard_rates.shape} != ({n},)"
            )
        if not bool(jnp.all(self.hazard_rates >= 0)):
            raise ValueError(
                "hazard_rates must be non-negative. "
                "Negative hazard rates violate the gauge group (R+,×)."
            )

    @property
    def n(self) -> int:
        return len(self.maturities)

    def cumulative_hazard(self, T: float) -> jax.Array:
        """
        Cumulative hazard Λ(T) = ∫_0^T h(0,u) du.

        Trapezoidal integration on the maturity grid.
        """
        mats = np.array(self.maturities)
        haz  = np.array(self.hazard_rates)
        # Interpolate at T
        dt = np.diff(mats, prepend=0.0)
        cum_haz = np.cumsum(haz * dt)
        if T <= 0.0:
            return jnp.zeros(())
        return jnp.array(float(np.interp(T, mats, cum_haz)))

    def survival_probability(self, T: float) -> jax.Array:
        """
        Survival probability Q(τ > T) = exp(-Λ(T)).

        Parallel transport along the credit fibre from T to 0.
        Formally: P^credit(0,T) in the yield-curve analogy.
        """
        return jnp.exp(-self.cumulative_hazard(T))

    def conditional_survival(self, T1: float, T2: float) -> jax.Array:
        """
        Conditional survival Q(τ > T2 | τ > T1) = Q(τ>T2) / Q(τ>T1).

        Forward survival probability = forward discount on the credit fibre.
        """
        return self.survival_probability(T2) / self.survival_probability(T1)

    def default_density(self, T: float, dT: float = 1e-4) -> jax.Array:
        """
        Default density f_τ(T) = -dQ(τ>T)/dT = h(0,T) · Q(τ>T).

        Probability of defaulting in [T, T+dT].
        """
        sp1 = self.survival_probability(T + dT / 2)
        sp0 = self.survival_probability(T - dT / 2)
        return -(sp1 - sp0) / dT

    def temporal_curvature(self, T1: float, T2: float) -> jax.Array:
        """
        Temporal curvature on the credit fibre.

        F = log Q(τ>T1) + log Q(τ>T2|τ>T1) - log Q(τ>T2).
        Zero by construction (identical to YieldCurve.temporal_curvature).
        """
        return (
            jnp.log(self.survival_probability(T1))
            + jnp.log(self.conditional_survival(T1, T2))
            - jnp.log(self.survival_probability(T2))
        )

    @staticmethod
    def flat(hazard_rate: float, maturities: jax.Array,
             name: str = "obligor") -> 'HazardRateConnection':
        """Flat hazard rate: constant default intensity."""
        mats = jnp.array(maturities)
        haz  = jnp.full_like(mats, hazard_rate)
        return HazardRateConnection(maturities=mats, hazard_rates=haz, name=name)

    @staticmethod
    def from_survival_probabilities(
        maturities: jax.Array,
        survival_probs: jax.Array,
        name: str = "obligor",
    ) -> 'HazardRateConnection':
        """
        Construct from observed survival probabilities (e.g. from CDS spreads).

        Infers forward hazard rates from -d/dT log Q(τ>T).
        """
        mats = np.array(maturities)
        sps  = np.array(survival_probs)
        log_sp = np.log(sps)
        # Forward hazard rates: h(T) ≈ -d(log Q)/dT
        haz = -np.gradient(log_sp, mats)
        haz = np.maximum(haz, 0.0)   # enforce non-negativity
        return HazardRateConnection(
            maturities=jnp.array(mats),
            hazard_rates=jnp.array(haz),
            name=name,
        )

    def __repr__(self) -> str:
        h_mean = float(jnp.mean(self.hazard_rates))
        sp_5y  = float(self.survival_probability(min(5.0, float(self.maturities[-1]))))
        return (
            f"HazardRateConnection('{self.name}', {self.n} tenors, "
            f"h_mean={h_mean:.2%}, Q(5y)={sp_5y:.3f})"
        )


# ---------------------------------------------------------------------------
# CVA — credit valuation adjustment as a curvature integral
# ---------------------------------------------------------------------------

class CVAResult(NamedTuple):
    """Result of a CVA computation."""
    cva: float          # Credit Valuation Adjustment (always ≥ 0)
    lgd: float          # Loss Given Default used
    expected_exposure: jax.Array   # EE(t) profile, shape (n_grid,)
    default_density: jax.Array     # f_τ(t) profile, shape (n_grid,)
    discount_factors: jax.Array    # D(t) = P(0,t) profile, shape (n_grid,)
    grid: jax.Array                # integration grid, shape (n_grid,)


def cva(
    exposure_profile: jax.Array,
    hazard_connection: HazardRateConnection,
    discount_curve: YieldCurve,
    lgd: float = 0.6,
    grid: Optional[jax.Array] = None,
) -> CVAResult:
    """
    CVA = LGD · ∫_0^T EE(t) · D(t) · f_τ(t) dt

    where:
        EE(t)  = Expected Exposure at time t
        D(t)   = P(0,t) risk-free discount factor
        f_τ(t) = h(t) · Q(τ>t) default density
        LGD    = Loss Given Default (1 - Recovery)

    This is the integral of the joint curvature contribution over the trade
    lifetime: CVA is the curvature penalty from the counterparty credit fibre.

    Args:
        exposure_profile: shape (n,) — EE(t_k) at a grid of times
        hazard_connection: the counterparty's credit connection
        discount_curve:   the risk-free yield curve
        lgd:              loss given default ∈ (0,1]
        grid:             shape (n,) — time grid matching exposure_profile;
                          defaults to hazard_connection.maturities

    Returns:
        CVAResult namedtuple with cva scalar and component profiles.
    """
    if grid is None:
        grid = hazard_connection.maturities

    grid_np = np.array(grid)
    n = len(grid_np)

    # Build profiles on the integration grid
    disc   = jnp.array([float(discount_curve.discount(float(t))) for t in grid_np])
    surv   = jnp.array([float(hazard_connection.survival_probability(float(t))) for t in grid_np])
    haz    = jnp.array([float(np.interp(t, np.array(hazard_connection.maturities),
                                         np.array(hazard_connection.hazard_rates)))
                        for t in grid_np])
    f_tau  = haz * surv   # default density

    # Ensure exposure_profile matches grid length
    if len(exposure_profile) != n:
        raise ValueError(
            f"exposure_profile length {len(exposure_profile)} != grid length {n}"
        )

    # CVA integral via trapezoidal rule
    integrand = jnp.array(exposure_profile) * disc * f_tau
    cva_value = float(lgd * jnp.trapezoid(integrand, jnp.array(grid_np)))

    return CVAResult(
        cva=cva_value,
        lgd=lgd,
        expected_exposure=jnp.array(exposure_profile),
        default_density=f_tau,
        discount_factors=disc,
        grid=jnp.array(grid_np),
    )


# ---------------------------------------------------------------------------
# XVA decomposition
# ---------------------------------------------------------------------------

class XVAResult(NamedTuple):
    """XVA breakdown: each adjustment is a curvature contribution."""
    cva: float    # counterparty credit
    dva: float    # own credit (benefit; usually negative sign)
    fva: float    # funding
    total: float  # CVA + DVA + FVA (first-order, ignoring off-diagonal curvature)


def xva(
    exposure_profile: jax.Array,
    counterparty_hazard: HazardRateConnection,
    own_hazard: HazardRateConnection,
    discount_curve: YieldCurve,
    funding_spread: float = 0.0,
    lgd_counterparty: float = 0.6,
    lgd_own: float = 0.6,
    grid: Optional[jax.Array] = None,
) -> XVAResult:
    """
    First-order XVA decomposition.

    CVA: cost of counterparty default (curvature on counterparty credit fibre).
    DVA: benefit of own default (curvature on own-credit fibre; negative sign).
    FVA: funding cost above risk-free (curvature on funding fibre).

    The off-diagonal curvature terms (CVA×FVA, DVA×FVA correlations) are
    not included here — see Paper 299 for the full non-additive XVA tensor.

    Args:
        exposure_profile:    EE(t) on the grid
        counterparty_hazard: counterparty credit connection
        own_hazard:          bank's own credit connection
        discount_curve:      risk-free yield curve
        funding_spread:      s_FVA — funding spread above risk-free (flat)
        lgd_counterparty, lgd_own: loss given default for each party
        grid:                integration time grid
    """
    if grid is None:
        grid = counterparty_hazard.maturities

    cva_result = cva(exposure_profile, counterparty_hazard, discount_curve,
                     lgd=lgd_counterparty, grid=grid)

    # DVA: symmetric to CVA but on own-credit fibre; benefit = negative cost
    # Exposure from own perspective is -exposure_profile (if you default, counterparty loses)
    negative_exposure = jnp.maximum(-jnp.array(exposure_profile), 0.0)
    dva_result = cva(negative_exposure, own_hazard, discount_curve,
                     lgd=lgd_own, grid=grid)
    dva_value  = -dva_result.cva   # DVA is a benefit

    # FVA: integral of funding_spread × positive exposure × discount
    grid_np = np.array(grid)
    disc     = jnp.array([float(discount_curve.discount(float(t))) for t in grid_np])
    pos_exp  = jnp.maximum(jnp.array(exposure_profile), 0.0)
    fva_value = float(funding_spread * jnp.trapezoid(pos_exp * disc, jnp.array(grid_np)))

    total = cva_result.cva + dva_value + fva_value
    return XVAResult(cva=cva_result.cva, dva=dva_value, fva=fva_value, total=total)
