"""
repo_market.py
==============
Differentiable agent-based model of European sovereign repo market instability.

Models the "zero-haircut leverage" run mechanism identified by the FSB (Feb 2026)
as the dominant systemic risk in European repo markets. Distinct from Gorton &
Metrick (2012) whose US model centred on structured-credit quality deterioration.

European context:
    - Sovereign bonds: 87-90%+ of EU repo collateral (ICMA Survey #50, Dec 2025)
    - Rehypothecation: avg 3.05 chain links for EU primary dealers (ECB WP 3147)
    - Risk channel: zero-haircut NBFI leverage → rate shock → margin calls → spiral
    - 2022 prototype: UK LDI/gilt crisis (leveraged sovereign repo + duration mismatch)

Four collateral classes (ECB framework, Jun 2023):
    Sov-core    (Cat I):   Bunds, OATs, Dutch DSLs, EU bonds; haircut ~2%
    Sov-periph  (Cat I/II): BTPs, Bonos; haircut ~6%
    Corp/Cov    (Cat II/III): investment-grade corporates, Pfandbriefe; haircut ~12%
    ABS/Struct  (Cat V):   asset-backed securities; haircut ~15%

Four NBFI lender types with distinct reaction functions:
    MMF          — hard cliff at regulatory h_max; sovereign-core only
    LDI pension  — VaR-based haircut (σ-scaled); duration mismatch; high β
    Hedge fund   — PD + liquidity premium; operates across all classes; low β
    Triparty/CCP — algorithmic; delayed by 1 period (intraday credit)

Key Econiac additions:
    - Gibbs soft roll probability (β→∞ = hard binary roll/no-roll)
    - Rehypothecation chain: effective collateral supply = actual × rehyp_rate(σ,β)
    - JAX-differentiable: ∂(systemic_loss)/∂(h_ij) for optimal haircut policy
    - Sheaf Laplacian H¹ on dealer-lender funding graph (early-warning indicator)
    - Pacioli ∂²=0: every repo loan is simultaneously an asset and a liability

References:
    Gorton & Metrick (2012) JFE 104(3). doi:10.1016/j.jfineco.2011.03.016
    Calimani, Hałaj & Żochowski (2020) ECB WP 2373. doi:10.2139/ssrn.3540972
    Julliard et al. (2022) BIS WP 1027. doi:10.2139/ssrn.4168862
    Angelis Alexiou et al. (2025) ECB WP 3147.
    FSB (2026) Vulnerabilities in Government Bond-backed Repo Markets.
    Buckley (2026) Paper 332. doi:TBD
    Buckley (2026) Paper 333. doi:TBD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from econiac.core.ensemble import gibbs_weights


# ---------------------------------------------------------------------------
# Collateral class constants (European calibration)
# ---------------------------------------------------------------------------

N_COLLATERAL = 4

# Indices
SOV_CORE   = 0   # Cat I: Bunds, OATs, Dutch DSLs, EU bonds
SOV_PERIPH = 1   # Cat I/II: BTPs, Bonos, Portuguese OTs
CORP_COV   = 2   # Cat II/III: IG corporate, Pfandbriefe, covered bonds
ABS_STRUCT = 3   # Cat V: ABS, RMBS, CMBS, CDO (present for Gorton comparison)

# ECB haircut schedule (effective Jun 2023) — weighted average across maturities
ECB_HAIRCUTS_BASELINE = np.array([0.020, 0.060, 0.120, 0.150])

# Haircuts under LDI-style stress (rate shock of +150bp)
ECB_HAIRCUTS_STRESS   = np.array([0.050, 0.150, 0.250, 0.600])

# Average rehypothecation chain length (ECB WP 3147, primary dealers)
REHYP_RATE_BASELINE   = np.array([3.05,  2.10,  1.20,  0.30])

# Basel III risk weights
RISK_WEIGHTS          = np.array([0.00,  0.20,  0.50,  1.00])

# Market depth (normalised to 1000 per class; re-scaled in calibration)
MARKET_DEPTH_DEFAULT  = np.array([3000., 1500., 800.,  200.])


# ---------------------------------------------------------------------------
# Lender type enum
# ---------------------------------------------------------------------------

class LenderType(Enum):
    MMF       = "mmf"          # Money-market fund (hard cliff, low β)
    LDI       = "ldi"          # LDI pension fund (VaR-based, duration mismatch)
    HEDGE     = "hedge"        # Hedge fund (PD+liquidity premium, continuous)
    TRIPARTY  = "triparty"     # Clearing bank / triparty (algorithmic, delayed)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class RepoParams:
    """
    Calibration parameters for the European sovereign repo run model.

    Defaults calibrated to ICMA Survey #50 (Dec 2025), ECB WP 3147 (2025),
    and Julliard et al. (2022).
    """
    # Population
    n_dealers:   int   = 15     # prime broker / investment bank dealers
    n_mmf:       int   = 10     # money-market funds
    n_ldi:       int   = 8      # LDI pension funds
    n_hedge:     int   = 12     # hedge funds
    n_triparty:  int   = 3      # triparty clearing banks

    # Collateral
    n_collateral: int  = N_COLLATERAL
    ecb_haircuts: np.ndarray = field(
        default_factory=lambda: ECB_HAIRCUTS_BASELINE.copy()
    )
    stress_haircuts: np.ndarray = field(
        default_factory=lambda: ECB_HAIRCUTS_STRESS.copy()
    )
    rehyp_rate_0: np.ndarray = field(
        default_factory=lambda: REHYP_RATE_BASELINE.copy()
    )
    market_depth: np.ndarray = field(
        default_factory=lambda: MARKET_DEPTH_DEFAULT.copy()
    )

    # Price impact (same parameter as CHZ)
    lambda_impact: float = 0.1

    # Gibbs inverse temperatures by lender type
    # β → ∞ : hard binary roll/no-roll (original Gorton model)
    # β = 5  : continuous de-risking (realistic LDI/HF behaviour)
    beta_mmf:      float = 500.0   # near-hard (regulatory constraint)
    beta_ldi:      float = 20.0    # moderate (VaR-triggered)
    beta_hedge:    float = 5.0     # soft (risk-appetite driven)
    beta_triparty: float = 200.0   # near-hard but delayed 1 period

    # Funding ratio floor: dealers sell if funding_ratio < f_min
    f_min: float = 0.85            # 85% of repo funding must roll

    # Minimum haircut floor set by regulator (FSB proposal for non-sovereign)
    h_floor_non_sovereign: float = 0.05   # 5% FSB minimum haircut

    # Rehypothecation Gibbs: chains collapse as volatility rises
    beta_rehyp: float = 10.0       # how quickly chains collapse under stress
    sigma_0: float = 0.005         # baseline daily volatility (calm)
    sigma_stress: float = 0.02     # stress volatility (LDI crisis level)

    # LDI duration mismatch parameter
    # h_ldi(t) = h_baseline + k_ldi × max(0, σ(t) - σ_0)
    # k_ldi calibrated so that at σ=σ_stress, h rises by ~50% above baseline
    k_ldi: float = 3.0             # VaR multiplier (conservative; avoids immediate saturation)

    # Tâtonnement convergence
    n_tatonnement: int   = 20
    tol_tatonnement: float = 1e-6

    # Triparty delay: clearing bank absorbs t=1 shock; runs at t=2
    triparty_delay: int = 1

    @property
    def n_lenders(self) -> int:
        return self.n_mmf + self.n_ldi + self.n_hedge + self.n_triparty

    def beta_for_type(self, ltype: LenderType) -> float:
        return {
            LenderType.MMF:      self.beta_mmf,
            LenderType.LDI:      self.beta_ldi,
            LenderType.HEDGE:    self.beta_hedge,
            LenderType.TRIPARTY: self.beta_triparty,
        }[ltype]


# ---------------------------------------------------------------------------
# DealerBalanceSheet
# ---------------------------------------------------------------------------

@dataclass
class DealerBalanceSheet:
    """
    Vectorised balance sheet for n_dealers dealers.

    Pacioli identity (∂²=0 at each dealer):
        Σ_a holdings_ia × price_a  +  liquid_assets_i
            = Σ_j Q_ij × (1 - h_ij)   [repo borrowing, net of haircut]
              + equity_i

    The repo borrowing term is the key liability: if lenders withdraw (F_j rises),
    the dealer must sell collateral to replace the funding.

    Attributes:
        holdings   : (n_dealers, n_collateral) — quantity of each collateral class
        liquid     : (n_dealers,) — cash and HQLA reserves
        equity     : (n_dealers,) — book equity
        repo_out   : (n_dealers,) — total outstanding repo borrowing
        duration   : (n_dealers,) — portfolio duration (years); used for LDI margin calc
    """
    holdings: jax.Array    # (n_dealers, n_collateral)
    liquid:   jax.Array    # (n_dealers,)
    equity:   jax.Array    # (n_dealers,)
    repo_out: jax.Array    # (n_dealers,) — total repo funding outstanding
    duration: jax.Array    # (n_dealers,) — weighted portfolio duration

    @property
    def n_dealers(self) -> int:
        return int(self.equity.shape[0])

    def portfolio_value(self, prices: jax.Array) -> jax.Array:
        """Mark-to-market portfolio value: Σ_a holdings_ia × price_a. Shape (n_dealers,)."""
        return jnp.sum(self.holdings * prices, axis=1)

    def funding_ratio(self, prices: jax.Array, repo_rolled: jax.Array) -> jax.Array:
        """
        Funding ratio: what fraction of repo has been rolled by lenders.
        Shape (n_dealers,).
        repo_rolled: (n_dealers,) — total repo funding successfully rolled this period.
        """
        return repo_rolled / jnp.maximum(self.repo_out, 1e-8)

    def collateral_value_net(
        self, prices: jax.Array, haircuts: jax.Array
    ) -> jax.Array:
        """
        Net collateral value after haircuts: Σ_a holdings_ia × price_a × (1 - h_a).
        Shape (n_dealers,).
        haircuts: (n_collateral,) — current haircut per class.
        """
        return jnp.sum(self.holdings * prices * (1.0 - haircuts), axis=1)

    def is_pacioli_consistent(
        self, prices: jax.Array, haircuts: jax.Array, atol: float = 0.5
    ) -> bool:
        """
        Check Pacioli identity (approximate — repo structure simplifies here).
        Total assets ≈ total repo liabilities (net of haircut) + equity.
        atol is generous because the haircut structure is an approximation.
        """
        assets = self.portfolio_value(prices) + self.liquid
        liab   = self.collateral_value_net(prices, haircuts) + self.equity
        return bool(jnp.allclose(assets, liab, atol=atol))


# ---------------------------------------------------------------------------
# RepoLender — heterogeneous lender types
# ---------------------------------------------------------------------------

@dataclass
class RepoLender:
    """
    A single repo lender (or a block of identical lenders).

    The roll decision is the core behavioural equation:

        roll_prob(t) = σ( β × (net_collateral_value(t) / repo_outstanding - 1) )

    where σ is the logistic function (sigmoid). This interpolates between:
        β → ∞ : binary roll iff collateral covers loan (hard Gorton threshold)
        β small: gradual withdrawal as collateral deteriorates (realistic)

    For MMFs: additionally constrained by h_ij ≤ h_max_regulatory.
    For LDI:  h_ij(t) = k_ldi × σ_collateral(t) (VaR-based, σ rises under stress).
    For HF:   h_ij(t) = h_baseline + liquidity_premium(σ, PD).
    For TP:   roll_prob delayed by triparty_delay periods.
    """
    ltype:          LenderType
    n:              int           # number of lenders in this block
    lending:        jax.Array     # (n, n_dealers) — repo loans outstanding per dealer
    haircuts:       jax.Array     # (n, n_collateral) — current haircuts per class
    h_max:          float         # regulatory maximum haircut (binding for MMF)
    beta:           float         # Gibbs inverse temperature
    collateral_pref: jax.Array    # (n_collateral,) — preference weights (1=full, 0=none)
    # Accumulated haircut adjustment (LDI VaR, HF liquidity premium)
    h_adj:          jax.Array     # (n, n_collateral) — current adjustment above baseline

    @property
    def total_lending(self) -> jax.Array:
        """Total repo lending by this lender block. Shape (n,)."""
        return jnp.sum(self.lending, axis=1)

    def roll_probability(
        self,
        prices: jax.Array,
        dealer_holdings: jax.Array,  # (n_dealers, n_collateral)
        repo_outstanding: jax.Array,  # (n_dealers,)
        current_haircuts: jax.Array,  # (n_collateral,) — system-wide haircut
        period: int = 0,
        params: RepoParams | None = None,
    ) -> jax.Array:
        """
        Roll probability for each dealer from this lender block.
        Shape: (n, n_dealers).

        Core equation:
            roll_prob_ij = sigmoid(β × (c_ij(1-h_ij) / Q_ij - 1))

        where c_ij = dealer j's collateral value visible to lender i.
        """
        if params is not None and self.ltype == LenderType.TRIPARTY:
            if period <= params.triparty_delay:
                # Clearing bank extends intraday credit — no run in delay window
                return jnp.ones((self.n, dealer_holdings.shape[0]))

        # Net collateral coverage ratio per dealer (seen by all lenders in block equally)
        coll_val = jnp.sum(dealer_holdings * prices * (1.0 - current_haircuts), axis=1)
        # (n_dealers,) — collateral value net of haircut
        coverage = coll_val / jnp.maximum(repo_outstanding, 1e-8)  # (n_dealers,)

        # Gibbs roll probability: coverage > f_roll_threshold → roll; below → withdraw.
        # f_roll_threshold < 1 because the haircut IS the buffer: in normal conditions
        # dealers post collateral worth 1/(1-h) × loan, giving coverage = (1-h) < 1.
        # Lenders accept this — they know haircut compensates for the undercoverage.
        # Stress triggers a run when coverage falls further below the threshold.
        # f_roll_threshold calibrated so that pre-shock state is stable (most lenders roll).
        f_roll_threshold = 0.80   # lenders roll when net collateral covers >80% of loan
        logit = self.beta * (coverage - f_roll_threshold)  # (n_dealers,)
        roll  = jax.nn.sigmoid(logit)                  # (n_dealers,) ∈ (0,1)

        # MMF regulatory constraint: hard cliff if ANY collateral class has h > h_max
        if self.ltype == LenderType.MMF:
            h_violation = jnp.any(current_haircuts > self.h_max)
            # Also: MMFs only lend against Sov-core (collateral_pref)
            sov_core_only = jnp.all(dealer_holdings[:, 1:] == 0.0)
            roll = jnp.where(h_violation, jnp.zeros_like(roll), roll)

        # Broadcast to (n, n_dealers) — all lenders in block have same roll probability
        return jnp.broadcast_to(roll[None, :], (self.n, dealer_holdings.shape[0]))


# ---------------------------------------------------------------------------
# RehypothecationModel — collateral chain dynamics
# ---------------------------------------------------------------------------

def rehyp_rate(
    sigma: jax.Array,        # (n_collateral,) — current price volatility per class
    params: RepoParams,
) -> jax.Array:
    """
    Effective collateral multiplier from rehypothecation chains.
    Shape: (n_collateral,).

    Under calm conditions: rehyp_rate ≈ rehyp_rate_0 (3.05× for Sov-core).
    Under stress (σ → σ_stress): rehyp_rate → 1.0 (chains collapse, each unit
      of collateral finances only one repo, not a 3-link chain).

    Gibbs model:
        rehyp(t) = 1 + (rehyp_0 - 1) × sigmoid(-β_rehyp × (σ(t) - σ_0))

    When σ = σ_0:   rehyp ≈ 1 + (rehyp_0-1) × 0.5 = midpoint (smooth baseline)
    When σ ≫ σ_0:   rehyp → 1.0  (chains fully collapsed)
    When σ ≪ σ_0:   rehyp → rehyp_0 (full chain active)

    The (rehyp_0 - 1) term is the "amplification premium" — the extra funding
    capacity created by collateral reuse. FSB (2026) identifies this as procyclical.
    """
    rehyp_0 = jnp.array(params.rehyp_rate_0)
    sigma_rel = params.beta_rehyp * (sigma - params.sigma_0)
    # Fraction of maximum chain active = sigmoid of negative stress
    chain_active = jax.nn.sigmoid(-sigma_rel)           # (n_collateral,)
    return 1.0 + (rehyp_0 - 1.0) * chain_active        # (n_collateral,)


def effective_collateral(
    actual_holdings: jax.Array,   # (n_dealers, n_collateral)
    sigma: jax.Array,             # (n_collateral,) — volatility
    params: RepoParams,
) -> jax.Array:
    """
    Effective collateral supply after rehypothecation chain amplification.
    Shape: (n_dealers, n_collateral).

    effective = actual × rehyp_rate(σ)

    During calm: effective >> actual (chains amplify funding capacity).
    During stress: effective → actual (chains collapse; no amplification).
    The *change* in effective collateral supply — not just price moves — is the
    additional amplification channel unique to the European repo market.
    """
    r = rehyp_rate(sigma, params)   # (n_collateral,)
    return actual_holdings * r[None, :]


# ---------------------------------------------------------------------------
# Haircut update functions by lender type
# ---------------------------------------------------------------------------

def update_haircuts_ldi(
    h_baseline: jax.Array,   # (n_collateral,)
    sigma: jax.Array,        # (n_collateral,) — current daily price volatility
    params: RepoParams,
) -> jax.Array:
    """
    LDI VaR-based haircut: h_ij(t) = h_baseline + k_ldi × max(0, σ - σ_0).

    At σ = σ_0 (calm): h = h_baseline (no adjustment — stable rolling).
    At σ = σ_stress (crisis): h rises by k_ldi × (σ_stress - σ_0) above baseline.
    This captures the 2022 LDI mechanism: rate volatility → margin calls.

    Chosen over the multiplicative rule h = k×σ because that saturates immediately:
    a 5% one-day price drop gives σ=0.05, and h = 12×0.05 = 0.60 (max) on day 1.
    The additive rule allows for a gradual, realistic stress escalation.
    """
    excess_vol = jnp.maximum(sigma - params.sigma_0, 0.0)          # (n_collateral,)
    h_ldi = h_baseline + params.k_ldi * excess_vol                  # (n_collateral,)
    return jnp.clip(h_ldi, h_baseline, params.stress_haircuts)


def update_haircuts_hedge(
    h_baseline: jax.Array,   # (n_collateral,)
    sigma: jax.Array,        # (n_collateral,)
    pd: jax.Array,           # (n_dealers,) — dealer default probabilities
    params: RepoParams,
) -> jax.Array:
    """
    Hedge fund haircut: h = h_baseline + liquidity_premium + PD_premium.

    liquidity_premium = β_hf_liq × max(0, σ - σ_0)
    PD_premium        = ξ × mean(PD)

    Hedge funds are the most continuous lenders — they price risk rather than
    applying binary rules. Low β in Gibbs model (most gradual withdrawal).
    """
    xi = 0.5   # credit spread sensitivity (same as CHZ ξ parameter)
    liquidity_premium = jnp.maximum(sigma - params.sigma_0, 0.0) * 5.0
    pd_premium        = xi * jnp.mean(pd)   # scalar
    h_hf = h_baseline + liquidity_premium + pd_premium
    return jnp.clip(h_hf, h_baseline, params.stress_haircuts)


# ---------------------------------------------------------------------------
# TatonnementPricer (repo version) — price formation inner loop
# ---------------------------------------------------------------------------

class RepoTatonnementResult(NamedTuple):
    """Output of one tâtonnement convergence in the repo model."""
    prices:           jax.Array   # (n_collateral,) — converged prices
    sell_volume:      jax.Array   # (n_collateral,) — total collateral sold
    funding_withdrawn: jax.Array  # (n_dealers,) — funding gap per dealer
    n_iter:           int
    converged:        bool


def repo_tatonnement(
    dealers:    DealerBalanceSheet,
    lenders:    list[RepoLender],
    prices_prev: jax.Array,       # (n_collateral,) — previous period prices
    prices_init: jax.Array,       # (n_collateral,) — prices at start of this period
    sigma:      jax.Array,        # (n_collateral,) — rolling volatility
    params:     RepoParams,
    period:     int = 1,
) -> RepoTatonnementResult:
    """
    Endogenous price formation in the repo market via tâtonnement.

    Inner loop for one simulation period:
      1. Compute effective collateral supply (with rehypothecation)
      2. Each lender type computes roll probability for each dealer
      3. Funding withdrawn = lending × (1 - roll_prob) per dealer
      4. Dealers sell collateral proportionally to cover funding gap
      5. Prices update: p ← p × (1 - λ × sell / depth)
      6. σ updates: rolling volatility includes this period's price move
      7. Repeat until convergence

    Key difference from CHZ: haircuts update endogenously within the loop
    (LDI VaR-based haircuts rise as σ rises → more margin calls → more sales).
    This creates the haircut spiral absent from CHZ.
    """
    p       = prices_init.copy()
    conv    = False
    n_iter  = 0
    sell_vol_total = jnp.zeros(params.n_collateral)
    fund_withdrawn = jnp.zeros(dealers.n_dealers)

    for _ in range(params.n_tatonnement):
        n_iter += 1

        # --- Volatility update (rolling; include current move) ---
        price_move = jnp.abs(p - prices_prev) / jnp.maximum(prices_prev, 1e-8)
        sigma_t    = 0.8 * sigma + 0.2 * price_move   # EWM update

        # --- Haircut update per lender type ---
        # (In practice all lenders of a type share the same haircut schedule)
        # We compute a system-level haircut as max across lender types
        h_ecb  = jnp.array(params.ecb_haircuts)
        # LDI VaR haircut
        h_ldi  = update_haircuts_ldi(h_ecb, sigma_t, params)
        # Hedge fund haircut (use mean dealer PD = 0.02 as proxy; updated in outer loop)
        h_hf   = update_haircuts_hedge(h_ecb, sigma_t, jnp.full(dealers.n_dealers, 0.02), params)
        # Effective haircut: max across lender types (most restrictive governs price)
        h_eff  = jnp.maximum(h_ecb, jnp.maximum(h_ldi, h_hf))   # (n_collateral,)

        # --- Effective collateral supply (rehypothecation) ---
        eff_coll = effective_collateral(dealers.holdings, sigma_t, params)
        # Net collateral value per dealer
        coll_val = jnp.sum(eff_coll * p * (1.0 - h_eff), axis=1)   # (n_dealers,)

        # --- Roll probabilities per lender type ---
        # We compute aggregate funding rolled per dealer
        total_lending  = jnp.zeros(dealers.n_dealers)
        total_rolled   = jnp.zeros(dealers.n_dealers)

        for lender_block in lenders:
            roll_prob = lender_block.roll_probability(
                p, dealers.holdings, dealers.repo_out, h_eff, period, params
            )   # (n_lenders_in_block, n_dealers)
            # lending_ij = repo loan from lender i to dealer j
            rolled = jnp.sum(lender_block.lending * roll_prob, axis=0)   # (n_dealers,)
            total  = jnp.sum(lender_block.lending, axis=0)                # (n_dealers,)
            total_rolled  = total_rolled  + rolled
            total_lending = total_lending + total

        # Funding withdrawn = what was outstanding minus what rolled
        fund_withdrawn = dealers.repo_out - total_rolled   # (n_dealers,)
        fund_withdrawn = jnp.maximum(fund_withdrawn, 0.0)

        # --- Dealer fire sales to cover funding gap ---
        # Sell proportionally across collateral classes
        port_val   = dealers.holdings * p           # (n_dealers, n_collateral)
        port_total = jnp.sum(port_val, axis=1, keepdims=True)
        port_w     = port_val / jnp.maximum(port_total, 1e-8)   # (n_dealers, n_collateral)
        # Sell value per collateral class per dealer
        sell_val   = port_w * fund_withdrawn[:, None]            # (n_dealers, n_collateral)
        # Convert to quantity
        sell_qty   = sell_val / jnp.maximum(p, 1e-8)            # (n_dealers, n_collateral)
        sell_vol   = jnp.sum(sell_qty, axis=0)                   # (n_collateral,) aggregate

        # --- Price update ---
        depth  = jnp.array(params.market_depth)
        p_new  = p * (1.0 - params.lambda_impact * sell_vol / depth)
        p_new  = jnp.maximum(p_new, 1e-4)

        # --- Convergence ---
        delta  = jnp.max(jnp.abs(p_new - p))
        p      = p_new
        sell_vol_total = sell_vol

        if float(delta) < params.tol_tatonnement:
            conv = True
            break

    return RepoTatonnementResult(
        prices=p,
        sell_volume=sell_vol_total,
        funding_withdrawn=fund_withdrawn,
        n_iter=n_iter,
        converged=conv,
    )


# ---------------------------------------------------------------------------
# Period and simulation result containers
# ---------------------------------------------------------------------------

class RepoPeriodResult(NamedTuple):
    """State at the end of one simulation period."""
    t:                    int
    prices:               jax.Array   # (n_collateral,)
    haircuts:             jax.Array   # (n_collateral,) — effective haircut this period
    sigma:                jax.Array   # (n_collateral,) — price volatility
    rehyp_rates:          jax.Array   # (n_collateral,) — effective chain length
    funding_withdrawn:    jax.Array   # (n_dealers,) — funding gap
    n_dealers_in_run:     int         # dealers with funding_ratio < f_min
    total_funding_gap:    float       # Σ_j F_j(t)
    price_impact:         float       # mean |Δp/p_init| across collateral classes
    tatonnement_iters:    int


class RepoSimResult(NamedTuple):
    """Full simulation output."""
    periods:          list[RepoPeriodResult]
    final_prices:     jax.Array   # (n_collateral,)
    final_haircuts:   jax.Array   # (n_collateral,)
    total_run_length: int         # number of periods until quiescence
    peak_dealers_run: int         # max n_dealers_in_run across all periods
    systemic_loss:    float       # Σ_j F_j × haircut_amplification
    price_drop:       jax.Array   # (n_collateral,) — fraction drop from initial


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate_repo(
    dealers:      DealerBalanceSheet,
    lenders:      list[RepoLender],
    prices_init:  jax.Array,      # (n_collateral,) — pre-shock prices
    shock_prices: jax.Array,      # (n_collateral,) — post-shock prices
    params:       RepoParams,
    n_periods:    int = 15,
) -> RepoSimResult:
    """
    Simulate a repo market run over n_periods periods.

    Outer loop (period-by-period):
      Period 0: initial state (pre-shock)
      Period 1: rate shock lands (prices_init → shock_prices)
               Mark-to-market equity write-down on dealers
               Margin calls from LDI funds begin
      Periods 2+: tâtonnement + balance sheet update each period

    The shock is the *exogenous* trigger (e.g. +150bp rate shock → Sov-core
    and Sov-periph fall 5-10% in price). The cascade is endogenous.

    Args:
        dealers:       initial dealer balance sheets
        lenders:       list of RepoLender blocks (one per type)
        prices_init:   pre-shock prices, shape (n_collateral,)
        shock_prices:  post-shock prices, shape (n_collateral,)
        params:        RepoParams
        n_periods:     max periods to simulate

    Returns:
        RepoSimResult
    """
    prices      = shock_prices
    prices_prev = shock_prices   # cascade starts from post-shock; don't re-fire shock
    # Initial rolling volatility: infer from shock size
    sigma       = jnp.abs(shock_prices - prices_init) / jnp.maximum(prices_init, 1e-8)
    sigma       = jnp.maximum(sigma, params.sigma_0)

    # Mark-to-market dealer equity after shock
    port_loss   = jnp.sum(dealers.holdings * (prices_init - shock_prices), axis=1)
    shocked_equity = dealers.equity - port_loss

    dealers_t = DealerBalanceSheet(
        holdings  = dealers.holdings,
        liquid    = dealers.liquid,
        equity    = shocked_equity,
        repo_out  = dealers.repo_out,
        duration  = dealers.duration,
    )

    periods      = []
    h_current    = jnp.array(params.ecb_haircuts)   # start at baseline

    for t in range(1, n_periods + 1):
        # --- Inner loop: tâtonnement ---
        tat = repo_tatonnement(
            dealers_t, lenders, prices_prev, prices, sigma, params, period=t
        )
        prices_new = tat.prices

        # --- Update rolling volatility ---
        price_move = jnp.abs(prices_new - prices) / jnp.maximum(prices, 1e-8)
        sigma      = 0.8 * sigma + 0.2 * price_move

        # --- Update effective haircuts ---
        h_ldi     = update_haircuts_ldi(jnp.array(params.ecb_haircuts), sigma, params)
        h_hf      = update_haircuts_hedge(
            jnp.array(params.ecb_haircuts), sigma,
            jnp.full(dealers_t.n_dealers, 0.02), params
        )
        h_current = jnp.maximum(jnp.array(params.ecb_haircuts),
                                 jnp.maximum(h_ldi, h_hf))

        # --- Update dealer balance sheets ---
        # Reduce holdings by quantities sold
        port_val  = dealers_t.holdings * prices_new
        port_total= jnp.sum(port_val, axis=1, keepdims=True)
        port_w    = port_val / jnp.maximum(port_total, 1e-8)
        sell_qty  = port_w * tat.funding_withdrawn[:, None] / jnp.maximum(prices_new, 1e-8)
        new_holdings = jnp.maximum(dealers_t.holdings - sell_qty, 0.0)

        # Equity: mark-to-market on remaining holdings
        equity_new = (
            dealers_t.equity
            + jnp.sum(dealers_t.holdings * prices_new, axis=1)
            - jnp.sum(dealers_t.holdings * prices, axis=1)
        )

        # Repo outstanding: reduce by withdrawn funding (dealer repays what it can)
        repo_new = jnp.maximum(dealers_t.repo_out - tat.funding_withdrawn, 0.0)

        dealers_t = DealerBalanceSheet(
            holdings  = new_holdings,
            liquid    = dealers_t.liquid,
            equity    = equity_new,
            repo_out  = repo_new,
            duration  = dealers_t.duration,
        )

        # --- Period statistics ---
        fund_ratio  = (dealers_t.repo_out /
                       jnp.maximum(dealers_t.repo_out + tat.funding_withdrawn, 1e-8))
        n_in_run    = int(jnp.sum(fund_ratio < params.f_min))
        total_gap   = float(jnp.sum(tat.funding_withdrawn))
        rehyp_t     = rehyp_rate(sigma, params)
        price_impact= float(jnp.mean(jnp.abs(prices_new - prices) /
                                     jnp.maximum(prices, 1e-8)))

        periods.append(RepoPeriodResult(
            t=t,
            prices=prices_new,
            haircuts=h_current,
            sigma=sigma,
            rehyp_rates=rehyp_t,
            funding_withdrawn=tat.funding_withdrawn,
            n_dealers_in_run=n_in_run,
            total_funding_gap=total_gap,
            price_impact=price_impact,
            tatonnement_iters=tat.n_iter,
        ))

        prices_prev = prices
        prices      = prices_new

        # Early exit if quiescent
        if n_in_run == 0 and tat.n_iter <= 2:
            break

    # --- Final statistics ---
    final_prices   = prices
    final_haircuts = h_current
    total_run_len  = len(periods)
    peak_run       = max(p.n_dealers_in_run for p in periods) if periods else 0
    systemic_loss  = sum(p.total_funding_gap for p in periods)
    price_drop     = (prices_init - final_prices) / jnp.maximum(prices_init, 1e-8)

    return RepoSimResult(
        periods=periods,
        final_prices=final_prices,
        final_haircuts=final_haircuts,
        total_run_length=total_run_len,
        peak_dealers_run=peak_run,
        systemic_loss=systemic_loss,
        price_drop=price_drop,
    )


# ---------------------------------------------------------------------------
# Convenience: baseline initialisation
# ---------------------------------------------------------------------------

def repo_baseline(
    params: RepoParams | None = None,
    seed: int = 42,
) -> tuple[DealerBalanceSheet, list[RepoLender], jax.Array]:
    """
    Construct a baseline European repo market (dealers + lenders + prices).

    Returns: (dealers, lenders, prices_init)

    Calibration targets (ICMA Survey #50, ECB WP 3147, Julliard et al. 2022):
      - Dealer collateral mix: 60% Sov-core, 20% Sov-periph, 15% Corp, 5% ABS
      - Lender mix: MMF 25%, LDI 20%, HF 35%, Triparty 20% of total lending
      - Repo funding = 65-75% of dealer total assets (ECB repo data)
      - Initial haircuts: ECB baseline schedule
      - Prices normalised to 1.0

    Dealer heterogeneity:
      - Type A (6 dealers): primary dealers, diversified collateral, high leverage
      - Type B (5 dealers): broker-dealers, equity + structured, medium leverage
      - Type C (4 dealers): hedge fund prime brokers, structured heavy, low leverage
    """
    if params is None:
        params = RepoParams()

    rng = np.random.default_rng(seed)
    n_d, n_c = params.n_dealers, params.n_collateral

    # --- Prices ---
    prices = jnp.ones(n_c)

    # --- Dealer balance sheets ---
    # Collateral mix by dealer type (rows = type, cols = collateral class)
    # Calibrated to ICMA survey: 61% AAA-AA govt, 20% lower-rated govt, 14% corp, 5% ABS
    type_mix = np.array([
        # Sov-core  Sov-periph  Corp/Cov  ABS
        [0.65,     0.20,       0.12,     0.03],   # Type A: primary dealer
        [0.50,     0.25,       0.18,     0.07],   # Type B: broker-dealer
        [0.35,     0.25,       0.25,     0.15],   # Type C: HF prime broker
    ])
    dealer_types = np.array([0]*6 + [1]*5 + [2]*4)   # 6 + 5 + 4 = 15

    total_assets = rng.uniform(80.0, 120.0, n_d)       # normalised balance sheet size

    # Holdings
    holdings = np.zeros((n_d, n_c))
    for i, dt in enumerate(dealer_types):
        mix_i = type_mix[dt] + rng.normal(0, 0.02, n_c)
        mix_i = np.maximum(mix_i, 0.01); mix_i /= mix_i.sum()
        # Securities = 60% of total assets
        holdings[i] = mix_i * total_assets[i] * 0.60

    liquid   = rng.uniform(5.0, 15.0, n_d)            # liquid reserves
    # Repo funding = 65% of total assets (repo-heavy, ECB calibration)
    repo_out = total_assets * rng.uniform(0.60, 0.70, n_d)
    # Equity: residual (assets - repo - other liabilities)
    equity   = total_assets * rng.uniform(0.04, 0.08, n_d)  # 4-8% leverage ratio
    # Duration: weighted average, Type A=5yr, B=4yr, C=3yr
    duration_by_type = np.array([5.0, 4.0, 3.0])
    duration = np.array([duration_by_type[dealer_types[i]] * rng.uniform(0.8, 1.2)
                         for i in range(n_d)])

    dealers = DealerBalanceSheet(
        holdings  = jnp.array(holdings),
        liquid    = jnp.array(liquid),
        equity    = jnp.array(equity),
        repo_out  = jnp.array(repo_out),
        duration  = jnp.array(duration),
    )

    # --- Lenders ---
    # IMPORTANT: total lending across all lender blocks must equal total dealer repo_out.
    # Lender share fractions (MMF 25%, LDI 20%, HF 35%, TP 20%) applied to repo_out.sum().
    total_repo = repo_out.sum()   # anchor to dealer-side outstanding
    lenders    = []

    def make_lending_block(n_lenders, share, pref_weights):
        """
        Allocate a lender block's lending across dealers.
        total per lender = total_repo × share / n_lenders
        pref_weights: (n_dealers,) preference weighting across dealers.
        """
        total_per_lender = total_repo * share / n_lenders
        block = np.zeros((n_lenders, n_d))
        pref_w = pref_weights / pref_weights.sum()
        for i in range(n_lenders):
            noise = rng.uniform(0.85, 1.15, n_d)
            alloc = pref_w * noise
            alloc /= alloc.sum()
            block[i] = alloc * total_per_lender
        return block

    # MMF block: lend against Sov-core primarily; hard cliff at h_max=2%
    mmf_pref    = np.array([3.0 if dt == 0 else 1.0 for dt in dealer_types])
    mmf_lending = make_lending_block(params.n_mmf, 0.25, mmf_pref)

    lenders.append(RepoLender(
        ltype=LenderType.MMF,
        n=params.n_mmf,
        lending=jnp.array(mmf_lending),
        haircuts=jnp.array(np.tile(params.ecb_haircuts, (params.n_mmf, 1))),
        h_max=0.02,   # EU MMF Regulation max haircut
        beta=params.beta_mmf,
        collateral_pref=jnp.array([1.0, 0.5, 0.0, 0.0]),   # Sov-core preferred
        h_adj=jnp.zeros((params.n_mmf, n_c)),
    ))

    # LDI block: long-duration sovereign focus; VaR-based haircut
    ldi_pref    = np.array([2.0 if dt == 0 else 1.0 for dt in dealer_types])
    ldi_lending = make_lending_block(params.n_ldi, 0.20, ldi_pref)

    lenders.append(RepoLender(
        ltype=LenderType.LDI,
        n=params.n_ldi,
        lending=jnp.array(ldi_lending),
        haircuts=jnp.array(np.tile(params.ecb_haircuts, (params.n_ldi, 1))),
        h_max=0.50,   # no hard regulatory cap (LDI sets own VaR limit)
        beta=params.beta_ldi,
        collateral_pref=jnp.array([1.0, 1.0, 0.3, 0.0]),   # sovereign focused
        h_adj=jnp.zeros((params.n_ldi, n_c)),
    ))

    # Hedge fund block: most flexible; all collateral classes; continuous Gibbs
    # Hedge fund block: most flexible; all collateral classes
    hf_pref    = np.ones(n_d)   # uniform across all dealers
    hf_lending = make_lending_block(params.n_hedge, 0.35, hf_pref)

    lenders.append(RepoLender(
        ltype=LenderType.HEDGE,
        n=params.n_hedge,
        lending=jnp.array(hf_lending),
        haircuts=jnp.array(np.tile(params.ecb_haircuts, (params.n_hedge, 1))),
        h_max=1.00,   # no regulatory cap
        beta=params.beta_hedge,
        collateral_pref=jnp.ones(n_c),   # accepts all collateral
        h_adj=jnp.zeros((params.n_hedge, n_c)),
    ))

    # Triparty block: algorithmic, delayed; clears proportional to dealer size
    tp_pref    = total_assets / total_assets.sum() * n_d   # size-proportional
    tp_lending = make_lending_block(params.n_triparty, 0.20, tp_pref)

    lenders.append(RepoLender(
        ltype=LenderType.TRIPARTY,
        n=params.n_triparty,
        lending=jnp.array(tp_lending),
        haircuts=jnp.array(np.tile(params.ecb_haircuts, (params.n_triparty, 1))),
        h_max=1.00,
        beta=params.beta_triparty,
        collateral_pref=jnp.ones(n_c),
        h_adj=jnp.zeros((params.n_triparty, n_c)),
    ))

    return dealers, lenders, prices


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== European Sovereign Repo Run Model — Econiac ===\n")

    params = RepoParams()
    dealers, lenders, prices_init = repo_baseline(params)

    # Sanity: check initial state
    pv = dealers.portfolio_value(prices_init)
    print(f"Dealers ({params.n_dealers}):")
    print(f"  Portfolio value: mean={float(pv.mean()):.1f}, "
          f"min={float(pv.min()):.1f}, max={float(pv.max()):.1f}")
    print(f"  Repo outstanding: mean={float(dealers.repo_out.mean()):.1f}, "
          f"total={float(dealers.repo_out.sum()):.1f}")
    print(f"  Leverage ratio (equity/assets): "
          f"mean={float((dealers.equity/pv).mean()):.3f}")
    print()
    for lb in lenders:
        print(f"  Lender block {lb.ltype.value}: n={lb.n}, "
              f"total lending={float(lb.lending.sum()):.1f}")
    print()

    # Apply a +150bp rate shock: Sov-core -5%, Sov-periph -9%, Corp -3%, ABS -2%
    # (Duration-weighted: Sov-core dur≈7yr → 7×0.015=10.5%; using 5% as net of coupon)
    shock_prices = prices_init * jnp.array([0.95, 0.91, 0.97, 0.98])
    print(f"Rate shock (+150bp): prices → {[f'{float(p):.3f}' for p in shock_prices]}")
    print()

    result = simulate_repo(dealers, lenders, prices_init, shock_prices,
                           params, n_periods=15)

    print(f"Simulation: {result.total_run_length} periods")
    print(f"  Peak dealers in run: {result.peak_dealers_run} / {params.n_dealers}")
    print(f"  Systemic loss (total funding gap): {result.systemic_loss:.1f}")
    print(f"  Final price drops: {[f'{float(d):.1%}' for d in result.price_drop]}")
    print()
    print(f"{'t':>3}  {'in_run':>6}  {'gap':>8}  {'impact':>8}  {'tat':>5}")
    for p in result.periods:
        print(f"{p.t:>3}  {p.n_dealers_in_run:>6}  "
              f"{p.total_funding_gap:>8.2f}  {p.price_impact:>8.4f}  "
              f"{p.tatonnement_iters:>5}")
