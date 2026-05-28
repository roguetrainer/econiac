"""
primitives.py
=============
Primitive contagion operators — the four Hurd channels plus Rehypothecation.

Each primitive is a function that returns an ``Operator`` instance.
All accept ``beta`` as an explicit keyword argument so that ``gibbs_lift()``
can re-parameterise them for the phase diagram sweep.

Channel map (Hurd 2017 → Econiac):
    S_direct  — Eisenberg-Noe solvency cascade (interbank default propagation)
                Adjoint of L_direct (AL symmetry, Hurd Proposition 1)
    L_direct  — Gai-Kapadia liquidity cascade (funding withdrawal propagation)
                Adjoint of S_direct
    L_A       — Fire-sale / indirect contagion (price-mediated)
                Extracts core logic from econiac.finance.fire_sales
    S_D       — Bank panic / repo run (deposit or repo withdrawal → solvency loss)
                Extracts core logic from econiac.finance.repo_market
    Rehyp     — Rehypothecation chain collapse (novel Econiac channel)
                Unique to European sovereign repo; no Hurd equivalent

Design principles:
    - Each primitive accepts the application-specific balance-sheet object and
      a ``beta`` keyword. It returns an Operator whose ``forward`` is a closure
      over those objects and that beta.
    - The ``adjoint`` of S_direct is L_direct and vice versa (structural AL symmetry).
    - L_A, S_D, and Rehyp have adjoints defined via the Pacioli inner product but
      they are not economically distinct channels — they are set to ``_no_adjoint``
      with a docstring explaining the mathematical dual.
    - Tâtonnement price formation is shared across L_A and S_D via the
      ``_tatonnement_price_step`` helper.
    - All primitives clip their output to [0,1]^n × R_{>0}^m after each step.

AL symmetry (Hurd Proposition 1):
    In Hurd's notation: Δ_i^(n) (solvency loss) maps to Σ_i^(n) (liquidity loss)
    under A↔D, Z↔X, C↔E, Δ↔Σ. In Econiac: this is the adjoint functor.
    SolvencyState and LiquidityState are dual under the Pacioli inner product
    ⟨f(x), y⟩_P = ⟨x, f†(y)⟩_P, which is enforced by Pacioli ∂²=0.

References:
    Hurd, T.R. (2017). Bank Panics and Fire Sales. arXiv:1711.05289v1.
    Eisenberg, L. & Noe, T.H. (2001). Management Science 47(2), 236-249.
    Gai, P. & Kapadia, S. (2010). Proc. R. Soc. A 466, 2401-2423.
    Calimani, Hałaj & Żochowski (2020). ECB WP 2373.
    Buckley (2026). Paper 332: CHZ fire sales. doi:TBD
    Buckley (2026). Paper 333: Sovereign repo run. doi:TBD
    Buckley (2026). Paper 334: Contagion operator algebra. doi:TBD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from econiac.finance.contagion.operators import (
    Operator,
    SystemState,
    SolvencyState,
    LiquidityState,
    PriceState,
    _no_adjoint,
)
from econiac.finance.contagion.gibbs import (
    gibbs_threshold,
    gibbs_rehyp,
    gibbs_weight_asset,
)


# ---------------------------------------------------------------------------
# Shared tâtonnement price step
# ---------------------------------------------------------------------------

def _tatonnement_price_step(
    sell_qty:      jax.Array,   # (n_assets,) — total quantity sold this step
    prices:        jax.Array,   # (n_assets,) — current prices
    market_depth:  jax.Array,   # (n_assets,) — market depth per class
    lambda_impact: float,        # price-impact coefficient
) -> jax.Array:
    """
    One tâtonnement price update: p ← p × (1 - λ × sell / depth).

    Shared between L_A (fire sales) and S_D (repo run forced selling).
    Prices are floored at 1e-4 to prevent numerical collapse.

    This is the Walrasian tâtonnement price mechanism from CHZ §2.3 and
    repo_market.py, extracted into the shared library to eliminate duplication.

    Args:
        sell_qty:      total quantity sold across all sellers, per asset
        prices:        current asset prices
        market_depth:  total market depth per asset class
        lambda_impact: price-impact coefficient (CHZ: 0.1; repo: 0.1)

    Returns:
        (n_assets,) — updated prices
    """
    impact  = lambda_impact * sell_qty / jnp.maximum(market_depth, 1e-8)
    p_new   = prices * (1.0 - impact)
    return jnp.maximum(p_new, 1e-4)


# ---------------------------------------------------------------------------
# S_direct — Eisenberg-Noe solvency cascade
# ---------------------------------------------------------------------------

@dataclass
class SolvencyParams:
    """
    Parameters for the direct solvency cascade (S channel).

    Implements the Eisenberg-Noe (EN) proportional clearing model with
    Gibbs softening of the solvency threshold.

    Fields:
        exposure_matrix: (n_agents, n_agents) — A_ij = amount i is owed by j.
                          Normalised: A_ij = bilateral_exposure_ij / total_liabilities_i.
        external_assets: (n_agents,) — claims on entities outside the network.
        total_liabilities: (n_agents,) — total obligations of each agent.
        solvency_floor:  threshold p* below which an agent is counted as defaulting
                         (feeds into the indicator function / Gibbs approximation).
    """
    exposure_matrix:    jax.Array   # (n, n) — normalised relative liabilities
    external_assets:    jax.Array   # (n,) — external claims
    total_liabilities:  jax.Array   # (n,) — total obligations
    solvency_floor:     float = 0.95


def S_direct(params: SolvencyParams, beta: float = 50.0) -> Operator:
    """
    Eisenberg-Noe solvency cascade operator (S channel).

    One step of the EN clearing iteration:
        p_i^(n+1) = min(1,  (e_i + Σ_j A_ji × p_j^(n) × L_j) / L_i)

    where:
        e_i  = external assets (cash + non-network claims)
        A_ji = fraction of j's obligations owed to i (normalised)
        L_j  = total liabilities of j
        p_j  = current clearing vector (solvency fraction)

    Gibbs softening replaces the hard min(1, ...) with a sigmoid:
        p_i = σ(β × (recovery_i / L_i − solvency_floor))
    At β → ∞ this recovers the EN hard rule.

    The operator reads: SolvencyState + LiquidityState + PriceState (for MtM equity).
    It writes: SolvencyState only.

    Adjoint: L_direct (AL symmetry, Hurd Proposition 1).

    Args:
        params: SolvencyParams — calibrated exposure network
        beta:   inverse temperature (β → ∞ = hard EN threshold)

    Returns:
        Operator with channel="S"
    """
    A   = params.exposure_matrix       # (n, n)
    e   = params.external_assets       # (n,)
    L   = params.total_liabilities     # (n,)

    def forward(x: SystemState, beta: float = beta) -> SystemState:
        p = x.solvency.fractions       # (n,)

        # Recovery value: external assets + interbank receipts
        # Σ_j A_ji × p_j × L_j = payments received from counterparties
        interbank_receipts = A.T @ (p * L)              # (n,)
        recovery           = e + interbank_receipts      # (n,)

        # Gibbs clearing vector: logit = recovery/L - solvency_floor
        logit     = recovery / jnp.maximum(L, 1e-8) - params.solvency_floor
        p_new     = gibbs_threshold(logit, beta)         # (n,) in (0, 1)
        p_new     = jnp.clip(p_new, 0.0, 1.0)

        return SystemState(
            solvency  = SolvencyState(fractions=p_new),
            liquidity = x.liquidity,
            price     = x.price,
        )

    # Adjoint: L_direct is constructed lazily (defined below)
    # We use a placeholder here; wire_al_symmetry() completes the pair.
    return Operator(
        forward  = forward,
        adjoint  = _no_adjoint,   # replaced by wire_al_symmetry(S, L)
        name     = "S",
        is_gibbs = True,
        channel  = "S",
    )


# ---------------------------------------------------------------------------
# L_direct — Gai-Kapadia liquidity cascade (adjoint of S)
# ---------------------------------------------------------------------------

@dataclass
class LiquidityParams:
    """
    Parameters for the direct liquidity cascade (L channel).

    Implements the Gai-Kapadia (GL) model: funding withdrawal propagates
    through the network of bilateral repo / interbank lending relationships.

    Fields:
        funding_matrix:  (n_agents, n_agents) — F_ij = fraction of i's funding
                          sourced from j. Normalised: Σ_j F_ij = 1.
        liquid_buffer:   (n_agents,) — liquid asset buffer (HQLA) as fraction
                          of total funding. Agents survive a run up to this buffer.
        liquidity_floor: threshold p̃* below which an agent is "in run".
    """
    funding_matrix:  jax.Array   # (n, n) — normalised funding sources
    liquid_buffer:   jax.Array   # (n,) — liquid asset buffer as funding fraction
    liquidity_floor: float = 0.80


def L_direct(params: LiquidityParams, beta: float = 20.0) -> Operator:
    """
    Gai-Kapadia liquidity cascade operator (L channel).

    One step of the GL clearing iteration:
        p̃_i^(n+1) = σ(β × (buffer_i + Σ_j F_ij × p̃_j^(n) − liquidity_floor))

    where:
        buffer_i = liquid asset buffer (fraction of funding)
        F_ij     = fraction of i's funding from j
        p̃_j     = current liquidity fraction of lender j

    Gibbs softening replaces the hard floor with a sigmoid.
    At β → ∞: binary liquid/illiquid.
    At β = 20: gradual de-leveraging (LDI calibration).

    The operator reads: LiquidityState + PriceState.
    It writes: LiquidityState only.

    Adjoint: S_direct (AL symmetry — solvency ↔ liquidity under A↔D, Z↔X).

    Args:
        params: LiquidityParams
        beta:   inverse temperature

    Returns:
        Operator with channel="L"
    """
    F = params.funding_matrix    # (n, n)
    b = params.liquid_buffer     # (n,)

    def forward(x: SystemState, beta: float = beta) -> SystemState:
        p_tilde = x.liquidity.fractions    # (n,)

        # Funding available = buffer + weighted average of lenders' liquidity
        funding_from_network = F @ p_tilde             # (n,)
        total_funding        = b + funding_from_network

        # Gibbs liquidity fraction
        logit   = total_funding - params.liquidity_floor
        q_new   = gibbs_threshold(logit, beta)         # (n,) in (0, 1)
        q_new   = jnp.clip(q_new, 0.0, 1.0)

        return SystemState(
            solvency  = x.solvency,
            liquidity = LiquidityState(fractions=q_new),
            price     = x.price,
        )

    return Operator(
        forward  = forward,
        adjoint  = _no_adjoint,   # replaced by wire_al_symmetry(S, L)
        name     = "L",
        is_gibbs = True,
        channel  = "L",
    )


def wire_al_symmetry(s_op: Operator, l_op: Operator) -> tuple[Operator, Operator]:
    """
    Wire AL symmetry: set S.adjoint = L.forward and L.adjoint = S.forward.

    Must be called after constructing both S_direct and L_direct.
    Returns updated (S, L) operator pair with adjoints correctly set.

    Usage:
        S = S_direct(s_params, beta=50.0)
        L = L_direct(l_params, beta=20.0)
        S, L = wire_al_symmetry(S, L)
        # Now: S.adjoint(x) = L.forward(x) and L.adjoint(x) = S.forward(x)
    """
    s_new = Operator(
        forward  = s_op.forward,
        adjoint  = l_op.forward,
        name     = s_op.name,
        is_gibbs = s_op.is_gibbs,
        channel  = s_op.channel,
    )
    l_new = Operator(
        forward  = l_op.forward,
        adjoint  = s_op.forward,
        name     = l_op.name,
        is_gibbs = l_op.is_gibbs,
        channel  = l_op.channel,
    )
    return s_new, l_new


# ---------------------------------------------------------------------------
# L_A — fire sale (indirect contagion via price impact)
# ---------------------------------------------------------------------------

@dataclass
class FireSaleParams:
    """
    Parameters for the fire-sale operator (L^A channel).

    Extracts the core tâtonnement logic from econiac.finance.fire_sales.
    Application papers (Paper 332) supply CHZParams; this is the generic version.

    Fields:
        portfolio:      (n_agents, n_assets) — quantity of each asset held
        capital_ratios: (n_agents,) — current γ_i = equity / RWA
        capital_floor:  γ_min — agents sell when γ < γ_min
        market_depth:   (n_assets,) — total securities outstanding
        lambda_impact:  price-impact coefficient
        risk_weights:   (n_assets,) — Basel III risk weights (for RWA calc)
        sell_utility:   (n_assets,) — utility of selling each asset (for Gibbs
                         portfolio allocation; default = uniform → proportional sell)
    """
    portfolio:      jax.Array    # (n_agents, n_assets)
    capital_ratios: jax.Array    # (n_agents,)
    capital_floor:  float        # γ_min
    market_depth:   jax.Array    # (n_assets,)
    lambda_impact:  float = 0.1
    risk_weights:   Optional[jax.Array] = None
    sell_utility:   Optional[jax.Array] = None


def L_A(params: FireSaleParams, beta: float = 50.0) -> Operator:
    """
    Fire-sale operator (L^A channel in Hurd's notation).

    Mechanism (CHZ §2.3 / Calimani-Hałaj-Żochowski 2020):
      1. Capital shortfall = max(0, γ_min − γ_i(P))
      2. Sell pressure (Gibbs): w_i = σ(β × capital_shortfall_i)
      3. Sell volume per asset: Gibbs-weighted portfolio allocation
      4. Price update: P ← tatonnement(P, sell_vol, market_depth)
      5. New solvency fractions from updated capital ratios

    Reads:  SolvencyState (capital ratios), PriceState
    Writes: PriceState + SolvencyState (via updated capital ratios)

    Adjoint note: The mathematical adjoint of L^A in the Pacioli inner product
    is a "price-to-capital" operator (injection of value from prices into equity),
    which has no natural economic interpretation as a contagion channel. The adjoint
    is therefore set to _no_adjoint — do not call it.

    Args:
        params: FireSaleParams
        beta:   inverse temperature for capital constraint softening
                (β → ∞ recovers hard γ_min floor; β = 50 is CHZ calibration)

    Returns:
        Operator with channel="LA"
    """
    portfolio    = params.portfolio     # (n, m)
    cap_floor    = params.capital_floor
    market_depth = params.market_depth  # (m,)
    lam          = params.lambda_impact

    # Sell utility: uniform by default (proportional sell across assets)
    sell_util = (
        params.sell_utility
        if params.sell_utility is not None
        else jnp.zeros(portfolio.shape[1])
    )

    def forward(x: SystemState, beta: float = beta) -> SystemState:
        prices   = x.price.prices        # (m,)
        sol_frac = x.solvency.fractions  # (n,)

        # --- Capital shortfall ---
        port_val = portfolio * prices                             # (n, m)
        port_tot = jnp.sum(port_val, axis=1)                    # (n,)
        # Current capital ratio: use solvency fraction as a proxy
        # (full model: capital_ratio = equity / RWA from BankBalanceSheet.capital_ratio())
        # Here we work directly with solvency fractions (p_i ≈ γ_i / γ_min)
        capital_ratio = sol_frac * cap_floor                      # (n,) — rescaled

        shortfall   = jnp.maximum(cap_floor - capital_ratio, 0.0)  # (n,)
        sell_weight = gibbs_threshold(shortfall, beta)              # (n,) — sell pressure

        # --- Portfolio allocation (Gibbs over assets) ---
        asset_weights = gibbs_weight_asset(sell_util, beta=1.0)    # (m,) uniform default

        # Sell quantity per agent per asset
        # Each agent sells shortfall_i × RWA_i worth; simplified to shortfall × port_tot
        sell_value_per_agent = shortfall * port_tot                 # (n,)
        sell_qty = (
            (sell_value_per_agent[:, None] * asset_weights[None, :])
            / jnp.maximum(prices, 1e-8)
        )   # (n, m)
        total_sell_qty = jnp.sum(sell_qty, axis=0)                  # (m,)

        # --- Price update ---
        prices_new = _tatonnement_price_step(
            total_sell_qty, prices, market_depth, lam
        )

        # --- Updated solvency fractions ---
        # Agents who sold enough to cover shortfall recover to floor;
        # agents in excess distress remain below floor.
        price_change    = jnp.sum(portfolio * (prices_new - prices), axis=1)  # (n,)
        capital_new     = capital_ratio + price_change / jnp.maximum(port_tot, 1e-8)
        sol_new         = jnp.clip(capital_new / cap_floor, 0.0, 1.0)

        return SystemState(
            solvency  = SolvencyState(fractions=sol_new),
            liquidity = x.liquidity,
            price     = PriceState(prices=prices_new),
        )

    return Operator(
        forward  = forward,
        adjoint  = _no_adjoint,
        name     = "L^A",
        is_gibbs = True,
        channel  = "LA",
    )


# ---------------------------------------------------------------------------
# S_D — bank panic / repo run operator
# ---------------------------------------------------------------------------

@dataclass
class PanicParams:
    """
    Parameters for the bank panic / repo run operator (S^D channel).

    Generalises across two application contexts:
      - Deposit run (Diamond-Dybvig): depositors withdraw when they expect others to
      - Repo run (Gorton-Metrick / Paper 333): lenders withdraw when haircut threshold
        is breached or collateral coverage falls below their floor

    Fields:
        funding_outstanding: (n_agents,) — total short-term funding per agent
        collateral_value:    (n_agents,) — net collateral value (after haircuts)
        coverage_floor:      threshold below which lenders run (default 0.80)
                             repo model: net_collateral / repo_out < 0.80 → run
        market_depth:        (n_assets,) — depth for fire-sale price impact
        lambda_impact:       price-impact coefficient
        portfolio:           (n_agents, n_assets) — for fire-sale allocation
    """
    funding_outstanding: jax.Array   # (n_agents,)
    collateral_value:    jax.Array   # (n_agents,) — net collateral coverage
    coverage_floor:      float = 0.80
    market_depth:        Optional[jax.Array] = None
    lambda_impact:       float = 0.1
    portfolio:           Optional[jax.Array] = None


def S_D(params: PanicParams, beta: float = 100.0) -> Operator:
    """
    Bank panic / repo run operator (S^D channel in Hurd's notation).

    Mechanism (Gorton-Metrick 2012 / repo_market.py):
      1. Coverage ratio: c_j = net_collateral_value_j / funding_outstanding_j
      2. Roll probability (Gibbs): p̃_j = σ(β × (c_j − coverage_floor))
      3. Funding withdrawn: F_j = funding_j × (1 − p̃_j)
      4. Forced selling to cover funding gap (proportional across assets)
      5. Price impact: P ← tatonnement(P, sell_vol)

    Reads:  LiquidityState (funding fractions), PriceState
    Writes: LiquidityState + PriceState

    Default beta = 100 (sharp but not a cliff; MMF uses beta=500 via gibbs_lift).
    For the full heterogeneous lender model (Paper 333), call S_D once per
    lender type with the appropriate beta, then compose their outputs.

    Adjoint note: Mathematical adjoint of S^D is a "liquidity-to-funding" operator
    with no standard economic interpretation. Set to _no_adjoint.

    Args:
        params: PanicParams
        beta:   inverse temperature (β → ∞ = hard run/no-run cliff)

    Returns:
        Operator with channel="SD"
    """
    Q   = params.funding_outstanding   # (n,)
    C   = params.collateral_value      # (n,) — net collateral
    floor = params.coverage_floor
    lam = params.lambda_impact
    portfolio = params.portfolio       # (n, m) or None
    depth = params.market_depth        # (m,) or None

    def forward(x: SystemState, beta: float = beta) -> SystemState:
        liq_frac = x.liquidity.fractions   # (n,) — current roll fractions
        prices   = x.price.prices          # (m,)

        # Coverage ratio: net collateral / funding outstanding
        # In stressed conditions C falls (price impact) and Q stays fixed
        # → coverage falls → roll probability falls → run accelerates
        coverage = C / jnp.maximum(Q, 1e-8)               # (n,)
        logit    = coverage - floor                         # (n,) signed distance
        roll     = gibbs_threshold(logit, beta)            # (n,) ∈ (0,1)
        roll     = jnp.clip(roll, 0.0, 1.0)

        # Funding withdrawn
        funding_withdrawn = Q * (1.0 - roll)               # (n,)

        # Fire sales to cover funding gap (if portfolio provided)
        prices_new = prices
        if portfolio is not None and depth is not None:
            port_val   = portfolio * prices                # (n, m)
            port_tot   = jnp.sum(port_val, axis=1, keepdims=True)
            port_w     = port_val / jnp.maximum(port_tot, 1e-8)
            sell_val   = port_w * funding_withdrawn[:, None]   # (n, m)
            sell_qty   = sell_val / jnp.maximum(prices, 1e-8)
            total_sell = jnp.sum(sell_qty, axis=0)             # (m,)
            prices_new = _tatonnement_price_step(total_sell, prices, depth, lam)

        return SystemState(
            solvency  = x.solvency,
            liquidity = LiquidityState(fractions=roll),
            price     = PriceState(prices=prices_new),
        )

    return Operator(
        forward  = forward,
        adjoint  = _no_adjoint,
        name     = "S^D",
        is_gibbs = True,
        channel  = "SD",
    )


# ---------------------------------------------------------------------------
# Rehyp — rehypothecation chain collapse (novel Econiac channel)
# ---------------------------------------------------------------------------

@dataclass
class RehypParams:
    """
    Parameters for the rehypothecation chain operator.

    Captures the collateral reuse mechanism in European sovereign repo markets:
    under stress, counterparties refuse to re-use uncertain collateral,
    shortening chains from 3.05× (calm) toward 1.0× (stressed).

    This is the amplification channel identified by ECB WP 3147 (2025) and
    the FSB (Feb 2026) as the key European systemic risk beyond Gorton's model.

    Fields:
        rehyp_rate_0:    (n_assets,) — baseline chain length per collateral class
                          [3.05, 2.10, 1.20, 0.30] from ECB WP 3147 / ICMA #50
        sigma_0:         baseline daily volatility (calm regime)
        beta_rehyp:      how sharply chains collapse under stress
        funding_outstanding: (n_agents,) — repo funding per dealer
        market_depth:    (n_assets,) — for price-impact rescaling
    """
    rehyp_rate_0:        jax.Array   # (n_assets,) — baseline chain lengths
    sigma_0:             float = 0.005
    beta_rehyp:          float = 10.0
    funding_outstanding: Optional[jax.Array] = None   # (n_agents,)
    market_depth:        Optional[jax.Array] = None   # (n_assets,)


def Rehyp(params: RehypParams, beta: float = 10.0) -> Operator:
    """
    Rehypothecation chain collapse operator (novel Econiac channel).

    Mechanism (ECB WP 3147 / repo_market.py):
      1. Rolling volatility σ_a(t) rises as collateral prices fall
      2. Chain collapse: rehyp_a(t) = 1 + (rehyp_0_a − 1) × σ(−β × (σ_a − σ_0))
      3. Effective collateral supply falls: effective = actual × rehyp(t)
      4. This reduces net collateral coverage → triggers more S^D (roll withdrawals)

    The Rehyp operator modifies PriceState: it adjusts the effective price
    that lenders see (actual_price × rehyp_factor, normalised), which then
    feeds into the S^D coverage ratio calculation.

    Implementation note:
        Rehyp does not directly modify SolvencyState or LiquidityState.
        It adjusts PriceState by the effective collateral multiplier, which
        then propagates through S^D in the composition Rehyp ∘ S^D.
        The effective price is: p_eff_a = p_a × rehyp_a(σ).

    Reads:  PriceState
    Writes: PriceState (effective prices scaled by rehyp multiplier)

    Adjoint: chain amplification on the way up (not a standard contagion channel;
    set to _no_adjoint).

    Args:
        params: RehypParams
        beta:   inverse temperature for chain collapse (= beta_rehyp in RehypParams)

    Returns:
        Operator with channel="Rehyp"
    """
    rehyp_0 = params.rehyp_rate_0   # (m,)
    sigma_0 = params.sigma_0

    def forward(x: SystemState, beta: float = beta) -> SystemState:
        prices = x.price.prices     # (m,)

        # Estimate rolling volatility from price level
        # (In the full model, σ is passed explicitly from the outer loop.
        # Here we estimate: σ_a ≈ max(0, 1 - p_a) as a proxy for cumulative stress.)
        sigma_proxy = jnp.maximum(1.0 - prices, 0.0)   # (m,) in [0, 1]

        # Rehypothecation multiplier (from gibbs.py)
        r = gibbs_rehyp(
            sigma     = sigma_proxy,
            rehyp_0   = rehyp_0,
            beta_rehyp= beta,
            sigma_0   = sigma_0,
        )   # (m,) ∈ [1, rehyp_0]

        # Effective price: the collateral is worth r times as much in
        # terms of funding capacity (during calm) or exactly face value
        # (during stress). We normalise so that r_calm prices are preserved.
        rehyp_calm = 1.0 + (rehyp_0 - 1.0) * jax.nn.sigmoid(
            -beta * (jnp.zeros_like(prices) - sigma_0)
        )
        prices_eff = prices * (r / jnp.maximum(rehyp_calm, 1e-8))
        prices_eff = jnp.maximum(prices_eff, 1e-4)

        return SystemState(
            solvency  = x.solvency,
            liquidity = x.liquidity,
            price     = PriceState(prices=prices_eff),
        )

    return Operator(
        forward  = forward,
        adjoint  = _no_adjoint,
        name     = "Rehyp",
        is_gibbs = True,
        channel  = "Rehyp",
    )


# ---------------------------------------------------------------------------
# Pre-built ESL compositions for convenience
# ---------------------------------------------------------------------------

def esl_operator(
    fire_params:  FireSaleParams,
    panic_params: PanicParams,
    gibbs:        Optional["GibbsParams"] = None,  # type: ignore[name-defined]
) -> Operator:
    """
    Standard ESL composition: C = L^A ∘ S^D  (Hurd 2017).

    Constructs L^A and S^D with the Gibbs parameters from ``gibbs``,
    then composes them. This is the two-channel baseline used in Papers 332
    and 333. For the three-channel repo model, use ``repo_esl_operator()``.

    Args:
        fire_params:  FireSaleParams — calibration for L^A
        panic_params: PanicParams    — calibration for S^D
        gibbs:        GibbsParams    — inverse temperatures
                      (default: GibbsParams() with standard calibration)

    Returns:
        Operator "L^A ∘ S^D"
    """
    from econiac.finance.contagion.gibbs import GibbsParams
    g = gibbs or GibbsParams()

    l_a = L_A(fire_params,   beta=g.beta_fire_sale)
    s_d = S_D(panic_params,  beta=g.beta_panic)

    from econiac.finance.contagion.operators import compose
    return compose(l_a, s_d)


def repo_esl_operator(
    fire_params:  FireSaleParams,
    panic_params: PanicParams,
    rehyp_params: RehypParams,
    gibbs:        Optional["GibbsParams"] = None,  # type: ignore[name-defined]
) -> Operator:
    """
    Extended ESL for the European repo market: C = Rehyp ∘ L^A ∘ S^D.

    Adds the rehypothecation channel (novel Econiac extension) before the
    standard ESL pair. The composition order is:
        1. S^D:   repo run → lenders withdraw → funding gap
        2. L^A:   dealers fire-sell collateral → prices fall
        3. Rehyp: price falls → σ rises → chains shorten → effective supply falls

    This is the primary operator for Paper 333 (Section 3-5).

    Args:
        fire_params:  FireSaleParams
        panic_params: PanicParams
        rehyp_params: RehypParams
        gibbs:        GibbsParams

    Returns:
        Operator "Rehyp ∘ L^A ∘ S^D"
    """
    from econiac.finance.contagion.gibbs import GibbsParams
    from econiac.finance.contagion.operators import compose
    g = gibbs or GibbsParams()

    s_d   = S_D(panic_params,   beta=g.beta_panic)
    l_a   = L_A(fire_params,    beta=g.beta_fire_sale)
    rehyp = Rehyp(rehyp_params, beta=rehyp_params.beta_rehyp)

    return compose(rehyp, l_a, s_d)
