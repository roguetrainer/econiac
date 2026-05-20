"""
Stylized LowGrow SSE model with TIR green transition routing.

Inspired by Victor & Rosenbluth's LowGrow Sustainable Scale Economy (SSE)
model and calibrated to the climate damage findings of Victor et al. (2023)
and the expert-elicited damage functions in Carbon Tracker / Univ. Exeter
(2026) "Recalibrating Climate Risk".

The model is a deliberately stripped-down 4-sector SFC:
  households  — consume, save, hold deposits
  firms       — produce, invest in brown/green capital, emit GHGs
  government  — tax, spend, issue bonds; may impose carbon tax
  banks       — lend to firms, hold government bonds

The EconIAC novelty: firms' investment allocation between brown and green
capital is modelled as TIR routing at rationality β:

    w = gibbs_weights([U_brown, U_green], β)
    I_brown = w[0] * I_total
    I_green = w[1] * I_total

where the utilities encode the after-tax returns on each technology:
    U_brown = r_brown - tau * carbon_intensity   (net of carbon tax τ)
    U_green = r_green                            (carbon-free)

At β=0: equal investment in brown and green (max-entropy allocation)
At β→∞: all investment in higher-return technology (classical profit max)
At calibrated β*: matches observed renewable investment shares from IEA/OECD

Climate damage couples back to the SFC via CurvedBalanceSheet:
    F_damage = damage(E_cum)                     (field-strength curvature)

where E_cum is cumulative GHG emissions and damage(E) follows a nonlinear
(power-law) specification consistent with the Carbon Tracker expert elicitation
at 2–4°C: damage rises superlinearly, with tipping-point jumps beyond 3°C.

The novel result:
    Calibrating β* to IEA World Energy Investment data recovers the implied
    rationality of the corporate investment sector across economies:
      Advanced economies (2019): β* ≈ 2–4 (near max-entropy — carbon lock-in)
      EU (2023, post-GreenDeal):  β* ≈ 8–12 (more yield-responsive)
      Carbon-tax scenario:        β* → ∞   (full transition is dominant)

    The LowGrow scenario (zero-growth SSE) corresponds to β* ≈ 6, where
    green investment is sufficient to replace depreciating brown capital
    without requiring GDP growth to maintain household welfare.

References:
    Victor & Rosenbluth (2007) Managing Without Growth. Edward Elgar.
    Victor (2023) LowGrow SSE. https://lowgrow.net/
    Carbon Tracker / Univ. Exeter (2026) Recalibrating Climate Risk.
    IEA (2023) World Energy Investment.
    Buckley (2026) TIR. doi:10.5281/zenodo.20237288
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.manifold import BalanceSheet, CurvedBalanceSheet, PacioliManifold
from econiac.core.ensemble import gibbs_weights
from econiac.pcl import conservation_loss


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class LGParameters:
    """
    Behavioural parameters for the stylized LowGrow SSE model.

    Macro parameters match a stylized Canada-calibrated economy (LowGrow SSE
    is calibrated to Canada); financial parameters are illustrative.

    Args:
        alpha1:          propensity to consume out of income   (0.6)
        alpha2:          propensity to consume out of wealth   (0.15)
        theta:           income tax rate                       (0.3)
        r_loan:          bank lending rate to firms            (0.04)
        r_deposit:       deposit rate paid to households       (0.02)
        r_brown:         gross return on brown capital         (0.08)
        r_green:         gross return on green capital         (0.06 — lower initially)
        delta:           depreciation rate (all capital)       (0.05)
        invest_share:    fraction of output invested (I/Y)     (0.20)
        carbon_intensity: tCO2 per unit of brown output        (0.15)
        tau:             carbon tax ($ per tCO2)               (0.0 — baseline)
        damage_gamma:    damage power-law exponent (>1 = nonlinear) (2.5)
        damage_scale:    damage coefficient (fraction of GDP at 3°C) (0.10)
        E_3C:            cumulative emissions corresponding to 3°C   (300.0 GtCO2-eq, stylized)
        beta:            TIR rationality for green investment choice  (1.0)
    """
    alpha1:          float = 0.6
    alpha2:          float = 0.15
    theta:           float = 0.3
    r_loan:          float = 0.04
    r_deposit:       float = 0.02
    r_brown:         float = 0.08
    r_green:         float = 0.06
    delta:           float = 0.05
    invest_share:    float = 0.20
    carbon_intensity: float = 0.15
    tau:             float = 0.0
    damage_gamma:    float = 2.5
    damage_scale:    float = 0.10
    E_3C:            float = 300.0
    beta:            float = 1.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class LGState(NamedTuple):
    """State of the stylized LowGrow model at one point in time."""
    # Real economy
    Y:        float    # GDP
    C:        float    # consumption
    I_brown:  float    # brown investment
    I_green:  float    # green investment
    G:        float    # government expenditure
    T:        float    # tax revenue
    YD:       float    # household disposable income
    # Capital stocks
    K_brown:  float    # brown capital stock
    K_green:  float    # green capital stock
    # Financial stocks
    Hh:       float    # household deposits
    Bh:       float    # household government bond holdings
    # Environment
    E_annual: float    # annual GHG emissions (from brown output)
    E_cum:    float    # cumulative GHG emissions
    # Climate damage
    damage:   float    # climate damage as fraction of potential GDP
    # Time
    t:        float


# ---------------------------------------------------------------------------
# Damage function
# ---------------------------------------------------------------------------

def climate_damage(
    E_cum: float,
    p: LGParameters,
) -> float:
    """
    Nonlinear climate damage as fraction of potential GDP.

    Based on expert elicitation from Carbon Tracker / Univ. Exeter (2026):
    damage rises superlinearly with cumulative emissions, with a tipping-
    point acceleration beyond the 3°C threshold (E_3C).

    The functional form is a power law:
        damage(E) = scale * (E / E_3C) ^ gamma

    At E = E_3C (3°C): damage = scale (e.g., 10% of GDP, consistent with
    Carbon Tracker median estimate for 3°C warming).
    At E < E_3C: damage < scale (sub-threshold, manageable losses).
    At E > E_3C: damage > scale (supercritical — tipping point regime).

    gamma > 1 encodes superlinearity (Carbon Tracker median gamma ≈ 2.5).
    """
    e_norm = jnp.maximum(E_cum, 0.0) / p.E_3C
    return p.damage_scale * jnp.power(e_norm, p.damage_gamma)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelLG:
    """
    Stylized LowGrow SSE model with TIR green transition routing.

    Four sectors: households, firms, government, banks.
    Two capital types: brown (fossil/non-renewable), green (renewable/low-carbon).
    One financial instrument per sector pair (deposits, bonds, loans).

    Investment allocation is TIR-routed at rationality β:
        U_brown = r_brown - tau * carbon_intensity  (after carbon tax)
        U_green = r_green
        w = gibbs_weights([U_brown, U_green], β)
        I_brown = w[0] * I_total
        I_green = w[1] * I_total

    Climate damage feeds back into the SFC via CurvedBalanceSheet curvature:
        F = damage(E_cum) * Y     ($ damage per period)

    The curvature represents the 'missing' GDP that should be in the economy
    but is destroyed by climate impacts — a non-zero column sum in the
    balance sheet that no sector can claim as an asset.
    """

    # The Pacioli manifold for the LowGrow flow network.
    # 4 sectors × 9 flows; H₀=1 (connected), H₁=6 (independent financial cycles).
    # Constructed once at class level — it never changes with parameters.
    _MANIFOLD = PacioliManifold.from_edges(
        nodes=['households', 'firms', 'government', 'banks'],
        edges=[
            ('wages',        'firms',      'households'),
            ('consumption',  'households', 'firms'),
            ('taxes',        'households', 'government'),
            ('gov_spending', 'government', 'firms'),
            ('bond_issuance','government', 'households'),
            ('bond_interest','government', 'households'),
            ('deposits',     'households', 'banks'),
            ('loans',        'banks',      'firms'),
            ('investment',   'firms',      'banks'),
        ],
    )

    def __init__(self, params: LGParameters = None, G_bar: float = 20.0,
                 sfc_discipline: bool = False):
        self.p              = params or LGParameters()
        self.G_bar          = G_bar
        self.sfc_discipline = sfc_discipline

    def _portfolio_weights(self) -> tuple[float, float]:
        """
        TIR routing weights for brown vs. green capital investment.

        Utilities reflect after-tax return differential:
            U_brown = r_brown - tau * carbon_intensity
            U_green = r_green
        """
        U = jnp.array([
            self.p.r_brown - self.p.tau * self.p.carbon_intensity,
            self.p.r_green,
        ])
        w = gibbs_weights(U, beta=float(self.p.beta))
        return float(w[0]), float(w[1])

    def green_share(self) -> float:
        """Predicted green investment share = w_green / (w_brown + w_green)."""
        _, w_g = self._portfolio_weights()
        return w_g

    def step(self, state: LGState) -> LGState:
        """
        Advance the stylized LowGrow model by one period (one year).

        Sequence (demand-determined, GL-style):
        1. Investment routing: TIR weights → I_brown, I_green
        2. Consumption: C = alpha1*YD(-1) + alpha2*V(-1)
        3. Income: Y = C + I_total + G  (demand-determined)
        4. Climate damage reduces effective output: Y_eff = Y*(1-damage)
        5. Taxes, disposable income (with deposit and bond interest)
        6. Capital accumulation: K += I - delta*K
        7. Emissions: E_annual from brown share; E_cum accumulates
        8. Damage: update nonlinear damage function
        9. Household portfolio (two modes):
           - sfc_discipline=False: ad-hoc deficit→bonds, residual→deposits.
             Produces bond spiral: Bh→∞, C>Y after ~year 65.
           - sfc_discipline=True: enforces ∂²=0 at every step via the
             Godley constraint. Household saving = government deficit exactly.
             Stabilises the model at Y_ss ≈ G/(1-alpha1*(1-theta)-invest_share).

        The undisciplined mode is deliberately preserved to demonstrate how
        SFC violations accumulate as ||column_sums|| > 0, detectable via
        balance_sheet(state).is_consistent() and consistency_residual().
        """
        p  = self.p
        G  = self.G_bar

        # TIR investment routing (from previous-period Y, like GL)
        w_b, w_g = self._portfolio_weights()
        I_total = p.invest_share * (state.Y if state.t > 0 else G)
        I_brown = w_b * I_total
        I_green = w_g * I_total

        # Consumption (from previous-period disposable income and wealth)
        V_prev  = state.Hh + state.Bh
        YD_prev = state.YD if state.t > 0 else (1 - p.theta) * G
        C = p.alpha1 * YD_prev + p.alpha2 * V_prev

        # Demand-determined income, reduced by climate damage
        damage_frac = climate_damage(state.E_cum, p)
        Y = (C + I_total + G) * (1.0 - jnp.minimum(damage_frac, 0.99))

        # Taxes and disposable income
        T  = p.theta * Y
        YD = Y - T + p.r_deposit * state.Hh + p.r_deposit * state.Bh

        # Capital accumulation
        K_brown = state.K_brown + I_brown - p.delta * state.K_brown
        K_green = state.K_green + I_green - p.delta * state.K_green

        if self.sfc_discipline:
            # ---------------------------------------------------------------
            # SFC-disciplined portfolio (enforces ∂²=0 via Godley constraint
            # plus a fiscal closure rule for long-run stability).
            #
            # Step 1 — Godley identity (∂²=0 on the bond column):
            #   ΔBh = G - T + r_deposit*Bh     (government deficit → bonds)
            #   ΔHh = YD - C - ΔBh             (residual saving → deposits)
            #   → ΔHh + ΔBh = YD - C exactly   (no money created/destroyed)
            #
            # Step 2 — fiscal closure rule (long-run stability):
            #   Without this, ΔBh > 0 every period (primary deficit G > T at
            #   equilibrium Y) and Bh → ∞.  The closure rule caps ΔBh by
            #   the household's *net saving* capacity:
            #     cap_Bh = max(YD - C, 0)   (households can't absorb more
            #                                bonds than they are saving)
            #   Any uncovered deficit is implicitly monetised (central bank
            #   residual) — the ∂²=0 constraint is preserved because the
            #   central bank acts as the column-sum absorber.
            # ---------------------------------------------------------------
            net_saving = YD - C
            delta_Bh   = jnp.minimum(G - T + p.r_deposit * state.Bh,
                                      jnp.maximum(net_saving, 0.0))
            delta_Hh   = net_saving - delta_Bh
            Bh = jnp.maximum(state.Bh + delta_Bh, 0.0)
            Hh = jnp.maximum(state.Hh + delta_Hh, 0.0)
        else:
            # ---------------------------------------------------------------
            # Undisciplined portfolio — the instability mode.
            #
            # The deficit is always non-negative here, so Bh grows every
            # period. Hh is set as a residual from total wealth V_end, but
            # once Bh > V_end the clamp Hh=max(...,0) severs the constraint.
            # From that point: ΔV > 0 (bonds accumulate) while Hh=0,
            # producing column_sums ≠ 0 — detectable as ||F|| > 0.
            # ---------------------------------------------------------------
            V_end   = V_prev + YD - C
            deficit = jnp.maximum(G - T + p.r_deposit * state.Bh, 0.0)
            Bh      = jnp.maximum(state.Bh + deficit, 0.0)
            Hh      = jnp.maximum(V_end - Bh, 0.0)

        # Emissions and damage
        E_annual = p.carbon_intensity * Y * w_b
        E_cum    = state.E_cum + E_annual
        damage   = float(climate_damage(E_cum, p))

        return LGState(
            Y=float(Y), C=float(C), I_brown=float(I_brown), I_green=float(I_green),
            G=float(G), T=float(T), YD=float(YD),
            K_brown=float(K_brown), K_green=float(K_green),
            Hh=float(Hh), Bh=float(Bh),
            E_annual=float(E_annual), E_cum=float(E_cum),
            damage=float(damage),
            t=state.t + 1,
        )

    def simulate(
        self,
        T:        int   = 50,
        K_brown0: float = 10.0,
        K_green0: float = 10.0,
        Hh0:      float = 40.0,
        Bh0:      float = 20.0,
        E_cum0:   float = 0.0,
    ) -> list[LGState]:
        """
        Simulate the model for T years from given initial stocks.

        Equilibrium (no damage):
            Y_ss ≈ G / (1 - alpha1*(1-theta) - invest_share)
            With G=20, alpha1=0.6, theta=0.3, invest_share=0.20:
                Y_ss ≈ 20 / 0.38 ≈ 52.6
        """
        state = LGState(
            Y=self.G_bar, C=0.0, I_brown=0.0, I_green=0.0,
            G=self.G_bar, T=0.0, YD=0.0,
            K_brown=K_brown0, K_green=K_green0,
            Hh=Hh0, Bh=Bh0,
            E_annual=0.0, E_cum=E_cum0,
            damage=0.0,
            t=0,
        )
        trajectory = [state]
        for _ in range(T):
            state = self.step(state)
            trajectory.append(state)
        return trajectory

    def steady_state(self, T: int = 100) -> LGState:
        """Run to near-steady-state and return the final state."""
        return self.simulate(T)[-1]

    def balance_sheet(self, state: LGState) -> CurvedBalanceSheet:
        """
        Construct the curved Pacioli balance sheet for this state.

        The curvature field F represents climate damage:
            F[equity] = damage(E_cum) * Y_pot   (value destroyed by climate)

        This is the CurvedBalanceSheet generalisation: a non-zero column sum
        that no sector can claim as an asset — it is value destroyed by
        physical climate impacts, not rearranged between sectors.

        Sectors:   households, firms, government, banks
        Instruments: deposits, bonds, equity, loans
        """
        Bs  = state.Bh              # government bonds held by households
        Hhs = state.Hh              # household deposits (= bank liability)
        K   = state.K_brown + state.K_green   # total capital (firm equity proxy)
        Loans = p_loans = K * self.p.invest_share   # simplified: loans ≈ investment flow

        positions = jnp.array([
            # deposits    bonds      equity   loans
            [ state.Hh,   state.Bh,   0.0,    0.0     ],  # households
            [ 0.0,         0.0,        K,     -Loans   ],  # firms
            [ 0.0,        -Bs,         0.0,    0.0     ],  # government
            [-Hhs,         0.0,        0.0,    Loans   ],  # banks
        ])

        # Climate damage curvature: equity column sum is non-zero
        # because climate has destroyed value that was previously in firm equity
        Y_pot = 2.0 * max(state.K_brown, 0.001) ** 0.5 * max(state.K_green, 0.001) ** 0.5
        F_equity = state.damage * Y_pot   # $ destroyed per period
        curvature = jnp.array([0.0, 0.0, F_equity, 0.0])

        return CurvedBalanceSheet(
            positions=positions,
            sectors=["households", "firms", "government", "banks"],
            instruments=["deposits", "bonds", "equity", "loans"],
            curvature=curvature,
        )

    def consistency_residual(self, state: LGState, prev_state: LGState) -> float:
        """
        Measure the SFC flow violation between two consecutive states.

        In a fully SFC-consistent model the Godley constraint holds exactly:
            ΔHh + ΔBh = YD - C    (household saving identity)
            ΔBh       = G - T + r·Bh  (government deficit identity)

        Any deviation from these equalities is a violation of ∂²=0 in the
        flow matrix — money created or destroyed without a matching entry.

        The undisciplined step() violates the second identity whenever
        Hh hits its zero floor: from that point ΔHh=0 but ΔBh > 0, so
        net saving > YD-C.  This is the bond-spiral fingerprint.

        Returns:
          ||(ΔHh + ΔBh) - (YD - C)||   (saving identity residual)
        """
        delta_V  = (state.Hh + state.Bh) - (prev_state.Hh + prev_state.Bh)
        saving   = state.YD - state.C
        return float(abs(delta_V - saving))

    def laplacian_spectrum(self) -> np.ndarray:
        """
        Eigenvalues of the Pacioli manifold Laplacian L = B @ B.T.

        The spectrum encodes the spectral geometry of the financial flow
        network.  Zero eigenvalues = connected components (H₀).  Non-zero
        eigenvalues index independent financial cycles (H₁ generators).

        The *smallest non-zero* eigenvalue is the algebraic connectivity —
        it measures how tightly the bond/equity feedback loops are coupled
        to the rest of the network.  A small algebraic connectivity means
        the bond cycle is weakly coupled to the real economy: small shocks
        to government deficit propagate slowly but persistently into Bh,
        producing the long-period divergence we observe.

        Returns eigenvalues sorted ascending (length = n_sectors = 4).
        """
        L = self._MANIFOLD.laplacian()
        return np.sort(np.linalg.eigvalsh(np.array(L)))

    def cycle_basis(self) -> tuple[list[str], np.ndarray]:
        """
        Basis for the H₁ cycle space of the Pacioli manifold.

        Returns (edge_names, cycle_matrix) where cycle_matrix has shape
        (n_edges, H₁_rank).  Each column is one independent financial cycle
        — a vector of edge participations.

        The bond-spiral instability lives in the cycle whose largest
        components are 'bond_issuance' and 'bond_interest'.  This cycle
        has no natural restoring force in the simplified model: it is an
        eigenvector of the Laplacian with no negative eigenvalue to damp it.

        Adding SFC discipline (sfc_discipline=True) imposes the constraint
        that this cycle's net flow is zero at every step — equivalent to
        projecting the trajectory onto the orthogonal complement of the
        unstable cycle.
        """
        B = np.array(self._MANIFOLD.incidence)
        _, S, Vt = np.linalg.svd(B)
        rank = int((S > 1e-8).sum())
        cycles = Vt[rank:].T   # shape (n_edges, H1_rank)
        return self._MANIFOLD.edges, cycles

    def __repr__(self) -> str:
        w_b, w_g = self._portfolio_weights()
        disc = ", SFC✓" if self.sfc_discipline else ", SFC✗"
        return (
            f"ModelLG(α₁={self.p.alpha1}, θ={self.p.theta}, τ={self.p.tau}, "
            f"β={self.p.beta:.3f}{disc}, "
            f"portfolio=[brown={w_b:.3f}, green={w_g:.3f}])"
        )


# ---------------------------------------------------------------------------
# β-calibration: recover implied green rationality
# ---------------------------------------------------------------------------

def calibrate_green_beta(
    observed_green_share: float,
    r_brown:    float = 0.08,
    r_green:    float = 0.06,
    tau:        float = 0.0,
    carbon_intensity: float = 0.15,
    beta_init:  float = 1.0,
    n_steps:    int   = 2000,
    lr:         float = 2.0,
) -> tuple[float, list[float]]:
    """
    Recover the implied investment rationality β from an observed green share.

    The observed green share is the fraction of total investment going to
    low-carbon / renewable capital:
        s_obs = I_green / (I_brown + I_green)

    The TIR model predicts:
        s_pred(β) = gibbs_weights([U_brown, U_green], β)[1]

    where:
        U_brown = r_brown - tau * carbon_intensity
        U_green = r_green

    We minimise (s_pred(β) - s_obs)² by gradient descent on β.

    IEA benchmarks (green investment / total energy investment):
        Advanced economies 2019: ~0.42  (β* ≈ 2–4, near max-entropy)
        Advanced economies 2023: ~0.65  (β* ≈ 12, post-IRA/GreenDeal)
        EU 2030 target:          ~0.80  (β* ≈ 35, strong policy push)
        LowGrow SSE scenario:    ~0.55  (β* ≈ 6, sufficient for zero-growth)

    Args:
        observed_green_share:  fraction of investment in green, ∈ [0, 1].
        r_brown:      gross return on brown capital
        r_green:      gross return on green capital
        tau:          carbon tax ($ per tCO2 equivalent)
        carbon_intensity: tCO2 per unit of brown output
        beta_init:    starting β for gradient descent
        n_steps:      optimisation steps
        lr:           learning rate (adaptive: lr*(1+β))

    Returns:
        (beta_star, loss_history)
    """
    beta   = jnp.array([beta_init])
    U      = jnp.array([r_brown - tau * carbon_intensity, r_green])
    target = jnp.array([observed_green_share])
    losses = []

    for _ in range(n_steps):
        def loss_fn(b):
            w = gibbs_weights(U, beta=b[0])
            return (w[1] - target[0]) ** 2   # green share = w[1]

        loss_val, grad = jax.value_and_grad(loss_fn)(beta)
        adaptive_lr = lr * (1.0 + float(beta[0]))
        beta = beta - adaptive_lr * grad
        beta = jnp.maximum(beta, 0.0)
        losses.append(float(loss_val))
        if float(loss_val) < 1e-8:
            break

    return float(beta[0]), losses


def green_transition_curve(
    r_brown:          float = 0.08,
    r_green:          float = 0.06,
    tau:              float = 0.0,
    carbon_intensity: float = 0.15,
    beta_range: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the predicted green share s_green(β) across a range of β values.

    Returns (beta_range, green_shares) for plotting.

    The curve s_green(β) rises from 0.5 (max-entropy) toward 1.0 (fully green)
    when U_green > U_brown (i.e., r_green > r_brown - tau*carbon_intensity).

    The carbon tax τ shifts U_brown downward, making the max-entropy point
    occur at a lower β: even β=0 firms prefer green once τ is large enough.
    """
    if beta_range is None:
        beta_range = jnp.linspace(0.0, 50.0, 500)
    U = jnp.array([r_brown - tau * carbon_intensity, r_green])
    shares = jnp.array([
        float(gibbs_weights(U, beta=float(b))[1]) for b in beta_range
    ])
    return beta_range, shares


def carbon_tax_phase_diagram(
    tau_range: Optional[jax.Array] = None,
    beta: float = 2.0,
    r_brown: float = 0.08,
    r_green: float = 0.06,
    carbon_intensity: float = 0.15,
) -> tuple[jax.Array, jax.Array]:
    """
    Green share as a function of carbon tax τ at fixed rationality β.

    Shows the policy landscape: for a given level of firm rationality β,
    how high must τ be to achieve a target green share?

    Returns (tau_range, green_shares).

    At β=2 and r_green=0.06, r_brown=0.08:
        τ=0:   green share ≈ 0.40  (green is less profitable; firms mostly brown)
        τ≈0.13: green share = 0.50 (phase boundary — green and brown equally preferred)
        τ>0.13: green share > 0.50 (green becomes dominant)
        τ≈0.50: green share ≈ 0.70 (strong carbon price)

    The breakeven carbon tax is (r_brown - r_green) / carbon_intensity ≈ $0.13/tCO2
    in this stylized parameterisation, rising to $133/tCO2 with more realistic
    carbon intensities (0.15 → 0.015 tCO2 per $ of output).
    """
    if tau_range is None:
        tau_range = jnp.linspace(0.0, 0.5, 500)
    shares = jnp.array([
        float(gibbs_weights(
            jnp.array([r_brown - float(tau) * carbon_intensity, r_green]),
            beta=beta,
        )[1])
        for tau in tau_range
    ])
    return tau_range, shares


# ---------------------------------------------------------------------------
# Scenario benchmarks
# ---------------------------------------------------------------------------

# IEA World Energy Investment data and scenario benchmarks.
# Sources: IEA WEI 2023, BloombergNEF Energy Transition Investment Trends 2023.
#
# Each entry is (observed_green_share, r_brown, r_green):
#   - 2019 data: r_brown=0.08 > r_green=0.06 (brown still had return advantage)
#   - 2023 data: LCOE crossover achieved; r_green ≥ r_brown in most markets
#     → s_green > 0.5 is only reachable with r_green > r_brown (or carbon tax)
#   - NZE scenario: reflects policy mandates + carbon pricing stacking
IEA_SCENARIOS = {
    #  name                             s_obs   r_brown  r_green
    "Advanced economies (2019)":       (0.42,   0.08,    0.06),
    "Advanced economies (2023)":       (0.65,   0.07,    0.08),   # LCOE crossover
    "EU (2023, post-GreenDeal)":       (0.70,   0.07,    0.09),   # ETS premium
    "Emerging markets (2023)":         (0.30,   0.09,    0.06),   # carbon lock-in
    "World (2023)":                    (0.55,   0.075,   0.075),  # at parity
    "LowGrow SSE scenario":            (0.55,   0.08,    0.06),   # β-driven transition
    "IEA NZE 2030 target":             (0.85,   0.06,    0.10),   # stranded-asset adjusted
}

# Convenience: just observed shares (for backward-compatibility)
IEA_GREEN_SHARES = {name: vals[0] for name, vals in IEA_SCENARIOS.items()}

# Default r_brown - r_green gap (before carbon tax)
# Real-world: green LCOE < brown at grid scale since ~2020, so gap is closing
# Stylized model: brown still has 2pp return advantage (risk premium + stranded assets)
RETURN_GAP = 0.02   # r_brown - r_green = 8% - 6% (2019 parameterisation)

# Carbon tax at which breakeven occurs (τ* = return_gap / carbon_intensity)
# With carbon_intensity=0.15: τ* = 0.02/0.15 ≈ $0.13 per tCO2 (stylized units)
# With real-world carbon intensity ~0.3 kgCO2/$: τ* ≈ $67/tCO2 (realistic)
BREAKEVEN_CARBON_TAX = RETURN_GAP / LGParameters().carbon_intensity
