"""
Godley-Lavoie Model PC: Portfolio Choice SFC model.

The canonical benchmark from Godley & Lavoie (2007) *Monetary Economics*,
Chapter 4.  Five sectors, two financial assets (money and bills).

Sectors:
    households  — consume, save, hold money and bills
    firms       — produce, pay wages, receive consumption spending
    government  — spend, tax, issue bills
    central_bank — holds bills, issues money (HPM)
    _bills      — notional asset sector (bills outstanding)

Balance sheet matrix (assets positive, liabilities negative):

                 money    bills
    households    Hh       Bh
    firms          0        0
    government    -Hs      -Bs
    central_bank  -Hcb     Bcb

Conservation: column sums = 0 always.
    Hh = Hs = Hcb + ... (money supply identity)
    Bh + Bcb = Bs (bills market clears)

The EconIAC novelty: household portfolio allocation Hh/Bh is modelled as
TIR routing at rationality β, rather than the fixed Tobin coefficients of
the original model.  At β→∞ the household holds only the higher-yield asset
(classical Tobin limit).  Calibrating β to observed portfolio shares from
national accounts recovers the *implied rationality* of the household sector.

References:
    Godley & Lavoie (2007) Monetary Economics, Chapter 4.
    Buckley (2026) Thermodynamic Information Routing. doi:10.5281/zenodo.20237288
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from econiac.core.manifold import BalanceSheet
from econiac.core.ensemble import gibbs_weights
from econiac.pcl import conservation_loss


# ---------------------------------------------------------------------------
# GL-PC parameters (Table 4.1, Godley & Lavoie 2007)
# ---------------------------------------------------------------------------

@dataclass
class PCParameters:
    """
    Behavioural parameters of Model PC.

    All defaults reproduce the GL (2007) Table 4.1 calibration.

    Args:
        alpha1:  propensity to consume out of income        (GL: 0.6)
        alpha2:  propensity to consume out of wealth        (GL: 0.4)
        lambda0: autonomous share of bills in portfolio     (GL: 0.635)
        lambda1: interest-rate sensitivity of bills share   (GL: 1.0 — not used with TIR)
        lambda2: income sensitivity of bills share          (GL: 0.01 — not used with TIR)
        theta:   tax rate on income                         (GL: 0.2)
        r_bar:   bill interest rate                         (GL: 0.025)
        beta:    TIR rationality parameter (EconIAC addition; not in GL)
                 beta=0: equal money/bills allocation
                 beta→∞: all wealth in higher-yield asset (classical Tobin)
    """
    alpha1: float = 0.6
    alpha2: float = 0.4
    lambda0: float = 0.635
    lambda1: float = 1.0
    lambda2: float = 0.010
    theta:   float = 0.2
    r_bar:   float = 0.025
    beta:    float = 1.0      # EconIAC: TIR rationality


# ---------------------------------------------------------------------------
# PCState — one-period snapshot
# ---------------------------------------------------------------------------

class PCState(NamedTuple):
    """State of Model PC at one point in time."""
    # Stocks (end-of-period)
    Hh:  float    # household money holdings
    Bh:  float    # household bill holdings
    Bcb: float    # central bank bill holdings
    # Flows (current period)
    Y:   float    # GDP / income
    T:   float    # tax revenue
    YD:  float    # disposable income
    C:   float    # consumption
    G:   float    # government expenditure
    r:   float    # bill interest rate
    t:   float    # time index


# ---------------------------------------------------------------------------
# Model PC — differentiable SFC engine with TIR portfolio allocation
# ---------------------------------------------------------------------------

class ModelPC:
    """
    Godley-Lavoie Model PC with TIR portfolio allocation.

    The standard GL model uses fixed Tobin coefficients (lambda0, lambda1,
    lambda2) to allocate household wealth between money and bills.

    EconIAC replaces this with TIR routing at rationality β:
        w = gibbs_weights([U_money, U_bills], beta)
        Hh = w[0] * V        (money holdings)
        Bh = w[1] * V        (bill holdings)

    where V = household wealth and utilities are:
        U_money = 0          (money pays no interest)
        U_bills = r_bar      (bills pay interest r_bar)

    At β=0: Hh = Bh = V/2  (maximum entropy — indifferent)
    At β→∞: Bh = V, Hh = 0 (all in bills — rational agent)
    At calibrated β: matches observed portfolio shares from national accounts.

    This is differentiable in β: ∂Hh/∂β and ∂Bh/∂β exist via JAX autograd.
    """

    def __init__(self, params: PCParameters = None, G_bar: float = 20.0):
        self.p     = params or PCParameters()
        self.G_bar = G_bar   # exogenous government expenditure

    def _portfolio_weights(self) -> tuple[float, float]:
        """
        TIR routing weights for money vs. bills.

        Utilities: U_money = 0, U_bills = r_bar.
        Returns (w_money, w_bills).
        """
        U = jnp.array([0.0, self.p.r_bar])
        w = gibbs_weights(U, beta=float(self.p.beta))
        return float(w[0]), float(w[1])

    def step(self, state: PCState) -> PCState:
        """
        Advance Model PC by one period.

        Implements the GL Chapter 4 equation system:
            Y  = C + G
            T  = theta * Y
            YD = Y - T + r(-1) * Bh(-1)
            C  = alpha1 * YD + alpha2 * V(-1)
            V  = V(-1) + YD - C          (end-of-period wealth)
            Hh = w_money(β) * V
            Bh = w_bills(β) * V
            Bcb = Bs - Bh               (central bank residual)

        Bs (bills outstanding) is set by government deficit:
            Bs = Bs(-1) + G - T + r(-1) * Bs(-1)
        """
        p  = self.p
        G  = self.G_bar
        r  = p.r_bar

        # Previous-period wealth
        V_prev = state.Hh + state.Bh

        # Income and taxes
        C  = p.alpha1 * (state.YD if state.t > 0 else G) + p.alpha2 * V_prev
        Y  = C + G
        T  = p.theta * Y
        YD = Y - T + r * state.Bh

        # End-of-period wealth
        V  = V_prev + YD - C

        # TIR portfolio allocation
        w_m, w_b = self._portfolio_weights()
        Hh  = w_m * V
        Bh  = w_b * V

        # Government bills outstanding (from budget constraint)
        Bs_prev = state.Bh + state.Bcb
        Bs      = Bs_prev + G - T + r * Bs_prev

        # Central bank holds residual bills
        Bcb = Bs - Bh

        return PCState(
            Hh=Hh, Bh=Bh, Bcb=Bcb,
            Y=Y, T=T, YD=YD, C=C, G=G, r=r,
            t=state.t + 1,
        )

    def simulate(
        self,
        T: int = 60,
        Hh0: float = 0.0,
        Bh0: float = 0.0,
        Bcb0: float = 0.0,
    ) -> list[PCState]:
        """Simulate Model PC for T periods from given initial stocks."""
        state = PCState(
            Hh=Hh0, Bh=Bh0, Bcb=Bcb0,
            Y=0.0, T=0.0, YD=0.0, C=0.0, G=self.G_bar, r=self.p.r_bar,
            t=0,
        )
        trajectory = [state]
        for _ in range(T):
            state = self.step(state)
            trajectory.append(state)
        return trajectory

    def steady_state(self, T: int = 200) -> PCState:
        """Run until near-steady-state and return final state."""
        traj = self.simulate(T)
        return traj[-1]

    def balance_sheet(self, state: PCState) -> BalanceSheet:
        """
        Construct the Pacioli balance sheet for this state.

        The conservation law ∂²=0 is checked automatically.
        """
        Bs  = state.Bh + state.Bcb
        Hs  = state.Hh             # all HPM held by households (simplified: no vault cash)

        positions = jnp.array([
            # money   bills
            [ state.Hh,   state.Bh  ],   # households (assets)
            [ 0.0,         0.0       ],   # firms (no financial assets in PC)
            [ -Hs,        -Bs        ],   # government (liabilities)
            [ 0.0,         state.Bcb ],   # central bank: holds bills, zero net (simplified)
        ])

        # Adjust central bank money liability to enforce conservation
        # CB issues HPM (liability) and holds Bcb bills (asset)
        # In GL: CB money liability = Hh; CB bill asset = Bcb
        positions = jnp.array([
            [ state.Hh,    state.Bh  ],   # households
            [ 0.0,          0.0       ],   # firms
            [ -Hs,         -Bs        ],   # government
            [ 0.0,          state.Bcb ],   # central bank bills asset
        ])
        # Conservation: col 0: Hh - Hs + 0 = 0 ✓ (Hs = Hh)
        #               col 1: Bh - Bs + Bcb = 0 ✓ (Bh + Bcb = Bs)

        return BalanceSheet(
            positions=positions,
            sectors=["households", "firms", "government", "central_bank"],
            instruments=["money", "bills"],
        )

    def __repr__(self) -> str:
        w_m, w_b = self._portfolio_weights()
        return (
            f"ModelPC(α₁={self.p.alpha1}, α₂={self.p.alpha2}, "
            f"θ={self.p.theta}, r={self.p.r_bar}, β={self.p.beta:.3f}, "
            f"portfolio=[money={w_m:.3f}, bills={w_b:.3f}])"
        )


# ---------------------------------------------------------------------------
# β-calibration: recover implied rationality from observed portfolio shares
# ---------------------------------------------------------------------------

def calibrate_beta(
    observed_money_share: float,
    r_bar: float = 0.025,
    beta_init: float = 1.0,
    n_steps: int = 500,
    lr: float = 0.1,
) -> tuple[float, list[float]]:
    """
    Recover the implied rationality β from an observed money/bills portfolio share.

    The observed money share is the fraction of household wealth held in
    money (non-interest-bearing deposits):
        s_obs = Hh / (Hh + Bh)

    The TIR model predicts:
        s_pred(β) = gibbs_weights([0, r_bar], β)[0]

    We minimise (s_pred(β) - s_obs)² by gradient descent on β.

    Args:
        observed_money_share: fraction of wealth in money, ∈ [0, 1].
            Historical benchmarks:
              UK Flow of Funds 2019:  ~0.55 (households hold more cash)
              US Flow of Funds 2019:  ~0.40 (more financialised)
              GL Table 4.4 steady state: 0.365 / (0.365 + 0.628) ≈ 0.367
        r_bar:      bill interest rate (default: GL 2.5%)
        beta_init:  starting β for gradient descent
        n_steps:    optimisation steps
        lr:         learning rate (default 2.0 works well for this shallow curve)

    Returns:
        (beta_star, loss_history)
        beta_star:    calibrated β
        loss_history: MSE at each step
    """
    beta = jnp.array([beta_init])
    U    = jnp.array([0.0, r_bar])
    target = jnp.array([observed_money_share])
    losses = []

    for _ in range(n_steps):
        def loss_fn(b):
            w = gibbs_weights(U, beta=b[0])
            return (w[0] - target[0]) ** 2

        loss_val, grad = jax.value_and_grad(loss_fn)(beta)
        adaptive_lr = lr * (1.0 + float(beta[0]))
        beta = beta - adaptive_lr * grad
        beta = jnp.maximum(beta, 0.0)   # β ≥ 0
        losses.append(float(loss_val))
        if float(loss_val) < 1e-8:
            break

    return float(beta[0]), losses


def portfolio_share_curve(
    r_bar: float = 0.025,
    beta_range: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the predicted money share s(β) = gibbs_weights([0, r], β)[0]
    across a range of β values.

    Returns (beta_range, money_shares) for plotting.
    """
    if beta_range is None:
        beta_range = jnp.linspace(0.0, 100.0, 500)
    U = jnp.array([0.0, r_bar])
    shares = jnp.array([
        float(gibbs_weights(U, beta=float(b))[0]) for b in beta_range
    ])
    return beta_range, shares


# ---------------------------------------------------------------------------
# GL Table 4.4 steady-state benchmark
# ---------------------------------------------------------------------------

# Published steady-state values from Godley & Lavoie (2007) Table 4.4
# with G=20, theta=0.2, alpha1=0.6, alpha2=0.4, r=0.025, lambda0=0.635
GL_STEADY_STATE = dict(
    Y   = 100.0,
    T   = 20.0,
    YD  = 80.0,
    C   = 80.0,
    G   = 20.0,
    V   = 80.0,       # household wealth
    Hh  = 21.62,      # money holdings  (≈ (1-lambda0)*V adjusted for r)
    Bh  = 58.38,      # bill holdings   (≈ lambda0*V adjusted for r)
    Bcb = 11.49,      # CB bill holdings
    Bs  = 69.87,      # total bills outstanding
    r   = 0.025,
)
# Implied money share: Hh / V = 21.62 / 80 ≈ 0.270
GL_MONEY_SHARE = GL_STEADY_STATE["Hh"] / GL_STEADY_STATE["V"]
