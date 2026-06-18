# EconIAC for Fund Managers

> *"The September 2022 LDI crisis was not a failure of duration management.
> Every fund had its duration right. The sector's joint $H^2$ was non-zero —
> and no fund manager was computing it."*

---

## The third dimension of portfolio risk

Every mainstream portfolio framework — Markowitz, Black-Litterman, factor
models, risk parity, LDI — optimises over two dimensions: expected return
and volatility. There is a third dimension that none of them address:
the **topology of the funding and collateral network**.

| Dimension | What it measures | Standard tools | EconIAC |
| --- | --- | --- | --- |
| Return | Expected gain | Mean-variance, BL | ✅ |
| Volatility | Dispersion of outcomes | VaR, CVaR, tracking error | ✅ |
| **Topology** | **Whether the portfolio can be liquidated in a crisis** | **❌ None** | **✅ $\beta_1$, $\beta^*(\rho)$, $\Delta\beta_2$** |

A portfolio's $\beta_1$ — the number of independent funding loops — and
$\beta_2$ — the number of irresolvable stress conflict cycles — determine
whether it survives a crisis. Volatility does not.

---

## What the UK LDI crisis revealed

In September 2022, UK gilt yields rose 150bp in days. Pension funds with
LDI strategies received margin calls. They sold gilts to meet them. Gilt
prices fell further. More margin calls. The Bank of England intervened.

**Why standard LDI risk management missed it:**

- $H^0$ ✅ — every fund's funding ratio was positive
- $H^1$ ✅ — every fund had adequate HQLA for expected redemptions
- **$H^2$ ❌ — the sector's joint $\beta_2$ was non-zero**

All funds held similar leveraged gilt positions. Their margin call
triggers were correlated. No individual fund could sell gilts without
worsening every other fund's position. This is $H^2$: a conflict cycle
that no bilateral or fund-level action can resolve.

The Bank of England's gilt purchase programme was the four-party
instrument that restored $H^2 = 0$ across the sector. Government
intervention was not merely helpful — it was **topologically necessary**
(see [Paper 397](https://doi.org/10.5281/zenodo.20642908), Theorem 4).

---

## The joint asset-liability sheaf

Standard ALM separates asset optimisation from liability management,
connecting them only through a scalar funding ratio. This is structurally
wrong.

The joint sheaf $F = F_A \times F_L$ can have $H^2(F_A \times F_L) \neq 0$
even when $H^2(F_A) = 0$ and $H^2(F_L) = 0$ separately. The coupling
between asset sales and liability triggers creates new conflict cycles not
present in either network alone.

The LDI gilt loop is the canonical example: margin calls on LDI derivatives
(asset event) force gilt sales → gilt prices fall → liability values rise
(liabilities discounted at gilt yields) → funding ratio worsens → more
margin calls. $H^2(F_A \times F_L) = 1$ at the fund level, replicated
across 1,000+ funds to give $H^2_\text{sector} \gg 1$.

---

## What EconIAC provides for fund managers

### Three new risk metrics

| Metric | Definition | Frequency | What it tells you |
| --- | --- | --- | --- |
| Portfolio $\beta_1$ | Independent funding loops in $F_A$ | Daily | How many cascade paths exist |
| $\beta^*(\rho)$ | Proximity to liquidation crisis threshold | Daily | How close to the edge |
| Sector $\Delta\beta_2$ | Your contribution to sector $H^2$ | Quarterly | Whether your strategy is adding systemic fragility |

These sit alongside standard metrics (VaR, CVaR, tracking error, information
ratio) and answer the question those metrics cannot: *can this portfolio be
liquidated without self-reinforcing cascade?*

### Mainstream frameworks rewritten

**Markowitz:** two portfolios with identical (return, volatility) can have
very different $\beta_1$. A long-only portfolio ($\beta_1 = 0$) and a
leveraged portfolio ($\beta_1 \gg 0$) with the same Sharpe ratio are not
equivalent. The H^k framework ranks the long-only portfolio strictly higher
under the same return/volatility constraint.

**Factor models:** factor crowding is $\beta_1$ growth — as more managers
hold similar exposures, $\beta_1(\text{factor network}) \to \beta^*(\rho)$,
and the factor becomes prone to a momentum crash. EconIAC provides a precise
crowding early-warning: when $\beta^*(\rho_\text{factor}) < \beta_1/m$,
the factor is approaching its fragility threshold.

**Risk parity:** allocates equal volatility but typically creates high
$\beta_1$ (leverage generates funding loops). March 2020 simultaneous
falls in equities and bonds — a risk parity diversification failure — was
a $H^1$ coupling event: both positions shared dealer repo funding. Volatility
measures were uncorrelated in historical data; the topological coupling
was invisible until stress.

**LDI:** correctly addresses $H^1(F_A \times F_L)$ at the fund level
(the duration gap). Does not address $H^2(F_A \times F_L)$ at the
sector level. The 2022 crisis is the proof.

### Currency overlays

A currency overlay is a TWIST opcode — it changes the numéraire of
positions without changing the underlying exposure. Standard overlay
analytics (hedge effectiveness, carry, transaction cost) do not measure
the topological cost: each OTC FX forward adds to $\beta_1$ if it
creates a new funding loop.

The optimal overlay minimises carry cost subject to $\beta_1 < \beta_1^\text{max}$
and $\beta_2 = 0$. Exchange-traded FX futures (star topology, $\beta_1 = 0$)
are preferred over OTC forwards when the dealer network $\beta^*(\rho)$ is
elevated.

### Alternative investments

Private equity creates low $\beta_1$ per position but high $H^2$ at the
LP commitment level: capital calls across multiple funds are correlated
in exactly the stress scenarios when liquid assets are most distressed.
The Yale endowment model is a classic example of $H^0$/$H^1$ optimisation
that produces latent $H^2$ exposure — the "liquidity illusion."

LTCM (1998) was an $H^2$ event: many spread trades were in topological
conflict, with $\beta_2 \gg 0$ visible in retrospect but invisible to
VaR-based risk management.

### New strategies enabled

| Strategy | Description |
| --- | --- |
| **Topology arbitrage** | When your $\beta^*(\rho) \ll$ sector $\beta^*(\rho)$, provide liquidity at a premium during sector stress |
| **Topology hedging** | Macro protection against $\beta^*(\rho_\text{sector})$ spikes — a hedge class not reducible to rates, credit, or volatility |
| **$H^2$-negative construction** | Positions with negative $\Delta\Delta\beta_2$ reduce sector topology risk; available at a discount with systemic risk premium |

---

## The five-step H^k ALM framework

1. **Measure $H^0$** (standard) — funding ratio, solvency ratio
2. **Measure $H^1$** (improved) — add $\beta_1(F_A \times F_L)$: the
   funding loops coupling asset sales to liability triggers
3. **Measure $\beta^*(\rho_\text{joint})$** (new) — if $> 0.7$, the
   ALM is topologically over-leveraged regardless of duration match
4. **Monitor sector $\beta_2$** (new, from regulator) — reduce sector
   contributions when sector $\beta_2$ rises
5. **Stress-test topology** (new) — for each stress scenario, check
   whether it creates new $H^2$ asset-liability coupling cycles;
   if yes, restructure pre-emptively rather than holding a capital buffer

---

## Papers

| Paper | Content |
| --- | --- |
| [429 — The Cohomological Fund Manager](https://doi.org/10.5281/zenodo.20702221) | Asset allocation, ALM, currency overlays, alternative investments under the Deep Framework |
| [397 — Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) | $H^2$ impossibility; government necessity corollary; 2008/2022 as $H^2$ events |
| [426 — The Cohomological Regulator](https://doi.org/10.5281/zenodo.20701681) | Topology capital charge; sector $\beta_1$/$\beta_2$ publication; SIFI theorem |
| [295 — Currency Bundles](https://doi.org/10.5281/zenodo.20242355) | FX overlay as TWIST opcode; cross-currency basis as curvature |
| [311 — Climate Yield Surface](https://doi.org/10.5281/zenodo.20291646) | Climate risk in portfolio construction; doomsday isocurve |
| [433 — The Climate Deep](https://doi.org/10.5281/zenodo.20733942) | Joint topology of physical and financial climate risk |
