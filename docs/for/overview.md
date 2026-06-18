# The Topology of Risk

> *"Just because I don't know what the connection is doesn't mean there isn't one."*
> — Douglas Adams, *The Long Dark Tea-Time of the Soul*

In 2008, every major bank's risk models passed their internal validation tests.
The system failed anyway — not because the models were wrong, but because they
were **locally correct and globally inconsistent**.

EconIAC is built on the insight that financial risk has three structural levels,
and that existing tools only address the first two.

---

## The three levels of financial risk

| Level | Name | What it is | Instruments | Who manages it |
| --- | --- | --- | --- | --- |
| $H^0$ | **Bilateral risk** | Risk between two parties | Forwards, swaps | Trading desks |
| $H^1$ | **Triangular risk** | Risk visible only when three parties interact | Options, swaptions, CDOs | XVA desks |
| $H^2$ | **Systemic risk** | Mutual inconsistency of institutions' triangular risks | CCPs, central banks | Regulators, CRO |

**Bilateral risk** ($H^0$) is perfectly hedgeable with forwards and swaps.
Every VaR model, every balance-sheet stress test, every bilateral exposure
report operates at this level.

**Triangular risk** ($H^1$) is the risk that only appears when three parties
interact. No portfolio of bilateral contracts can eliminate it — this is a
mathematical theorem, not a practical limitation. Convexity, basis risk,
correlation risk, CVA, FVA, and the volatility smile are all triangular risks.
Options and swaptions exist *because* $H^1 \neq 0$.

**Systemic risk** ($H^2$) is the risk of the whole: the mutual inconsistency of
individual institutions' triangular risk estimates. Wrong-way risk in XVA is $H^2$.
The 2008 crisis was $H^2$. No existing risk system computes $H^2$.

---

## What EconIAC computes that existing systems cannot

| Computation | Existing tools | EconIAC |
| --- | --- | --- |
| Bilateral stress test | ✅ Standard (VaR, balance sheet) | ✅ |
| Triangular risk ($H^1$) | Partial (XVA desks, CVA) | ✅ Full sheaf cohomology |
| Systemic stability ($H^2$) | ❌ Not computed anywhere | ✅ Pentagon identity test |
| Pre-crisis early warning | Contemporaneous indicators | ✅ $H^1$ leads cascade by 2–3 periods |
| Policy gradient $\partial\text{loss}/\partial\text{haircut}$ | ❌ | ✅ One backward pass |
| SIFI designation | Size-based (FSB) | ✅ Topological — $H^2$ contribution |
| Wrong-way risk in XVA | Parametric approximation | ✅ Exact $H^2$ class |

---

## Mathematical tools by user and problem

Each pillar of EconIAC addresses specific problems for specific users.
The grid below shows which tool is relevant to which problem — with the
specific result, not just a checkmark.

<div style="overflow-x:auto;">
<table>
<thead>
<tr>
<th>User / Problem</th>
<th>Gauge theory</th>
<th>Cohomology<br>(bilateral · triangular · systemic)</th>
<th>Thermodynamics<br>(Gibbs, β)</th>
<th>Sheaves</th>
<th>Differentiable ABM</th>
</tr>
</thead>
<tbody>

<tr>
<td><strong>Central bank /<br>stress testing</strong></td>
<td>FX triangular arbitrage = non-zero holonomy; FX reserves as parallel transport</td>
<td><strong>Core tool.</strong> Tier-0/1/2 stress test. H² stability class. SIFI theorem. 2008 as H² event.</td>
<td>Gibbs-lifted cascade: smooth sigmoid around hard threshold; β* = phase transition point</td>
<td>H¹ early warning: bilateral inconsistency 2–3 periods before cascade</td>
<td>Differentiable fire-sale and repo-run models; policy gradient ∂loss/∂haircut</td>
</tr>

<tr>
<td><strong>Bank / XVA desk</strong></td>
<td>CVA/DVA/FVA/MVA as gauge curvature on Pacioli manifold; Burgard–Kjaer PDE = flatness condition</td>
<td><strong>Core tool.</strong> CVA/FVA/MVA = H¹. Wrong-way risk = H². KVA at H¹/H² boundary. Model-free triangular pricing.</td>
<td>Rationality temperature β: calibrate from observed bid-ask spread variance</td>
<td>Model-free H¹: compute from CDX tranche prices, no parametric model needed</td>
<td>Smooth Greeks; differentiable netting set simulation; reverse stress test on XVA book</td>
</tr>

<tr>
<td><strong>Bank / market risk</strong></td>
<td>Yield curve as connection; convexity = curvature = H¹; vol surface as curvature field</td>
<td>Convexity = H¹ of discount factor sheaf (HJM drift = coboundary condition). Smile arbitrage = H¹ scanner.</td>
<td>Gibbs relaxation of hard VaR threshold; differentiable ES; β-parametric stress</td>
<td>Vol surface consistency: calendar + butterfly arbitrage as joint H¹ condition</td>
<td>Differentiable Monte Carlo; exact second-order sensitivities via jax.hessian</td>
</tr>

<tr>
<td><strong>Regulator / SIFI</strong></td>
<td>Balance sheet as gauge field; double-entry = flatness; money creation = curvature</td>
<td><strong>Core tool.</strong> SIFI = institution whose removal changes H². Size neither necessary nor sufficient. Topological capital charge.</td>
<td>Phase transition at β*: systemic fragility as thermodynamic bifurcation</td>
<td>Sheaf Laplacian on interbank graph; H¹ signal detects bilateral inconsistency</td>
<td>Differentiable resolution planning; optimal liquidation path via policy gradient</td>
</tr>

<tr>
<td><strong>Asset manager /<br>portfolio risk</strong></td>
<td>FX hedging as parallel transport; cross-currency basis = curvature; basis swap = holonomy</td>
<td>Basis risk = H¹ of funding sheaf. Correlation risk = H¹ of credit sheaf. CDO tranche price = H¹ class.</td>
<td>Thermal Shapley: attribute portfolio risk to factors in one backward pass</td>
<td>Correlation-consistent pricing: H¹ from basket option prices, model-free</td>
<td>Differentiable portfolio optimisation; exact attribution across correlated positions</td>
</tr>

<tr>
<td><strong>RegTech vendor</strong></td>
<td>Full gauge-theoretic finance stack: FX, rates, credit, XVA — one unified framework</td>
<td><strong>Core product.</strong> Cohomological stress testing as a service. H² monitor. Topological SIFI analytics. Model-free wrong-way risk.</td>
<td>Differentiable ABM platform: calibrate any contagion model by gradient descent</td>
<td>Real-time H¹ inconsistency scanner on EMIR data; leads cascade by 2–3 periods</td>
<td>All five Hurd contagion channels; policy gradient ∂H²/∂exposure; GPU-native</td>
</tr>

<tr>
<td><strong>Macro / climate<br>economist</strong></td>
<td>Double-entry accounting as discrete gauge theory; sectoral balances = Pacioli identity; carbon tax = gauge field</td>
<td>Supply chain inconsistency = H¹ of input-output sheaf; sector coupling = H² of multi-sector model</td>
<td><strong>Core tool.</strong> Rationality temperature β. Gibbs Keen/GEMMES. Differentiable Nash equilibria. Climate yield surface.</td>
<td>Stock-flow consistency: Godley table as sheaf; violation = non-zero coboundary</td>
<td>Differentiable GEMMES/LowGrow; calibrate to national accounts; reverse stress test climate scenarios</td>
</tr>

</tbody>
</table>
</div>

---

## Who EconIAC is for

- [**Central banks and regulators**](regulators.md) — cohomological stress testing,
  $H^2$ early-warning indicator, topological SIFI designation
- [**Banks and financial institutions**](banks.md) — XVA wrong-way risk, triangular
  risk classification, model-free correlation pricing
- [**Fund managers**](fund_managers.md) — $\beta_1$ portfolio topology, LDI crisis
  as $H^2$, currency overlays, ALM joint sheaf, topology arbitrage
- [**Broker-dealers and prime brokers**](broker_dealers.md) — matched book topology,
  rehypothecation chain depth, intermediation void risk, clearing scope
- [**RegTech and model vendors**](regtech.md) — the computational platform for
  bilateral/triangular/systemic risk as a service

---

## Start here

The fastest path to understanding the framework:

1. [The Topology of Risk: A Primer](https://doi.org/10.5281/zenodo.20642983) —
   13 pages, no prior mathematics required, Holmes vs Gently framing
2. [The Unhedgeability Theorem](https://doi.org/10.5281/zenodo.20635479) —
   the unhedgeability theorem and five-instance table
3. [Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) —
   cohomological stress test, SIFI theorem, XVA section
