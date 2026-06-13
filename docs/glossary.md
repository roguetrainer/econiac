# Glossary

Key terms used across the EconIAC framework. Entries cover the economic
and financial concepts specific to EconIAC. For the underlying mathematical
foundations — Fano plane, octonions, $G_2$, Origami ISA opcodes, pentagon
identity — see the
[ASA Glossary](https://roguetrainer.github.io/adelic-simplicial-architecture/glossary).

---

## Arbitrage

In EconIAC, **arbitrage** is not merely a trading opportunity — it is the
geometric signature of a non-zero **curvature** on the Pacioli manifold.
A triangular FX inconsistency (USD → EUR → GBP → USD ≠ 1) is a non-zero
holonomy on the currency bundle: parallel transport around the triangle
fails to return to the starting point.

*No arbitrage* ↔ *flat connection* ↔ *zero curvature* on the Pacioli manifold.

*See:* [Paper 295 (Currency Bundles)](https://doi.org/10.5281/zenodo.20242355),
[Paper 300 (Economic Gauge Theory)](https://doi.org/10.5281/zenodo.20259495)

---

## Bilateral Risk (H⁰)

**Bilateral risk** is the zeroth level of the cohomological risk hierarchy:
can each individual counterparty meet its obligations?

Formally: $H^0(\Gamma, \mathcal{F})$ = the space of globally consistent
sections of the pricing sheaf — the set of valuations under which every
bilateral exposure is exactly matched. A non-zero $H^0$ obstruction means
at least one counterparty cannot pay.

Bilateral risk is manageable by bilateral netting and standard credit
risk models (PD, LGD, EAD). It is the only level that pre-2008 models
measured.

*See:* [Paper 398 (Topology of Risk primer)](https://doi.org/10.5281/zenodo.20642983)

---

## Carnot Efficiency (η)

In EconIAC's thermodynamic framework, the **Carnot efficiency** $\eta$
measures how much of the available information-thermodynamic free energy
is converted into useful output.

The 6-731 broken-Fano topology (one Fano line weakened to coupling $r$)
achieves $\eta = 1 - r \approx 0.1825$ — matching the experimentally
measured FMO photosynthetic efficiency. This is not a coincidence: the
Fano symmetry breaking is the geometric source of both the biological
efficiency and the financial concept of the unhedgeable residual.

*See:* [Paper 325 (Topological Heat Engine)](https://doi.org/10.5281/zenodo.20400638)

---

## Climate Tipping Cascade

A **climate tipping cascade** occurs when crossing one tipping element's
threshold triggers others, producing an irreversible cascade. In the
EconIAC cohomological framework:

- $H^0$ = can each individual tipping element stay below its threshold?
- $H^1$ = are the bilateral coupling exposures globally consistent?
- $H^2$ = is there any intervention that can prevent cascade?

The $H^1$ inflection point is at $T^* \approx 1.8°C$ (between the Paris
targets of 1.5°C and 2.0°C). The $H^1$-corrected social cost of carbon
is $\approx \$316$/tonne, versus the bilateral baseline of $\approx
\$51$/tonne.

*See:* [Paper 403 (Tipping Points Are Topological)](https://doi.org/10.5281/zenodo.20653285),
[Paper 311 (Climate Hazard Yield Surface)](https://doi.org/10.5281/zenodo.20291646)

---

## Curvature (Financial)

The **curvature** of a financial instrument is the failure of parallel
transport to return to its starting point when transported around a
closed loop of trades. Concretely:

- **FX triangle:** USD→EUR→GBP→USD ≠ 1 means non-zero holonomy
- **XVA:** the valuation adjustment is the integral of curvature along
  the exposure path (the 6j symbol evaluated on the counterparty graph)
- **Yield curve:** the term structure is a temporal connection; convexity
  is its curvature

In the EconIAC library, curvature is computed via `econiac.pacioli`
connection objects. Arbitrage-free ↔ flat connection ↔ zero curvature.

*See:* [Paper 299 (XVA as Curvature)](https://doi.org/10.5281/zenodo.20257724),
[Paper 396 (6j Symbol as H¹)](https://doi.org/10.5281/zenodo.20635479)

---

## Differentiable Shapley Value

The **Differentiable Shapley value** computes marginal contributions via
the Gibbs ensemble at finite temperature $\beta$:

$$\phi_i(\beta) = \sum_{S \subseteq N\setminus\{i\}}
\frac{p_\beta(S)}{n} [v(S\cup\{i\}) - v(S)]$$

where $p_\beta(S) \propto \exp(-\beta\, \mathrm{cost}(S))$ is a Gibbs
weight over coalitions. At $\beta \to 0$: uniform (classical Shapley).
At $\beta \to \infty$: hard assignment to the most likely coalition.

The key advantage: $\phi_i(\beta)$ is differentiable in all model
parameters, so $\partial\phi_i/\partial\theta$ is available in one
backward pass via JAX autodiff. This enables gradient-based attribution
for systemic risk, carbon tax burden, and supply-chain criticality.

*See:* [Paper 293 (Thermal Attribution)](https://doi.org/10.5281/zenodo.20237288)

---

## Gauge Group

The **gauge group** of EconIAC is $(\mathbb{R}_{>0}, \times)$ — the
multiplicative group of positive reals. This encodes the invariance of
economic relationships under change of unit (currency, price level,
numeraire): scaling all prices by a positive constant leaves real
quantities unchanged.

This is a *local* gauge symmetry: different agents can use different
numeraires at different times, and the connection (the exchange rate)
encodes how to translate between them.

The gauge group $(\mathbb{R}_{>0}, \times)$ is abelian, making financial
gauge theory much simpler than the non-abelian gauge theories of physics
— but the mathematical structure (connection, curvature, holonomy) is
identical.

*See:* [Paper 291 (Topology of Conservation)](https://doi.org/10.5281/zenodo.20234853),
[Paper 301 (Primer on Economic Gauge Theory)](https://doi.org/10.5281/zenodo.20259505)

---

## Gibbs Distribution (Quantal Response Equilibrium)

The **Gibbs distribution** $p(x) \propto \exp(-\beta\, E(x))$ is
EconIAC's model of bounded rational agents. At $\beta \to \infty$
(zero temperature): perfect rationality (deterministic choice). At
$\beta \to 0$ (high temperature): uniform randomness. At finite $\beta$:
**Quantal Response Equilibrium** (QRE) — the empirically validated model
of human decision-making under uncertainty.

The parameter $\beta$ is **calibratable** from data via maximum
likelihood. Every EconIAC model is differentiable through its Gibbs
distributions, so $\partial\mathrm{welfare}/\partial\beta$ is available
in one backward pass.

*See:* [Paper 289 (Temperature of Rationality)](https://doi.org/10.5281/zenodo.20234841),
[Paper 313 (Thermal Economics)](https://doi.org/10.5281/zenodo.20318505)

---

## H⁰ / H¹ / H² (Cohomological Risk Hierarchy)

The three levels of financial risk, each corresponding to a sheaf
cohomology group of the pricing sheaf $\mathcal{F}$ on the exposure
network $\Gamma$:

| Level | Symbol | Name | Question answered |
|-------|--------|------|-------------------|
| 0 | $H^0$ | Bilateral | Can each counterparty pay? |
| 1 | $H^1$ | Triangular | Are bilateral exposures globally consistent? |
| 2 | $H^2$ | Systemic | Is there any consistent global resolution? |

**The 2008 crisis was an $H^2$ event**: the global interbank network
crossed a threshold beyond which no bilateral intervention (no netting,
no haircut, no bailout of individual institutions) could prevent cascade.
Standard models only measured $H^0$.

In `econiac.risk`: `cohomology_report(graph, section)` returns all three
levels in one call with a human-readable summary string.

*See:* [Paper 396 (6j Symbol as H¹)](https://doi.org/10.5281/zenodo.20635479),
[Paper 397 (Systemic Risk as H²)](https://doi.org/10.5281/zenodo.20642908),
[Paper 398 (Topology of Risk primer)](https://doi.org/10.5281/zenodo.20642983)

---

## Pacioli Identity

The **Pacioli identity** is the double-entry accounting principle: every
debit has a corresponding credit; every asset has a corresponding
liability. In EconIAC this is not a convention — it is the
**gauge invariance** of the economic system, the conservation law that
holds regardless of numeraire, currency, or unit of account.

Formally: the Pacioli identity is $d^2 = 0$ on the chain complex of the
account graph — the same pentagon identity that appears in the Origami
ISA as the no-arbitrage condition and in quantum gravity as the
Biedenhahn-Elliott identity for $6j$ symbols.

Money can be created by banks (by simultaneously creating an asset and a
liability), but the Pacioli constraint is always satisfied. Violations
of the Pacioli identity — accounts that do not balance — are detected by
`econiac.pacioli.pacioli_check()`.

*See:* [Paper 291 (Topology of Conservation)](https://doi.org/10.5281/zenodo.20234853)

---

## Pacioli Manifold

The **Pacioli manifold** $(M, \nabla)$ is the geometric space on which
EconIAC models live. Its points are economic states (stock-flow
consistent balance sheets); its tangent vectors are flows (transactions);
its connection $\nabla$ encodes exchange rates and discount factors.

The Pacioli manifold has gauge group $(\mathbb{R}_{>0}, \times)$. The
curvature of $\nabla$ measures arbitrage. Flat connections ($F_\nabla =
0$) are arbitrage-free. The holonomy of a closed loop of trades is the
total FX gain/loss or XVA contribution.

*See:* [Paper 291 (Topology of Conservation)](https://doi.org/10.5281/zenodo.20234853),
[Paper 409 (EconIAC Overview)](https://doi.org/10.5281/zenodo.20679006)

---

## Pricing Sheaf

The **pricing sheaf** $\mathcal{F}$ assigns to each node of the exposure
network $\Gamma$ a stalk (a vector space of prices/valuations) and to
each edge a restriction map (the bilateral exposure weight). A **global
section** of $\mathcal{F}$ is a consistent assignment of valuations
across all nodes — an equilibrium pricing.

The cohomology of $\mathcal{F}$:
- $H^0(\Gamma, \mathcal{F})$: the space of globally consistent pricings
- $H^1(\Gamma, \mathcal{F})$: triangular inconsistencies (XVA, wrong-way risk)
- $H^2(\Gamma, \mathcal{F})$: systemic irresolvability (the 2008 event)

The pricing sheaf is a strict generalisation of the **constant sheaf**
(Flood et al. 2017): where the constant sheaf gives topological Betti
numbers, the pricing sheaf gives financially loaded cohomology that
detects whether the network topology is carrying dangerous exposures.

*See:* [Paper 396 (6j Symbol as H¹)](https://doi.org/10.5281/zenodo.20635479),
[Paper 397 (Systemic Risk as H²)](https://doi.org/10.5281/zenodo.20642908)

---

## Sheaf Laplacian

The **sheaf Laplacian** $L_\mathcal{F} = \delta^{0\top}\delta^0$ is the
symmetric positive-semidefinite matrix built from the coboundary operator
$\delta^0$ on the pricing sheaf. Its eigenvalues measure the degree of
inconsistency in the exposure network:

- $\ker(L_\mathcal{F})$: globally consistent valuations ($H^0$)
- Near-zero eigenvalues: approximate consistency (low $H^1$ obstruction)
- Large eigenvalues: strongly inconsistent cycles

The **$H^1$ early-warning signal** is
$\|s - P_{\ker}s\| / \|s\|$ — the fraction of the health-ratio section
$s$ that cannot be reconciled to a consistent global valuation. This
signal leads cascade distress by 1–2 periods in the CHZ fire-sale
model (x332e).

In `econiac.risk`: `sheaf_laplacian(graph, section)` and
`h1_obstruction_signal(L_F, section)`.

*See:* [Paper 397 (Systemic Risk as H²)](https://doi.org/10.5281/zenodo.20642908)

---

## Social Cost of Carbon (SCC)

The **social cost of carbon** is the marginal damage of emitting one
additional tonne of CO₂. In EconIAC's cohomological climate framework:

- **Bilateral SCC** ($H^0$ baseline): $\approx \$51$/tonne — standard
  IAM estimates, ignoring tipping interactions
- **$H^1$-corrected SCC**: $\approx \$316$/tonne — accounts for
  triangular tipping inconsistencies (one tipping element triggering
  others via shared boundary conditions)
- **$H^2$-corrected SCC**: potentially unbounded — once the cascade
  is irresolvable, the marginal damage of the triggering emission
  includes the full cascade cost

The $H^1$ inflection at $T^* \approx 1.8°C$ is where the multiplier
jumps from $\approx 1\times$ to $\approx 6\times$ the bilateral baseline.

*See:* [Paper 403 (Tipping Points Are Topological)](https://doi.org/10.5281/zenodo.20653285)

---

## Stock-Flow Consistency (SFC)

**Stock-flow consistency** is the requirement that every flow (a
transaction) is accounted for in the balance sheets of both parties, and
that the sum of all sectoral balances equals zero. It is the macroeconomic
expression of the Pacioli identity.

In EconIAC, stock-flow consistency is enforced **algebraically** by the
gauge constraint on the Pacioli manifold — it cannot be violated by
construction. The `econiac.pacioli` module provides `SystemState` objects
whose `pacioli_check()` method verifies consistency and returns a
`PacioliReport` with the degree of any violation.

*See:* [Paper 291 (Topology of Conservation)](https://doi.org/10.5281/zenodo.20234853),
[Paper 316 (EconIAC: MONIAC for the 21st Century)](https://doi.org/10.5281/zenodo.20315689)

---

## Systemic Risk (H²)

**Systemic risk** is the $H^2$ level of the cohomological risk hierarchy:
the existence of a global state of the interbank (or climate, or supply
chain) network from which no bilateral or triangular intervention can
prevent cascade.

$H^2 \neq 0$ means the network has crossed a topological tipping point.
The **unhedgeable residual** is the component of stress that lies in
$H^2$ — it cannot be netted, haircut, or bailed out at the bilateral
level.

In `econiac.risk`: `h2_obstruction(graph, section)` returns the H²
signal (fraction of edge inconsistency that is irresolvable) and
`is_h2_event` (boolean flag). `cohomology_report()` assembles all three
levels with a `'bilateral' | 'triangular' | 'systemic'` risk
classification.

*See:* [Paper 397 (Systemic Risk as H²)](https://doi.org/10.5281/zenodo.20642908),
[Paper 398 (Topology of Risk primer)](https://doi.org/10.5281/zenodo.20642983)

---

## Thermal Attribution

**Thermal attribution** is the differentiable Shapley value at finite
temperature $\beta$ — it attributes systemic risk, carbon tax burden, or
supply-chain criticality to individual agents or edges via a single
backward pass through the Gibbs distribution.

Unlike classical Shapley (exponential in the number of players), thermal
attribution runs in $O(n)$ time via automatic differentiation:
$\phi_i = \partial F / \partial v_i$ where $F(\beta)$ is the free energy
of the system at temperature $\beta$.

In `econiac.routing`: `attribution.thermal_shapley(model, β)`.

*See:* [Paper 293 (Thermal Attribution)](https://doi.org/10.5281/zenodo.20237288)

---

## Unhedgeable Residual

The **unhedgeable residual** is the component of financial stress that
cannot be eliminated by any bilateral netting, haircut, or collateral
arrangement. It is the $H^2$ component of the pricing sheaf cohomology:
the part of the edge inconsistency vector that lies in the kernel of the
coboundary operator $\delta^1$ and therefore cannot be "corrected" by any
triangle-level renegotiation.

The unhedgeable residual is what made the 2008 crisis unresolvable by
bilateral intervention: AIG's counterparty web had a non-zero $H^2$
component that no amount of bilateral netting could eliminate.

In `econiac.risk`:
`h2_obstruction(graph, section).unhedgeable` gives the per-triangle
unhedgeable residual; `.h2_signal` gives the scalar fraction.

*See:* [Paper 397 (Systemic Risk as H²)](https://doi.org/10.5281/zenodo.20642908)

---

## XVA (Valuation Adjustments as Curvature)

**XVA** (the family of valuation adjustments: CVA, DVA, FVA, KVA, MVA)
are, in the EconIAC framework, integrals of the curvature of the pricing
sheaf along the exposure path.

The $6j$ symbol — the fundamental object of the Origami ISA — is the H¹
obstruction of the representation sheaf over the counterparty interaction
diagram. XVA is a special case: the $6j$ symbol evaluated on the
counterparty tetrahedron (four parties, six bilateral exposures).

This identification gives XVA an exact combinatorial formula in terms of
the counterparty graph topology, without requiring a model for the
correlation structure.

*See:* [Paper 299 (XVA as Curvature)](https://doi.org/10.5281/zenodo.20257724),
[Paper 396 (6j Symbol as H¹)](https://doi.org/10.5281/zenodo.20635479),
[Paper 399 (Origami ISA as Financial Middleware)](https://doi.org/10.5281/zenodo.20645695)

---

*For the underlying mathematical foundations (Fano plane, $G_2$, octonions,
Origami ISA opcodes, pentagon identity, associamancy), see the
[ASA Glossary](https://roguetrainer.github.io/adelic-simplicial-architecture/glossary).*
