# EconIAC for Broker-Dealers and Prime Brokers

> *"Dealer banks are now driven by profit per unit of balance sheet and
> repo is not at the top of the list."*
> — Manmohan Singh, IMF, December 2022

---

## The topology of intermediation

Broker-dealers occupy a structurally unique position in the financial
system: they are the nodes through which the obligation complex is
routed. Every bilateral trade, every repo, every securities lending
transaction, every prime brokerage relationship adds an edge to the
network. The topology of that network — not its aggregate notional —
determines whether the system can resolve itself in stress.

The H^k framework gives broker-dealers a precise language for the
risk they intermediate:

| Level | Risk type | Where it appears on the dealer book |
| --- | --- | --- |
| $H^0$ | **Bilateral risk** | Single-counterparty exposure; netting set |
| $H^1$ | **Funding loop risk** | Back-to-back repo chains; prime brokerage rehypothecation; matched book loops |
| $H^2$ | **Intermediation void risk** | Dealer withdrawal creates irresolvable four-party conflict in client network |

---

## The matched book and $\beta_1$

The matched book — in which a dealer offsets each client position with
a back-to-back hedge — appears bilateral on each leg but creates a
funding loop at the system level. Client A lends security S to the
dealer; the dealer lends S to client B; client B's collateral is
rehypothecated back to client A. The triangle is $H^1$: a funding
loop that the dealer's bilateral netting set does not capture.

As balance-sheet constraints tighten (leverage ratio, NSFR), dealers
reduce their matched book. Each eliminated match removes edges from the
obligation complex, reducing $\beta_1$ locally. But if the match was
bridging a cycle that clients relied on for funding, its removal
increases $\beta_1$ elsewhere — the loop migrates rather than
disappears.

**The $\beta_1$ budget:** a dealer with a topology capital charge
(Paper 426) is incentivised to track not just their netting set
exposures but their marginal contribution to sector $\beta_1$: how
many independent funding loops rely on their intermediation. Reducing
that contribution — by restructuring matched books to break rather
than merely relocate loops — reduces the topology charge.

---

## Prime brokerage and rehypothecation chains

Prime brokerage creates rehypothecation chains: hedge fund A posts
collateral to prime broker P; P rehypothecates to its own funding
counterparty B; B rehypothecates onward. Each link is a directed
edge in the obligation complex. A chain of length $k$ creates a
1-simplex path of length $k$, which participates in cycles if any
two endpoints are connected through another route.

The 2008 Lehman prime brokerage failure was an $H^2$ event: hedge
funds, Lehman, Barclays (which acquired the US business), and LBIE
administrators held mutually inconsistent claims on rehypothecated
assets that no bilateral close-out rule could resolve. The competing
claims formed a hollow tetrahedron — four parties, all four faces
present, interior empty. No private instrument could fill it.

**Rehypothecation depth as a topology metric:** the rehypothecation
chain depth $d$ (how many times an asset is re-used) is a direct
contributor to $\beta_1$. EconIAC computes the marginal $\beta_1$
contribution of each additional rehypothecation step, enabling
prime brokers to price the topological cost of chain depth into
their securities lending fees.

---

## Market-making withdrawal and the void

When a dealer withdraws from market-making in a product — whether due
to balance-sheet constraints, VAR limits, or risk appetite — it does
not just remove liquidity. It removes edges from the obligation complex.
If those edges were part of cycles, their removal changes $\beta_1$.
If the dealer was the sole intermediary connecting two sub-networks,
its withdrawal can disconnect them, increasing $\beta_0$ (isolated
components).

The critical case is the **intermediation void**: a dealer's withdrawal
creates $H^2 \neq 0$ in the residual network because the clients it
was connecting held mutually inconsistent positions that the dealer's
book was implicitly resolving. Without the dealer, the conflict is
irresolvable.

This is the topological explanation for why Treasury market liquidity
deteriorated after 2010 despite rising volumes: dealers reduced their
warehousing role (fewer edges), their clients' positions became less
connected (higher $\beta_0$, higher $\beta_1$), and the market
became more susceptible to intermediation void events.

---

## What EconIAC provides for broker-dealers

### Balance-sheet topology optimisation

Given a fixed balance-sheet budget, which trades to accept and which
to decline to minimise marginal topology capital charge $\Delta\beta_2$?
EconIAC computes:

- $\Delta\beta_1(\text{trade})$: marginal contribution of each new
  trade to the obligation complex's loop count
- $\Delta\beta_2(\text{trade})$: whether a new trade completes a
  hollow tetrahedron in the sector complex
- Optimal matched-book restructuring: which back-to-back pairs to
  eliminate to break loops rather than relocate them

### Rehypothecation chain analytics

- Chain depth $d$ vs $\beta_1$ contribution: price the topology cost
  into securities lending fees
- Minimum-depth rehypothecation structure achieving the same funding
  for the client: replace long chains with short ones
- Identify which assets in the chain have the highest cycle participation
  number $\nu$ — the marginal $\beta_1$ reduction from releasing them

### Stress scenario topology

For each stress scenario (counterparty default, market dislocation,
regulatory action):

- Does the scenario create $H^2 \neq 0$ in the residual network?
- Which client positions become mutually inconsistent if the dealer
  withdraws from intermediation?
- What is the minimum intervention (fewest new edges added back) that
  restores $H^2 = 0$?

This is the topological version of resolution planning: not just
*"who loses money if we fail?"* but *"which conflict cycles does our
failure create that no private agent can resolve?"*

### CCP clearing topology

Under EMIR/Dodd-Frank, dealers are the primary clearing members of
CCPs. Theorem 2 of Paper 439 shows that partial novation — clearing
some legs of a multi-product cycle but not others — leaves $\beta_1$
unchanged. Dealers who clear IRS and CDS but not FX at the same CCP
are leaving cross-product funding loops intact.

EconIAC identifies which cross-product cycles survive partial clearing
and computes the $\beta_1$ reduction achievable from simultaneous
multi-product clearing — quantifying the topology benefit of extending
clearing scope.

---

## The regulatory picture

Under the topology capital charge framework (Paper 426), dealers face:

- A charge proportional to $\Delta\beta_2(i)$ — their marginal
  contribution to irresolvable sector conflict cycles
- An incentive to reduce rehypothecation chain depth (reduces $\beta_1$)
- An incentive to accept cross-product clearing mandates (reduces
  cross-product $\beta_1$)
- An incentive to maintain market-making in products where their
  withdrawal would create an intermediation void

These incentives are better aligned with systemic stability than
balance-sheet size caps, which penalise large dealers regardless of
their topology contribution.

---

## Papers

| Paper | Content |
| --- | --- |
| [430 — The Topology of Intermediation](https://doi.org/10.5281/zenodo.20694463) | Broker-dealers, prime brokerage, and void risk under the Deep Framework |
| [439 — Cohomological Theory of Clearing](https://doi.org/10.5281/zenodo.20740253) | Six theorems on novation, compression, CCP concentration; $H^2$ impossibility |
| [397 — Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) | $H^2$ events; SIFI theorem; unhedgeability theorem |
| [426 — The Cohomological Regulator](https://doi.org/10.5281/zenodo.20701681) | Topology capital charge $K^\text{top}$; marginal $\Delta\beta_2(i)$ |
| [427 — XVA Desk](https://doi.org/10.5281/zenodo.20701689) | Wrong-way risk as $H^2$; KVA at SIFI boundary |
| [295 — Currency Bundles](https://doi.org/10.5281/zenodo.20242355) | FX intermediation as parallel transport; CIP deviation as curvature |
