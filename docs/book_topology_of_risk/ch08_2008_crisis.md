# Chapter 8: The 2008 Crisis as an $H^2$ Event

> *Every bilateral stress test passed. Every triangular risk was hedged. The system failed anyway.*

---

## The three levels of the crisis

The 2008 financial crisis operated simultaneously at all three cohomological levels. Understanding which level drove which phase clarifies why each intervention worked or failed.

| Phase | Level | Event | Why existing tools missed it |
|---|---|---|---|
| 2004--2007 buildup | $H^1$ | Repo chains, ABCP conduits, CDO structuring | $\beta_1$ rising; bilateral VaR stable |
| Aug 2007 -- Sep 2008 | $H^1$ snap | Money market freeze, Bear Stearns | Loops deleveraged; $\beta_0$ rising |
| Sep--Oct 2008 | $H^2$ | AIG, Lehman prime brokerage, tri-party repo | Hollow tetrahedra; no resolution mechanism |
| Oct 2008 -- 2010 | $H^2$ resolution | Fed/Treasury interventions | Exogenous injection of interior points |

---

## The $H^1$ buildup (2004--2007)

The pre-crisis period was characterised by explosive growth in repo chains, ABCP conduits, and CDO structures. Each of these added cycles to the obligation complex. $\beta_1(\Gamma_t)$ rose rapidly throughout 2005--2007.

A repo chain of length $k$ (A lends to B, B lends to C, ...) creates $k-1$ new edges and one new cycle if the chain closes. ABCP conduits created triangles: bank $\to$ conduit $\to$ commercial paper investor $\to$ bank. CDO tranching created triangles between originating banks, SPVs, and investors.

Bilateral VaR remained stable or fell (correlations appeared low; volatility was compressed). $\beta_1$ was the signal; no one was computing it.

---

## AIG: the $H^2$ contributor

AIG Financial Products wrote CDS protection on super-senior CDO tranches. The protection sellers were the same banks that held the underlying mortgages, had structured the CDOs, and had lent to AIGFP via repo. The four-party structure — AIG, originating banks, CDO SPVs, money market funds — formed hollow tetrahedra throughout the structured credit complex.

$\Delta\beta_2(\mathrm{AIG})$ was large: removing AIG from the network would have reduced $\beta_2$ substantially. AIG was a SIFI in the topological sense — not because of its size (it was an insurer, not a bank) but because its positions completed hollow tetrahedra across the structured credit complex.

When AIG was downgraded in September 2008, it could not post the collateral required by its CDS contracts. The hollow tetrahedra became irresolvable. The Federal Reserve's \$182bn intervention was the exogenous injection of a 3-simplex: a new interior point that resolved the four-party conflicts by making AIG's counterparties whole.

---

## Lehman prime brokerage: $H^2$ in practice

The Lehman LBIE administration illustrated $H^2$ irresolvability in the clearest possible form. Hedge fund clients had posted assets to LBIE. LBIE had rehypothecated those assets to its own funding counterparties. Barclays had acquired parts of Lehman's US business but not LBIE. The LBIE administrators held the remaining obligations.

The four parties held mutually inconsistent claims. Each bilateral claim was legally valid. Each triangular sub-claim was consistent. The four-party structure was a hollow tetrahedron. The administration lasted over a decade — not because the lawyers were slow, but because there was no interior point. Resolution required a court-imposed settlement: an exogenous 3-simplex.

---

## What the interventions did topologically

| Intervention | Topological action | Effect on $H^2$ |
|---|---|---|
| AIG bailout (\$182bn) | Injected new node (US Treasury) as interior point of AIG tetrahedra | $\beta_2 \downarrow$ for AIG-spanning tetrahedra |
| TARP capital injections | Added equity to bank nodes — strengthened existing edges | Limited $\beta_2$ effect |
| Fed commercial paper facility | Added Fed as new node bridging CP market loops | $\beta_1 \downarrow$, some $\beta_2 \downarrow$ |
| Dodd-Frank CCP mandates | Novated IRS/CDS to CCPs — added star topology | $\beta_1 \downarrow$ for cleared products; $\beta_2$ partially reduced |
| Basel III | Reduced edge weights (lower leverage) | Did not change topology; $\beta_1$, $\beta_2$ unchanged |

The most effective interventions (AIG bailout, Fed CP facility) were the ones that added new nodes or edges to fill hollow tetrahedra. The least topologically effective (Basel III leverage limits) reduced weights on existing edges without changing the cycle structure.

---

## What we have learned

The three chapters on $H^0$, $H^1$, $H^2$ give a complete decomposition of financial risk:

- **$H^0$**: bilateral inconsistency — triangular arbitrage, transfer pricing errors, balance sheet mismatches. Resolved by bilateral contracts and EN-style clearing.
- **$H^1$**: funding loops — CIP deviation, CVA wrong-way risk, basis, convexity. Partially addressed by options, swaptions, CDS. Cannot be eliminated by bilateral contracts alone.
- **$H^2$**: irresolvable four-party conflicts — the hollow tetrahedra of the 2008 crisis, the Lehman prime brokerage administration, the LDI crisis. Cannot be resolved by any private netting or clearing. Requires either a regulatory intervention (exogenous 3-simplex) or a deep instrument that transfers the risk to a party outside the tetrahedron.

The framework is not a prediction machine. It is a language for asking the right questions: not "how large are the bilateral exposures?" but "is $\beta_2 > 0$, and if so, who are the $\Delta\beta_2$ contributors?"

---

## Where to go next

The EconIAC papers develop each of these themes formally:

- [The Topology of Risk: A Primer](https://doi.org/10.5281/zenodo.20642983) — 13-page entry point, no mathematics required
- [Systemic Risk as $H^2$](https://doi.org/10.5281/zenodo.20642908) — formal proofs of the SIFI theorem and Impossibility Theorem
- [The Topology Trade](https://doi.org/10.5281/zenodo.20746651) — tradeable strategies derived from the framework
- [Deep Instruments](https://doi.org/10.5281/zenodo.20746204) — instrument designs for transferring $H^2$ risk

All papers are freely available at [zenodo.org/communities/econiac](https://zenodo.org/communities/econiac/).
