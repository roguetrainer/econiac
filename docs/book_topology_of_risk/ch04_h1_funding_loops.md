# Chapter 4: Funding Loops and the First Betti Number

> *A borrows from B, B borrows from C, C borrows from A. Each bilateral contract is fine. The loop is not.*

---

## The example

Three money market funds: A, B, C.

- A holds commercial paper issued by B (A has lent to B)
- B holds commercial paper issued by C
- C holds commercial paper issued by A

Each bilateral relationship is perfectly sound. A has assessed B's credit; B has assessed C's; C has assessed A's. No bilateral risk manager sees a problem.

But A's ability to repay C depends on B repaying A, which depends on C repaying B, which depends on A repaying C. The three contracts form a **funding loop**: a cycle of mutual dependence that has no external anchor.

If any one firm faces a liquidity shock — even a temporary one — it propagates around the loop. A cannot pay C because B has not paid A; B cannot pay A because C has not paid B. The loop amplifies rather than absorbs the shock.

---

## $\beta_1$: the first Betti number

The **first Betti number** $\beta_1(\Gamma)$ counts the number of independent funding loops in the obligation complex. It is the dimension of $H^1(\Gamma)$ — the first cohomology group.

- $\beta_1 = 0$: no funding loops; the network is a tree; shocks do not amplify through cycles
- $\beta_1 = 1$: one independent loop; one potential amplification circuit
- $\beta_1 = k$: $k$ independent loops; $k$ independent contagion channels

A tree network (no loops) cannot amplify contagion — a default propagates only downstream, not in circles. A looped network can. The 2007--2008 buildup was a period of rapidly rising $\beta_1$: repo chains, CDO/ABCP structures, and tri-party repo created an enormous number of funding loops, each an independent contagion channel.

---

## CIP deviation as $H^1$

Covered interest parity (CIP) deviation is the most precisely measurable $H^1$ signal in financial markets.

The CIA loop for USD/EUR: borrow in USD, convert to EUR at spot, invest at EUR rate, convert back at the forward rate. If CIP holds, the round trip returns exactly one dollar. CIP deviation = the holonomy of this loop: how much more or less than one dollar the round trip yields.

Before 2008, CIP deviation was < 5bp — essentially zero. Since 2008, persistent deviations of 20--100bp have been documented across all major currency pairs. This is $H^1 \neq 0$: the CIA loop is a non-trivial cycle in the currency obligation complex. The cost of filling the 2-simplex (the arbitrage trade) is precisely the regulatory capital cost that prevents full arbitrage — the Curvature Floor Theorem.

CIP deviation is the cleanest example of $H^1$ risk: directly observable, persistently non-zero, arising from a structural constraint rather than a model assumption.

---

## $H^1$ as a leading indicator

$\beta_1(\Gamma_t)$ rises before financial crises. As institutions extend intermediation chains in benign conditions — lending to each other through longer and longer chains, creating more and more cycles — $\beta_1$ increases. The crisis occurs when the loops can no longer be sustained: a snap deleveraging that converts a high-$\beta_1$ network into a disconnected one (rising $\beta_0$).

Empirically, $\beta_1$ leads cascade by 2--3 reporting periods. This is the $H^1$ early-warning signal: rising loop count in low-volatility conditions is the topological signature of a system building fragility.

---

## The formal statement

$H^1(\Gamma) = \ker \partial_1 / \mathrm{im}\, \partial_2$. For a graph (no 2-simplices), this simplifies to $H^1(\Gamma) = \ker \partial_1$ — the space of cycles with no boundary. $\beta_1 = \dim H^1 = |E| - |V| + \beta_0$, where $|E|$ is the number of edges and $|V|$ the number of nodes. Each independent cycle is one funding loop that bilateral risk management cannot see.

---

*Next: [Chapter 5 — Netting and Clearing](ch05_netting_clearing.md)*
