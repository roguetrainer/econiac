# Covered Interest Parity as an H¹ Condition

> *"The cross-currency basis should be zero. It isn't. And it hasn't been
> since 2008. The question is not whether the deviation is real — it is
> why it is structurally impossible to arbitrage away."*

Covered interest parity (CIP) is one of the most tested conditions in
international finance. It says: the cost of borrowing in one currency and
swapping into another via the forward market should equal the cost of borrowing
directly in the second currency. Before 2008, it held to within a few basis
points. Since 2008, persistent deviations of 20–100bp have been documented in
every major currency pair, correlated with regulatory capital costs and
quarter-end balance-sheet pressure.

The cohomological framework gives a precise answer to why these deviations
persist — and why they cannot be arbitraged away even when every participant
knows they exist.

---

## The CIA square

The covered interest arbitrage trade involves four quantities arranged in a
rectangle:

```
         S(t)
  USD ————————————> GBP
   |                 |
   | P_USD(t,T)      | P_GBP(t,T)
   |                 |
  USD ————————————> GBP
         F(t,T)⁻¹
```

- $S(t)$: spot rate (USD per GBP today)
- $F(t,T)$: forward rate (USD per GBP at time $T$)
- $P_{\text{USD}}(t,T)$: USD discount factor (price of $1 at $T$ in USD)
- $P_{\text{GBP}}(t,T)$: GBP discount factor (price of £1 at $T$ in GBP)

The **CIP condition** is that going around this square returns you to where
you started:

$$F(t,T) = S(t) \cdot \frac{P_{\text{USD}}(t,T)}{P_{\text{GBP}}(t,T)}$$

or equivalently, the **cross-currency basis** $\phi(t,T)$ is zero:

$$\phi(t,T) = \ln F(t,T) - \ln S(t) - \ln P_{\text{USD}}(t,T) + \ln P_{\text{GBP}}(t,T) = 0$$

---

## CIP is a holonomy condition (H⁰ and H¹)

In the language of the framework:

**H⁰:** each rate taken alone is internally consistent. The spot rate is
a well-defined price; the forward rate is a well-defined price; each
discount factor is a valid yield curve. No single rate is "wrong."

**H¹ = 0 ↔ CIP holds.** The four rates fit together around the square —
the loop $\text{USD} \to \text{GBP} \to \text{GBP}_T \to \text{USD}_T \to
\text{USD}$ returns a holonomy of exactly 1. The cross-currency basis is zero.
The square is the **boundary** of a filled-in 2-simplex — the no-arbitrage
surface exists, and the loop is trivial in H¹.

**H¹ ≠ 0 ↔ CIP fails.** The loop does not close. The cross-currency basis
$\phi(t,T) \neq 0$ is a non-trivial **H¹ class** of the currency sheaf. It is
not measurement error, not a bid-ask spread, not a model artefact. It is a
topological obstruction: a cycle that cannot be made into a boundary without
adding a 2-simplex.

The **Crown Jewel Theorem** (Paper 295) states this precisely:

> **CIP holds if and only if the connection on the currency bundle is flat.**
> CIP deviation = curvature = non-trivial H¹ class.

---

## Why can't you arbitrage it away?

Classical arbitrage theory says: if $\phi \neq 0$, trade the loop and pocket
the basis. Post-2008, this does not work — and the cohomological framework
explains exactly why.

Filling in the 2-simplex (making the CIA loop into a boundary) requires taking
on a balance-sheet commitment: you must simultaneously hold a spot FX position,
a forward FX position, and two interest rate positions for the full tenor $T$.
Each of these consumes regulatory capital under Basel III (leverage ratio,
LCR/NSFR). The **cost of filling the simplex** is the regulatory capital cost
of the four-leg trade.

The minimum achievable CIP deviation is therefore not zero — it is the minimum
curvature consistent with the balance-sheet constraints:

$$|\phi^*(t,T)| \geq f(\Lambda, \text{LCR}, \text{NSFR})$$

where $\Lambda$ is the Basel III leverage ratio requirement. Du, Tepper, and
Verdelhan (2018) document empirically that CIP deviations are systematically
correlated with leverage ratio pressure, quarter-end balance-sheet windows, and
dealer identity — exactly the pattern predicted by a curvature floor set by
regulatory capital costs.

**The topological reading:** post-2008 regulation made the 2-simplex expensive
to create. The H¹ class that used to be zero (cheap to trivialise) is now
persistently non-zero because the instrument that would trivialise it —
the fully funded four-leg CIA trade — costs more to hold than it earns.

---

## Central bank swap lines as topological surgery

A Federal Reserve swap line with (say) the ECB adds a new direct edge between
USD and EUR at the central bank level, bypassing the private dealer network.
In homological terms, this is **topological surgery**: a new edge is added to
the graph, which can fill in the CIA triangle at the central bank level even
when the private market cannot.

The swap-line theorem (Paper 295) states: a swap line between two central banks
partially cancels the CIP deviation but cannot fully eliminate it, because each
path in the private network is still subject to balance-sheet constraints. The
central bank fills one triangle; the private network still carries the residual
curvature on all other paths.

This is why swap lines stabilise FX markets in crises (2008, 2020) without
permanently eliminating the CIP basis: they reduce the H¹ class on the
central-bank triangle but leave the rest of the currency sheaf's curvature
unchanged.

---

## The multi-currency case: triangular arbitrage as H¹

With three currencies — USD, GBP, EUR — there are three CIA squares (one per
currency pair) and one triangular arbitrage condition:

$$S(\text{USD/GBP}) \cdot S(\text{GBP/EUR}) \cdot S(\text{EUR/USD}) = 1$$

The three CIA squares are edges in the currency complex; the triangular
arbitrage condition is the **face** — it is a 2-simplex. When spot markets are
liquid, this face is always filled: triangular spot arbitrage is fast, costless,
and keeps the triangle closed. The spot triangle contributes zero to H¹.

The forward triangle is different. Three CIA forward conditions
$\phi_{\text{USD/GBP}}(T)$, $\phi_{\text{GBP/EUR}}(T)$, $\phi_{\text{USD/EUR}}(T)$
each carry their own curvature. Even if each pairwise basis is non-zero, they
must satisfy a consistency condition around the triangle — a lower-dimensional
version of the Pentagon identity. If they do not, there is H¹ at the triangle
level: the three forward bases are mutually inconsistent, not just individually
non-zero.

This is the multi-currency basis inconsistency that appeared in
USD/EUR/JPY triangles during the 2008 and 2020 crises: each pairwise basis
looked explicable, but the three bases together failed to close the triangle.

---

## What this means for a rates desk

| Observable | Cohomological meaning | Instrument |
| --- | --- | --- |
| Cross-currency basis $\phi(t,T)$ | H¹ class of the currency sheaf | Cross-currency basis swap |
| CIP holds | H¹ = 0; connection is flat | No basis swap needed |
| Post-2008 persistent basis | H¹ ≠ 0; regulatory capital floors curvature | Basis swap; FX swap |
| Central bank swap line | Topological surgery; adds edge at CB level | N/A (CB instrument) |
| Triangular arbitrage closed | Forward triangle is a filled 2-simplex | Three-way FX package |
| Multi-currency basis inconsistency | H¹ at triangle level; three bases don't close | Three-way basis package |
| XVA on CIA trade | Cost of filling the 2-simplex | CVA/FVA/MVA charges |

The cross-currency basis is not noise — it is a signal about the topological
structure of the currency network. A desk that models it as a spread to be
fitted is working at H⁰. A desk that models it as a holonomy class is working
at H¹ — and can compute the policy gradient $\partial \phi / \partial \Lambda$
to answer: how much would the basis tighten if the leverage ratio were relaxed
by 1%?

---

## Further reading

- Buckley (2026). Paper 295: Currency Bundles on the Pacioli Manifold.
  doi:[10.5281/zenodo.20242355](https://doi.org/10.5281/zenodo.20242355)
  *(Crown Jewel Theorem, swap-line theorem, leverage ratio bound — full proofs)*
- Buckley (2026). Paper 296: Term Structure Bundles.
  doi:[10.5281/zenodo.20244445](https://doi.org/10.5281/zenodo.20244445)
  *(CIP as spatial flatness; HJM as temporal flatness — the unified space-time connection)*
- Du, W., Tepper, A. & Verdelhan, A. (2018). Deviations from Covered Interest
  Rate Parity. *Journal of Finance* 73(3), 915–957.
  *(The empirical documentation of post-2008 persistent CIP deviations)*
- Avdjiev, S., Du, W., Koch, C. & Shin, H.S. (2019). The dollar, bank leverage,
  and deviations from covered interest parity. *American Economic Review:
  Insights* 1(2), 193–208.
  *(Connects CIP deviations to bank balance-sheet constraints — the empirical
  counterpart of the leverage ratio bound in Paper 295)*
