# Why does Econiac use gauge theory?

*The choice of currency should not change the economics. The mathematical name for this obvious requirement is gauge invariance — and most models violate it.*

---

## The unit of account problem

Consider a simple model: firm A sells goods to firm B, who pays in euros. Now restate the model in dollars. The transaction is the same. The economy is the same. The model should give the same answer.

Does it? Not necessarily. If your model contains an equation of the form "profit = revenue − costs" and revenue is in euros but costs have a dollar component, the profit figure depends on the EUR/USD exchange rate in a way that is not economic content — it is an artefact of the chosen unit of account. A model that gives different economic predictions depending on whether you measure in euros or dollars has a bug. The bug is a failure of **gauge invariance**.

A gauge theory is a physical theory in which the predictions are independent of a local choice of reference frame. In electromagnetism, the reference frame is the phase of the electromagnetic potential — you can shift it by any smooth function without changing the observable fields. In economic gauge theory, the reference frame is the unit of account — you can rescale prices at each node of the Pacioli manifold by any positive real number without changing the real economic content.

The gauge group is (ℝ₊, ×): the positive reals under multiplication. A gauge transformation is a choice of numéraire.

## What gauge invariance rules out

Gauge invariance is a surprisingly strong constraint. It rules out:

**Additive price comparisons across sectors.** "The price level in sector A is 5 points higher than in sector B" is not gauge-invariant — it depends on the unit. "The ratio of prices in sector A to sector B is 1.05" is gauge-invariant.

**Interest rates stated as absolute levels.** "The risk-free rate is 4%" is not gauge-invariant across currencies. "The real yield spread between instrument X and instrument Y" is gauge-invariant.

**Welfare comparisons without a common numéraire.** Any model that adds utilities or profits across agents without a common gauge is implicitly assuming a particular unit of account. That assumption should be made explicit and its consequences examined.

In practice, most models make gauge choices implicitly (by working in a single currency, or normalising a price index to 1) and then forget that the choice was made. econiac requires gauge choices to be explicit — and checks that the outputs are gauge-invariant where they should be.

## Noether's theorem: conservation from symmetry

Emmy Noether's theorem (1915) states that every continuous symmetry of a physical system corresponds to a conserved quantity. In classical mechanics, time-translation symmetry gives energy conservation; spatial symmetry gives momentum conservation.

In economic gauge theory, the gauge symmetry — invariance under rescaling the unit of account — gives a conserved quantity via Noether's theorem. That conserved quantity is the **sectoral balance**: the net financial position of each sector of the economy. The conservation law is ∂²=0 restated in the language of symmetry.

This is the deepest result in econiac's theoretical foundations: **double-entry bookkeeping is the conserved charge of gauge symmetry**. It is not an accounting convention. It is the inevitable consequence of the requirement that economics is independent of the chosen unit of account.

## Connections and curvature, again

Once you have a gauge theory, the natural objects to study are:

- **Connections**: rules for comparing values across different nodes of the network (exchange rates, discount factors, survival probabilities). A connection tells you how to "parallel transport" a financial quantity from one sector or time to another.

- **Curvature**: the failure of parallel transport around a closed loop to return to the starting value. In financial terms: arbitrage, basis risk, or the path-dependence of XVA adjustments.

- **Holonomy**: the total accumulated effect of curvature around a loop. Triangular arbitrage is the holonomy of the FX connection around the triangle USD→EUR→GBP→USD.

These are not new financial concepts — they are existing financial concepts (no-arbitrage, XVA, basis risk) restated in a language that makes them computable and composable.

## Why gauge theory is the right language for finance

The existing language of quantitative finance — stochastic calculus, risk-neutral measure, HJM framework — is correct but incomplete. It handles individual instruments in isolation very well. It handles the *interaction* between instruments across sectors, currencies, and time horizons poorly — which is exactly where systemic risk lives.

Gauge theory handles interaction naturally, because the connection (the exchange rate, the discount factor) is the primary object, not the derived quantity. When you change the exchange rate, the gauge theory tells you automatically how every downstream quantity changes — including the curvature (the arbitrage opportunity) and its gradient (the trading signal).

econiac implements this: change an input market price, and every output — XVA adjustment, sectoral balance, policy sensitivity — updates automatically via JAX's autograd. The gauge structure ensures that the updates are economically meaningful, not unit-dependent artefacts.

## The short version

You need gauge theory because:

1. **The unit of account is a choice, not economics.** Models that give different predictions in different currencies have a bug. Gauge invariance is the requirement that fixes it.
2. **Noether's theorem connects the symmetry to a conservation law.** The conservation law is ∂²=0 — the accounting identity. Gauge invariance and double-entry bookkeeping are two statements of the same theorem.
3. **Connections and curvature are the natural language for exchange rates and arbitrage.** Every no-arbitrage condition is a flatness condition. econiac computes curvature — and its gradient — automatically.
