# Why does econiac use differential geometry?

*Exchange rates multiply. That single fact forces you into geometry whether you want to be or not.*

---

## The problem with adding exchange rates

Suppose USD/EUR = 0.92 and EUR/GBP = 0.86. What is USD/GBP?

You multiply: 0.92 × 0.86 = 0.79. You do not add. This seems obvious — but it has a profound consequence. Exchange rates live in a *multiplicative* world, not an additive one. The mathematical structure of a multiplicative world is a **Lie group**: the positive reals under multiplication, (ℝ₊, ×).

Standard econometrics is built on additive models: linear regression, vector autoregressions, additive error terms. These tools apply naturally to quantities that live in additive worlds. Exchange rates, discount factors, survival probabilities, and price ratios do not. Fitting a linear model to log-prices and calling it "because of the multiplicative structure" is a workaround — it works, but it hides the geometry that governs the underlying relationships.

econiac works directly in (ℝ₊, ×). The geometry is not a complication; it is the right language.

## Paths compose: why connections are unavoidable

Consider converting USD to GBP. You can go directly (USD→GBP), or via EUR (USD→EUR→GBP), or via JPY and then EUR (USD→JPY→EUR→GBP). In a perfectly efficient market, all paths give the same result. In practice they do not — and the difference is not noise. It is structure.

The mathematical object that describes "how quantities transform as you move along a path" is called a **connection**. In econiac:

- An **exchange rate** is a connection on the currency bundle.
- A **yield curve** is a connection on the time bundle (how value transforms as you move forward in time).
- A **survival probability** is a connection on the credit bundle (how value transforms as you move through the risk of default).

These are not metaphors. The equations are identical to those used in gauge theory in physics, with money playing the role of the field and the Pacioli manifold playing the role of spacetime.

## Curvature is arbitrage

When you travel around a closed loop — USD→EUR→GBP→USD — and arrive back with a different amount than you started with, you have found an arbitrage. The mathematical name for "the failure of parallel transport around a closed loop to return to the starting value" is **curvature**.

In a perfectly efficient market, every connection is **flat** (zero curvature). No-arbitrage conditions are flatness conditions. This is not a new insight — it is the standard result of mathematical finance restated in geometric language — but the geometric language makes it computable and generalisable.

econiac computes curvature automatically. Given a set of market prices (exchange rates, yield curve quotes, CDS spreads), it returns the curvature of the corresponding connection. Large curvature = large arbitrage or mispricing. The XVA adjustments that banks compute (CVA, DVA, FVA) are, in this language, curvature integrals.

## Why "differential" geometry

The "differential" in differential geometry means you can take derivatives. This is what connects geometry to the rest of econiac.

Because connections and curvature are differentiable objects, you can:

- Compute the **gradient of the curvature** with respect to model parameters — this is the sensitivity (the "Greek") of an XVA adjustment.
- Run **gradient descent** to find the connection (the set of market prices) that minimises curvature — this is calibration.
- Integrate curvature over a surface to get a **global invariant** — this is the Chern class, which in financial terms measures the total arbitrage in a market sector.

None of these computations are available in frameworks that do not make the geometry explicit.

## You already use geometry — implicitly

The Heath-Jarrow-Morton framework for interest rate modelling is, in geometric language, the statement that the yield curve connection must be flat (no-arbitrage). The Black-Scholes formula is the solution to a heat equation on a flat connection. The Heston model adds curvature (stochastic volatility) as a correction.

Every time you use a no-arbitrage condition, you are asserting flatness of a connection. Every time you compute a sensitivity, you are differentiating a geometric object. econiac makes this explicit — so that the geometry works for you rather than hiding behind notation.

## The short version

You need differential geometry because:

1. Exchange rates multiply — this forces the geometry of (ℝ₊, ×), not additive linear algebra.
2. Exchange rates, yield curves, and survival probabilities are all **connections** on the Pacioli manifold — the same mathematical object described in different financial languages.
3. **Curvature is arbitrage** — no-arbitrage conditions are flatness conditions, automatically enforced and automatically differentiated in econiac.
4. You are already using geometry implicitly in every no-arbitrage model you write. econiac makes it explicit so you can compute with it directly.
