# Why does EconIAC use connections and curvature?

*Conservation says every financial claim has a matching counter-claim.
Flatness says the path doesn't matter. Only the first is always true.*

---

## Conservation vs flatness

The Pacioli Combinator Library enforces ∂²=0 by construction: every debit has a credit, column
sums of the balance sheet are exactly zero. This is the double-entry identity — not a claim
that money cannot be created, but a claim that money can only be created by simultaneously
creating a matching liability. When a bank makes a loan it creates both an asset (the loan
receivable) and a liability (the deposit); when the central bank issues reserves it acquires
an asset (a bond) and creates a liability (reserves). ∂²=0 holds throughout. The same
structure governs Kirchhoff's current law at a junction: current is not destroyed, it is
rerouted — and every outflow has a matching inflow.

The flatness assumption is correct as a matter of accounting identity. It is frequently wrong as a
matter of data. EconIAC provides three mechanisms to handle the gap — not by abandoning
conservation, but by being explicit about where and why it is violated.

---

## The four cases

### 1. Measurement error and statistical discrepancy

National accounts are assembled from surveys, administrative records, and census data collected by
different agencies at different times. The headline GDP figures are reconciled by statisticians who
insert a **statistical discrepancy** line to make the totals agree. The discrepancy is real, it is
published, and it is sometimes large (US: routinely ±$50–100 billion per quarter).

The error is in the measurement, not in the double-entry identity. Every transaction balanced;
we just do not know exactly which sectors it moved between.

### 2. Reporting lags and float

At the moment a cheque is written, the payer's balance sheet shows a reduction in deposits. The
payee's balance sheet does not yet show the corresponding increase — the payment is **in transit**.
At any snapshot, the consolidated balance sheet appears to violate conservation. Over the interval
from sending to settlement, it does not.

The same phenomenon appears in interbank clearing, cross-border payments (correspondent banking),
and any system where transmission takes time. The books balance *eventually*, not instantaneously.

### 3. Genuine arbitrage — non-zero curvature

Two assets that ought to trade at the same price do not. A share of Apple trades at slightly
different prices on NASDAQ and the LSE. A government bond has a different yield than an equivalent
synthetic constructed from swaps. Covered interest parity fails persistently in some currency pairs.

In gauge-theoretic language: the Pacioli manifold is **curved**. The field strength
F = dA + A∧A is non-zero. A round trip — buy on one exchange, sell on the other, repatriate the
profit — does not return you to your starting point. The holonomy of the connection is non-trivial.

This is not a violation of conservation. It is a violation of **flatness** — the stronger condition
that the manifold has no curvature. Conservation (∂²=0) and flatness (F=0) are distinct. PCL
enforces conservation; EconIAC's geometry module provides the language for curvature.

### 4. Risk-adjusted mispricing — incomplete markets

The apparent arbitrage in case 3 is not free money. It persists because it is compensation for a
risk that the balance sheet does not represent: liquidity risk, settlement risk, regulatory capital
cost, currency repatriation risk. The price gap is fair given the full risk space; it looks like
mispricing only because the model's instrument set is incomplete.

The correct response is to extend the balance sheet — add the missing instrument (liquidity buffer,
regulatory capital account, repatriation reserve) — not to relax conservation. When the full risk
is on the balance sheet, the apparent arbitrage disappears.

---

## The three relaxation mechanisms

### Mechanism 1: `ResidualFlow` — explicit discrepancy accounting

For cases 1 and 2, EconIAC introduces a **residual sector** convention. Rather than letting an
imbalance hide in rounding error or disappear from the model, it is routed through a designated
`_residual` sector (or instrument):

```python
from econiac.pcl import flow, sequence

# Known flows
wages     = flow("firms",    "households", "deposits", 1000.0)
taxes     = flow("households", "government", "deposits",  200.0)

# Observed discrepancy: $7 unaccounted
discrepancy = flow("households", "_residual", "deposits", 7.0)

quarterly = sequence(wages, sequence(taxes, discrepancy))
```

Conservation is preserved — the `_residual` sector absorbs the imbalance. During model
calibration, the residual's magnitude enters the loss function: a well-fitted model minimises
the discrepancy. A poorly-fitted model has a large, persistent residual — a diagnostic, not a
hidden error.

For reporting lags, `_residual` is replaced by an explicit **float account**:

```python
in_transit = flow("payer", "_float", "deposits", payment)
settlement = flow("_float", "payee", "deposits", payment)
```

The float account has zero balance at settlement. Its time profile is observable and auditable.

### Mechanism 2: `CurvedBalanceSheet` — non-flat Pacioli manifold

For case 3, EconIAC extends `BalanceSheet` to carry a **curvature field** F — a skew-symmetric
tensor measuring the arbitrage / risk-premium at each point of the manifold:

```python
from econiac.core.manifold import BalanceSheet, CurvedBalanceSheet

bs = CurvedBalanceSheet(
    positions=...,
    sectors=...,
    instruments=...,
    curvature=F,   # field strength tensor; zero recovers flat PCL
)
```

The type system is extended: instead of checking `col_sums ≈ 0`, it checks
`col_sums ≈ F(bs)`. When F=0, `CurvedBalanceSheet` reduces to `BalanceSheet`; the flat PCL
type system is a special case.

Computations on a curved manifold accumulate **holonomy** — the path-dependent residual of a
round trip. Non-zero holonomy is the mathematical fingerprint of a genuine arbitrage opportunity.
EconIAC can compute it explicitly:

```python
from econiac.core.manifold import holonomy

h = holonomy(round_trip_computation, curved_balance_sheet)
# h ≈ 0: no arbitrage
# h ≠ 0: arbitrage profit equals ||h||
```

### Mechanism 3: `conservation_loss()` — soft enforcement for calibration

For gradient-based calibration against noisy real-world data, hard conservation is too brittle.
A model that fits historical national accounts must tolerate the statistical discrepancy in the
data — it cannot demand that the ONS or BEA data satisfy ∂²=0 to machine precision.

`conservation_loss()` replaces the binary `typecheck()` with a differentiable penalty:

```python
from econiac.pcl import conservation_loss

loss = conservation_loss(comp, balance_sheet, sigma=10.0)
# Returns ||col_sums(comp(bs))||² / sigma²
# sigma: measurement noise scale (e.g. £10bn for national accounts)
# At sigma → 0: hard conservation (infinite penalty for any violation)
# At finite sigma: Bayesian likelihood under Gaussian measurement noise
```

This is the **soft typecheck** — conservation is a prior, not a constraint. The calibration
minimises total loss (fit to data + conservation penalty), trading off the two according to σ.
When σ is estimated from the known measurement uncertainty of the data source (ONS publishes
standard errors for GDP components), the conservation penalty is automatically calibrated.

---

## What is not relaxed

**The PCL combinators themselves are never modified.** `flow`, `sequence`, `parallel`, `choose`,
`fold`, and `repeat` always enforce ∂²=0 by construction. The relaxation mechanisms are layered
on top:

- `_residual` and `_float` sectors live inside the balance sheet — the combinators see them as
  ordinary sectors and still enforce conservation across all sectors including the residual.
- `CurvedBalanceSheet` extends the type system — it does not change the combinators.
- `conservation_loss()` is a loss function used during calibration — it is not a combinator.

The circuit analogy holds: Kirchhoff's current law is never suspended. What changes is whether we
model an unaccounted current as a short circuit (error), a capacitor (lag), or a battery
(curvature source).

---

## What the central bank's liability is a claim on

Under a gold standard, the central bank's liability (banknotes) is redeemable
in a commodity — the connection has an external anchor, the manifold is flat.
Under fiat money, the central bank's reserve liability is redeemable in more
of the same liability. ∂²=0 still holds — the balance sheet balances — but
the *substance* of the liability is the continued functioning of the monetary
network itself, a topological rather than a balance-sheet property.

Different monetary regimes correspond to different gauge fixings of the unit
of account. Switching from gold standard to fiat is a gauge transformation;
the underlying real economic relationships are unchanged, but the reference
point for the unit of account floats free. This is why models calibrated
under one monetary regime give systematically different answers from models
calibrated under another — and why EconIAC's `CurvedBalanceSheet` makes the
gauge choice explicit rather than hiding it in a calibration constant.

For a fuller treatment see [What is money a claim on?](money.md).

---

## Design principle

> **Make imbalances visible, not impossible.**

Strict flatness catches bugs during development. Explicit residuals, curvature fields, and soft
penalties handle reality during deployment. The two modes are complementary: build with hard
conservation, calibrate with soft conservation, diagnose with the residual and the holonomy.
