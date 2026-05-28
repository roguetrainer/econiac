# What is money a claim on?

*Double-entry accounting says every liability has a matching asset. But when
a central bank creates reserves under fiat money, the liability is redeemable
in... more of the same liability. This is not a bug in the accounting. It is a
deep fact about what money is — and it has consequences for how EconIAC models
monetary systems.*

---

## The three monetary regimes

Money has existed in three structurally distinct regimes, each corresponding
to a different answer to the question "what is this liability ultimately a
claim on?":

### 1. Commodity money (gold standard)

The central bank holds gold. Banknotes are claims on that gold. The liability
is redeemable in a physical asset that exists independently of the monetary
system.

**∂²=0 holds**: the balance sheet balances, and the liability has external
substance. The constraint on money creation is real: you cannot issue more
notes than you have gold.

**Gauge-theory framing**: the unit of account is *fixed* to the commodity. The
connection has a preferred gauge — gold — and all other prices are measured
relative to it. Abandoning the gold standard is a gauge transformation: the
unit of account floats free, and the "zero of potential" is no longer pinned.

### 2. Credit money (commercial banking)

A commercial bank makes a loan. It simultaneously creates:
- **Asset**: the loan receivable (a claim on the borrower)
- **Liability**: a deposit in the borrower's account (a claim on the bank)

The deposit is redeemable in central bank reserves, which are redeemable in
banknotes, which are claims on the central bank. The chain terminates at the
central bank — but as long as the bank is solvent, the deposit is backed by
real assets (the loan portfolio).

**∂²=0 holds throughout**: every new claim has a matching counter-claim.
Money is created endogenously by the banking system, not exogenously by the
central bank. This is the core insight of post-Keynesian monetary theory
(Minsky \cite{minsky1986}, Keen \cite{keen2011}, Wray \cite{wray2012}).

**Gauge-theory framing**: credit money floats relative to the central bank
liability. The "gauge group" is the set of credit relationships. Defaults
(loan impairment) are gauge-field singularities — points where the connection
breaks down and the holonomy becomes non-trivial.

### 3. Fiat money (modern central banking)

The central bank creates reserves by purchasing an asset (a government bond,
a mortgage-backed security, a foreign currency). The accounting is:

| Central bank asset | Central bank liability |
| --- | --- |
| Government bond | Reserves credited to commercial banks |

The reserves are claims on the central bank. But what is the central bank's
liability redeemable *in*? In more reserves — or in banknotes, which are
themselves central bank liabilities. The chain is circular.

**∂²=0 still holds**: the balance sheet always balances. But the *substance*
of the liability is different from any private-sector liability. It is not
backed by a commodity (no gold) and not backed by a specific loan portfolio.
It is backed by:

1. **The taxing power of the state**: the government accepts its own
   currency in payment of taxes, which creates demand for the currency
   (Chartalism / MMT — see Wray \cite{wray2012}).
2. **Legal tender laws**: the state mandates acceptance of the currency
   in settlement of debts.
3. **The institutional topology of the monetary system as a whole**: the
   currency functions as long as the network of agents that accepts it
   continues to accept it.

The third point is the deepest. Under fiat money, the "backing" of the
currency is a topological property of the monetary network — not a
balance-sheet entry.

---

## The gauge-theory formalisation

EconIAC's Financial Gauge Theory (Paper 295, \cite{buckley2026_295}) treats
the unit of account as a **connection** on the Pacioli manifold.

In gauge theory, a connection describes how to compare quantities at different
points of a manifold. The curvature of the connection measures
path-dependence — whether the result of a sequence of transactions depends on
the order in which they are executed.

The three monetary regimes correspond to three types of connection:

| Regime | Connection type | Curvature | What "zero" means |
| --- | --- | --- | --- |
| Commodity money | Flat (zero curvature) | $F = 0$ | Pinned to gold |
| Credit money | Mildly curved | $F \neq 0$ locally | Pinned to central bank |
| Fiat money | Gauge-free | $F \neq 0$ globally | No external anchor |

**Switching between regimes is a gauge transformation**: it changes the
reference point for the unit of account without changing the real economic
relationships. This is why models calibrated under a gold standard give
different answers from models calibrated under fiat — not because the
underlying economics changed, but because the gauge changed.

EconIAC makes this explicit: Paper 295 proves that the apparent "anomalies"
of covered interest parity failure, FX basis persistence, and cross-currency
swap spreads are all **curvature effects** — consequences of operating in a
curved monetary manifold rather than the flat commodity-money ideal.

---

## Why ∂²=0 is stronger than "money is conserved in quantity"

A common misstatement of the Pacioli identity is "money cannot be created or
destroyed." This is wrong, and importantly wrong.

Under credit money, commercial banks create money every time they make a loan.
Under fiat money, central banks create reserves whenever they purchase assets.
Money creation is not prohibited — it is happening continuously.

What ∂²=0 says is subtler and more powerful:

> **Every new financial claim must be accompanied by a matching counter-claim.**

Money can be created, but only symmetrically. A bank that creates a deposit
without creating a matching loan receivable has violated ∂²=0 — not by
creating money, but by creating an *unmatched* claim. This is what fraud
looks like in EconIAC's type system: a flow that has no counter-flow, a
liability with no asset, a transaction that does not balance.

The conservation law is about **form**, not **quantity**. It constrains the
topology of creation, not the amount. This is why it is a genuinely different
kind of law from physical conservation laws (energy, charge, momentum) — those
do constrain quantity. ∂²=0 constrains structure.

---

## Implications for modelling

### Endogenous money is the default

EconIAC's SFC models (`econiac.economics.sfc`) treat money creation as
endogenous by default: loans create deposits, deposits fund spending, spending
generates income. The money supply is a consequence of credit decisions, not
a policy variable set by the central bank. This follows Godley and Lavoie
\cite{godley2007} directly.

The central bank controls the *price* of money (the interest rate) and
provides an elastic supply of reserves on demand — it does not control the
*quantity* of money in circulation. Quantity-theoretic models (MV=PQ) are
a special case that requires additional assumptions EconIAC does not make
by default.

### Different regimes need different gauge fixings

A model of the gold-standard era needs a flat connection (zero curvature).
A model of the post-2008 QE era needs a curved connection reflecting the
excess reserves created by asset purchases. Fitting the same structural model
to both periods without changing the gauge will produce spurious parameter
instability — the model will attribute to behavioural change what is actually
a change in monetary regime.

EconIAC's `CurvedBalanceSheet` and the curvature module of Paper 295 provide
the language to make this explicit.

### The central bank's liability is topological

Under fiat money, the central bank's reserve liability is not backed by any
specific asset in the way a commercial bank deposit is. Its "backing" is the
continued functioning of the monetary network — a topological property.

This has a concrete modelling consequence: the failure of a central bank (as
distinct from a commercial bank) is not a balance-sheet event. It is a
topological event — the collapse of the network of agents that accepts the
currency. EconIAC's sheaf H¹ signal (Paper 335, \cite{buckley2026_335}) is
the natural instrument for detecting this: $H^1 \neq 0$ on the monetary
network indicates that local acceptances of the currency can no longer be
globally reconciled — the precursor to a currency crisis, not merely a
solvency crisis.

---

## Further reading

- Minsky, H.P. (1986). *Stabilizing an Unstable Economy*. Yale University Press.
- Keen, S. (2011). *Debunking Economics*. Zed Books. Ch. 13 (endogenous money).
- Wray, L.R. (2012). *Modern Money Theory*. Palgrave Macmillan.
- Godley, W. & Lavoie, M. (2007). *Monetary Economics*. Palgrave Macmillan.
- Buckley (2026). Paper 291: The Topology of Conservation.
  doi:10.5281/zenodo.20234853
- Buckley (2026). Paper 295: Currency Bundles on the Pacioli Manifold.
  doi:10.5281/zenodo.20242355
- Buckley (2026). Paper 335: Topological Inconsistency. doi:TBD

---

## See also

- [Why topology?](topology.md) — ∂²=0 as the foundational identity
- [Why connections and curvature?](curvature.md) — non-flat connections and
  what it means for a monetary system to be curved
- [Why gauge theory?](gauge_theory.md) — the unit of account as a gauge choice
