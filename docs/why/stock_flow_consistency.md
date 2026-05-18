# Why does Econiac use stock-flow consistency?

*Every pound that exists was created somewhere and is owed to someone. Models that forget this are wrong in ways that only show up in crises.*

---

## What stock-flow consistency means

Stock-flow consistency (SFC) is the requirement that every financial stock — every asset, every liability, every balance sheet position — must be traceable to the flows that created it, and every flow must have a corresponding entry on both sides of every affected balance sheet.

In practice this means three things:

1. **Every asset is someone else's liability.** If households hold £1 trillion in bank deposits, then banks owe £1 trillion to households. There is no "net wealth" created by financial transactions — only redistribution.

2. **Flows integrate to stocks.** If the household sector saves £50bn per year, its net financial wealth increases by £50bn per year (plus revaluations). A model where this accounting does not hold is describing an economy where money appears from or disappears into nowhere.

3. **Sectoral balances sum to zero.** Household surplus + firm surplus + government surplus + external surplus = 0. Always. By construction. If your model has four sectors and three of them are in surplus, the fourth must be in deficit — no exceptions.

## Why standard models violate this

Dynamic Stochastic General Equilibrium (DSGE) models — the workhorse of central bank macroeconomics since the 1990s — routinely violate stock-flow consistency. The violations are usually not deliberate; they are a consequence of building models by writing down equilibrium conditions rather than balance sheets.

The most common failure: the government budget constraint is specified as a flow equation (spending - taxes = deficit) without a corresponding stock equation (what happens to the accumulated debt?). In a one-period model this does not matter. In a multi-period model, the accumulated debt must be held by someone — and that someone's portfolio must balance. If the model does not track this, it is implicitly assuming an infinitely patient agent who absorbs unlimited net liabilities with no feedback. This is not a modelling simplification. It is a mistake.

Godley (1999) identified this class of errors and showed that several widely-used models of the 1990s implicitly required the household sector to accumulate unlimited net debt — a trajectory that was unsustainable and which the models' internal logic could not detect. The 2008 crisis was, in part, the real-world arrival of the constraint the models had been ignoring.

## The Godley-Lavoie approach

Wynne Godley and Marc Lavoie (2007) developed a systematic framework for building stock-flow consistent macroeconomic models. The core tool is the **Godley table** — a matrix where rows are sectors (households, firms, banks, government, rest of world) and columns are financial instruments (deposits, loans, bonds, equity). Each entry records the flow of that instrument between that sector and the others in a given period.

The Godley table enforces stock-flow consistency by construction: each column sums to zero (every asset has a liability), and each row tracks the net financial position of that sector across all instruments.

This is exactly the Pacioli manifold: the Godley table is the incidence matrix of the directed graph, and the column-sum-to-zero condition is ∂²=0. econiac implements Godley tables as Pacioli manifold instances, so every model built in econiac is stock-flow consistent by construction — not by careful bookkeeping, but by the type system.

## What SFC gives you that DSGE does not

**Crisis detection**: SFC models track the accumulation of financial positions across sectors. An unsustainable trajectory — households accumulating debt faster than income growth, or the external sector running a persistent deficit — shows up as a diverging stock position before it shows up as a crisis. DSGE models, which often close with a transversality condition rather than an explicit balance sheet, cannot detect this.

**Sectoral balance accounting**: the national accounts identity (household surplus + government surplus + firm surplus + external surplus = 0) is automatically satisfied in every period in an SFC model. This constrains the model's dynamics in ways that are empirically verifiable against the national accounts.

**Genuine monetary analysis**: in an SFC model, money is created when banks make loans (a new asset and a new liability appear simultaneously) and destroyed when loans are repaid. This is how money actually works. Most DSGE models treat money as an exogenous quantity that the central bank controls directly — a description that has not been accurate since the abandonment of monetary targeting in the 1990s.

## The short version

You need stock-flow consistency because:

1. **Every pound exists on two balance sheets.** Models that do not track this are assuming money appears from nowhere — and will fail under any scenario where the missing constraint becomes binding.
2. **Crises are stock problems, not flow problems.** They occur when accumulated imbalances become unsustainable. SFC models detect this; flow-only models do not.
3. **econiac enforces SFC by construction.** The Pacioli manifold's defining property ∂²=0 is stock-flow consistency stated as a topological theorem. You cannot build a non-SFC model in econiac — it would be a type error.
