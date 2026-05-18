# Why does econiac use non-associative algebra?

*Policy interventions don't commute. Raise rates then regulate banks is not the same as regulate banks then raise rates. Standard causal models assume otherwise.*

---

## The hidden assumption in every causal model

The modern toolkit for causal inference — Pearl's do-calculus, directed acyclic graphs (DAGs), potential outcomes — has transformed empirical economics. But every one of these frameworks rests on a hidden assumption so natural that it is almost never stated: **interventions associate**.

What this means: if you apply policy A, then B, then C, the result is the same regardless of how you bracket the computation: (A then B) then C = A then (B then C). The order matters (A then B ≠ B then A in general), but the *grouping* does not.

This is the **associativity axiom**. It holds in standard algebra (addition, multiplication), and it is inherited by any framework built on standard algebra — including DAGs, linear models, and most structural equation models.

In real economies, it frequently fails.

## When associativity fails

Consider three interventions in sequence:
- **A**: the central bank raises interest rates
- **B**: the government tightens bank capital requirements
- **C**: a major corporate borrower defaults

Apply them in order A, B, C. The sequence (A then B) then C is: first the rate rise tightens credit conditions; then the capital requirement forces banks to reduce lending further; then the default hits a banking system that is already stretched. The result is severe credit contraction.

Now apply them as A then (B then C): first the rate rise; then the capital requirement combined with the default. The order of B and C relative to each other within the second step is different. The banks face the default *before* fully adjusting to the capital requirement. The result — which lenders are exposed, how the loss is distributed, what the fire-sale dynamics look like — is quantitatively different.

Same three interventions, same order A→B→C, but different bracketing produces different outcomes. This is non-associativity.

It is not exotic. It arises whenever:
- An intervention changes the *vulnerability* of the system to subsequent interventions (sequencing effects in stress tests)
- Multiple policies interact through a shared constraint (balance sheet capacity, regulatory capital, market liquidity)
- The transmission mechanism itself is state-dependent (monetary policy at the zero lower bound vs. in normal times)

## What standard models do instead

Standard models handle this by assuming away the non-associativity — either explicitly (by restricting attention to linear or additive policy effects) or implicitly (by not tracking the state-dependence of the transmission mechanism).

The consequence: policy interaction effects are systematically underestimated. Stress tests that treat each risk factor as independent miss the amplification that occurs when factors interact through a shared constraint. Models that estimate fiscal multipliers from normal-times data misapply them at the zero lower bound.

## The mathematics econiac uses

econiac models policy interventions using **magmas** — algebraic structures with a binary operation that is not required to be associative. The economy's response to a sequence of interventions is a path in a magma: the result depends on both the order and the bracketing.

The **associator** A(x, y, z) = (x·y)·z − x·(y·z) measures the failure of associativity for a specific triple of interventions. In econiac, the associator is a computable quantity: given a model of how interventions interact, you can compute the associator for any triple and identify which combinations are most sensitive to sequencing.

This is the same mathematics that underlies the exceptional Lie algebra G₂ and the octonions — which is not a coincidence. Non-associativity in algebra and non-commutativity in policy are both manifestations of the same underlying structure: operations that depend on context, not just on content.

## The practical payoff

For most policy analysis, the associator is small and the standard associative approximation is adequate. econiac does not require you to use non-associative algebra for everything — only for the cases where the sequencing and bracketing of interventions genuinely matters.

Those cases are precisely the ones that standard models miss: financial crises (where amplification mechanisms make the bracketing decisive), climate transition risk (where the sequence of regulatory changes and market adjustments determines whether the transition is orderly or disorderly), and monetary-fiscal interaction (where the order of commitment matters for credibility).

econiac computes the associator automatically. If it is small, the standard approximation is valid and you can proceed as normal. If it is large, econiac flags the interaction effect — and can compute the gradient of the associator with respect to policy parameters, telling you which parameter changes would most reduce the sequencing sensitivity.

## The short version

You need non-associative algebra because:

1. **Real policy interventions are not associative.** (A then B) then C ≠ A then (B then C) whenever the transmission mechanism is state-dependent.
2. **Standard causal models assume associativity.** DAGs, linear models, and structural equation models all encode this assumption silently.
3. **The associator is computable.** econiac calculates how much the bracketing matters for any triple of interventions — and flags it when the standard approximation breaks down.
4. **The cases where it matters most are the cases you care about most:** crises, transitions, and regime changes.
