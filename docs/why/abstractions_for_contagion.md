# Why does Econiac use sheaves, Gibbs lifts, and duals for contagion?

*You are an economist or financial regulator. You have read Gorton (2012) and
Eisenberg-Noe (2001). You want to know why the contagion library uses words like
"sheaf Laplacian", "Gibbs lift", and "adjoint operator" instead of just writing
down the fire-sale equations directly.*

The short answer: each piece of abstract mathematics solves a specific problem
that the standard toolkit cannot.

> **Note on scope**: Sheaves appear in several places in Econiac — contagion
> networks (this document), sheaf neural networks, and topological signal
> processing. This document focuses on the contagion use case. See
> [Why topology?](topology.md) for the general sheaf picture and the
> connection to Pacioli ∂²=0.

---

## 1. Sheaves — why do we need them for contagion specifically?

### The problem

When a repo market is under stress, a regulator wants to ask:

> **"Do different parts of the network agree on which dealers are solvent?"**

If MMFs think dealer A is fine but LDI pension funds think dealer A is in trouble,
and both are acting on that belief simultaneously — MMFs rolling their repo while
LDI funds are withdrawing — the system is *inconsistent*. This inconsistency is
not visible in any single balance sheet. It is a network-level phenomenon.

Standard stress tests measure whether individual banks breach individual thresholds.
They cannot detect network-level disagreement, because that disagreement lives
*between* the nodes, not at them.

### What a sheaf adds to the balance sheet picture

The [topology document](topology.md) explains that double-entry accounting is ∂²=0
and that the Pacioli manifold is the natural home for balance sheet dynamics.

What a sheaf adds is **restriction maps** — linear maps that say how much the
information at one node should agree with the information at an adjacent node,
weighted by the bilateral exposure between them. In our contagion context:

- **Stalk** at dealer j: funding ratio fⱼ = rolled funding / total repo out
- **Restriction map** on edge (lender i → dealer j): F_{i,e} = exposure_ij / assets_i
- **Consistency**: lender i's view of dealer j's solvency should agree with
  dealer j's actual funding ratio, weighted by their bilateral exposure

The **sheaf Laplacian** L_F measures the total inconsistency across all edges:

    H¹ signal = ‖L_F · s‖² / ‖s‖²

where s is the vector of health ratios. Zero = globally consistent. Large =
agents disagree across the network in a way that cannot be reconciled.

### Why it leads the cascade

The H¹ signal rises *before* any individual threshold is breached, because:

1. Inconsistency in assessments creates funding pressure (counterparties price
   in the risk they perceive, even before the default event)
2. Funding pressure erodes funding ratios
3. Threshold breach is the observable event we label "distress"

In the CHZ fire-sale model (Paper 332, experiment x332e, 20 random seeds):
H¹ peaks 2-3 periods before the distress count peaks.

### The three-way isomorphism

The same sheaf Laplacian construction appears in three different models in Econiac:

| Model                           | Graph                   | Section              | H1 signal           |
| ------------------------------- | ----------------------- | -------------------- | ------------------- |
| CHZ fire sales (Paper 332)      | Interbank exposure      | Capital ratio        | norm(L_F*g)/norm(g) |
| Sovereign repo run (Paper 333)  | Dealer-lender bipartite | Funding ratio        | norm(L_F*f)/norm(f) |
| FMO energy transfer (Paper 325) | Fano chromophore graph  | Energy efficiency    | norm(L_F*e)/norm(e) |

These are not analogies. The computation is identical on three different graphs.
Paper 334 §6 proves the structural isomorphism.

---

## 2. Lifts — what is a Gibbs lift and why do we need it?

### The hard-threshold problem

Gorton and Metrick (2012) model repo runs with a hard binary rule:

- Haircut exceeds threshold → lender withdraws all funding immediately
- Haircut within threshold → lender rolls all funding without question

This is economically wrong (LDI pension funds do not behave like a light switch)
and mathematically crippling: a hard cliff is not differentiable. Without a
derivative, you cannot compute the gradient of systemic loss with respect to the
haircut schedule. Without that gradient, you cannot do optimal policy — you can
only enumerate scenarios and compare them by hand.

### What a lift is

In mathematics, a **lift** takes a map that lives in a simpler world and finds
a corresponding map in a richer world that "projects back" to the original.

The classic example from topology: a path on the circle S¹ can be lifted to a
path on the real line ℝ — the covering space — where angles become unbounded
real numbers. The lift preserves the structure of the path but lives in a richer
space where more tools are available.

Our **Gibbs lift** takes the hard binary function (living in the world of
discontinuous 0/1 maps) and lifts it into the world of smooth functions:

    hard:    1[coverage > threshold]
    lifted:  σ(β × (coverage − threshold))     [sigmoid, logistic function]

The lifted function has three properties:

- Equals the hard function in the limit β → ∞
- Is smooth and differentiable at every finite β
- Has a direct economic interpretation: β measures how sharply lenders react

This is called a **Gibbs lift** because the sigmoid is the one-variable special
case of the Gibbs distribution from statistical physics. High β (cold system)
= sharp reaction; low β (hot system) = smooth, gradual response.

For lenders, calibrated β values are:

- β_mmf = 500: MMF regulatory constraint — near cliff-edge
- β_ldi = 20: LDI pension fund — gradual VaR-triggered de-leveraging
- β_hf = 5: Hedge fund — continuous risk-appetite repricing
- β → ∞: exact Gorton hard rule

### What the lift enables

Once operators are Gibbs-lifted, JAX autodiff gives ∂L/∂(policy parameter)
through the entire cascade in one backward pass:

- ∂L/∂h_ij: policy gradient on the bilateral haircut schedule
- ∂L/∂β: how fragility changes with constraint sharpness
- The (β, h_baseline) phase diagram: where the system transitions from stable to run

None of these exist in Gorton's hard-threshold model. The lift creates them.

---

## 3. Duals — why does every operator have an adjoint?

### The AL symmetry puzzle

Hurd (2017) proved that the solvency cascade has an exact dual — the liquidity
cascade — under the substitution A↔D, Z↔X, C↔E. He called this "AL symmetry".
Is this a coincidence of the Eisenberg-Noe model, or a universal property?

### What an adjoint is

For any linear map f, its **adjoint** f† is the unique map satisfying:

    ⟨f(x), y⟩ = ⟨x, f†(y)⟩

If f measures "how x influences y", then f† measures "how y influences x" —
the dual direction.

The inner product ⟨·,·⟩ used here is the **Pacioli inner product** — the
natural inner product on SystemState that respects ∂²=0.

### Why AL symmetry is structural

Under the Pacioli inner product, S† = L and L† = S. This is not specific to
Eisenberg-Noe — it follows from ∂²=0 alone. Any model that satisfies Pacioli's
identity has this solvency-liquidity duality.

**Library**: `wire_al_symmetry(S, L)` in `primitives.py` enforces this
structurally — the type system, not a proof obligation on each new model.

---

## 4. What mathematical tools was Hurd (2017) exploiting?

| Tool                            | Economic problem                                                              | Library                        |
| ------------------------------- | ----------------------------------------------------------------------------- | ------------------------------ |
| Lattice theory (Knaster-Tarski) | Who defaults depends on who defaults — circular. Fixed point resolves it.     | `fixed_point(op, upper_bound)` |
| Pacioli ∂²=0                    | Every asset is someone else's liability; models that violate this are wrong.  | `pacioli_check()`              |
| Operator composition            | Fire sales, funding withdrawals, price drops compound each other.             | `compose(L_A, S_D)`            |
| Adjoint functors (AL symmetry)  | Solvency-liquidity duality is universal for stock-flow consistent networks.   | `wire_al_symmetry(S, L)`       |

---

## What you do NOT need to know

These appear in the proofs but not in the code:

- **Natural transformations**: The Gibbs lift is one; you only need `gibbs_lift(op, beta)`.
- **Spectral sequences**: Not used. H¹ signal = eigenvalues of L_F (linear algebra).
- **Infinity-categories, derived categories, topos theory**: Not used here.

---

## Summary

| Concept                    | Economic meaning                              | Entry point                          |
| -------------------------- | --------------------------------------------- | ------------------------------------ |
| Sheaf                      | Local assessments that may disagree globally  | `FinancialGraph`, `sheaf_laplacian`  |
| H1 cohomology              | Measure of network-level inconsistency        | `h1_signal`, `sheaf_h1_signal`       |
| Gibbs lift                 | Smooth differentiable threshold               | `gibbs_lift(op, beta)`               |
| Beta (inverse temperature) | Sharpness of lender reaction function         | `GibbsParams`                        |
| Adjoint / dual             | Solvency-liquidity symmetry                   | `wire_al_symmetry(S, L)`             |
| Pacioli ∂²=0               | Double-entry accounting identity              | `pacioli_check()`                    |
| Knaster-Tarski fixed point | Cascade converges to max-recovery equilibrium | `fixed_point(op, upper_bound)`       |
| Operator composition       | Contagion channels compound                   | `compose(L_A, S_D)`                  |
| Policy gradient            | dL/dh for optimal haircut policy              | `policy_gradient()`, `ldi_surcharge` |
