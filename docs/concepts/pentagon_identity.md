# The Pentagon Identity

> *"There is a remarkably simple test for whether a system of local prices
> is globally consistent. It requires no model, no parameters, and no
> assumptions about distributions. It is a boundary condition — and it
> either holds or it does not."*

The Pentagon identity is the mathematical reason that $H^2$ risk is
definable at all, and the reason that a single pricing model can never
generate a systemic risk obstruction by itself.

---

## The simplest version: double-entry bookkeeping

Double-entry bookkeeping enforces a boundary condition: every debit has
a matching credit. In mathematical notation, if $\partial$ is the boundary
operator on the balance sheet complex, then

$$\partial^2 = 0$$

Apply $\partial$ once: a transaction creates a signed sum of balance-sheet
entries. Apply $\partial$ again: the result is always zero, because every
entry on a balance sheet has a counter-entry that cancels it. This is not
an accounting convention — it is a theorem about the structure of the
complex.

The Pentagon identity is the same statement, one dimension higher.

---

## The financial Pentagon identity

Take four institutions $A, B, C, D$. Each pair of triples among them —
$(A,B,C)$, $(A,B,D)$, $(A,C,D)$, $(B,C,D)$ — carries a residual $c$:
a measure of how far that triangle's pricing is from perfectly consistent.
In the correlation market, $c_{ABC}$ might be the residual correlation
implied by options on all three pairs versus the basket option price.

The Pentagon identity says:

$$c_{BCD} - c_{ACD} + c_{ABD} - c_{ABC} = 0$$

This is $\delta^2 \circ \delta^1 = 0$: the coboundary of a coboundary
vanishes. If all four residuals are computed from a **single consistent
pricing model**, this identity holds automatically — it is a tautology,
with zero empirical content.

**This is why a single-source model can never generate $H^2$ risk.**
No matter how complex the model, no matter how many parameters it has,
if all four triangles are priced from the same source, the alternating
sum is identically zero. The identity is not a constraint on the model —
it is a consequence of using one model.

---

## When the identity fails

The identity can fail only when the four triangular residuals come from
**more than one source** — different desks, different models, different
jurisdictions — that each produce internally consistent triangles but
whose residuals do not close around the tetrahedron.

A numerical example. Three faces priced from one correlation model:

```text
c_ABC = 0.012
c_ABD = 0.031
c_ACD = 0.043
```

Consistency forces the fourth:

```text
c_BCD = c_ACD − c_ABD + c_ABC = 0.043 − 0.031 + 0.012 = 0.024
```

Now suppose the BCD desk uses a different correlation assumption and
reports $c_{BCD} = 0.042$. Each triangle looks fine in isolation. But:

```text
c_BCD − c_ACD + c_ABD − c_ABC = 0.042 − 0.043 + 0.031 − 0.012 = 0.018 ≠ 0
```

The Pentagon identity fails. $H^2 \neq 0$. The system has a topological
obstruction that is invisible at every triangle taken alone.

This is the structure of the 2008 correlation desk failure. No single
desk's triangle looked wrong. The four desks' independently-priced,
overlapping correlation triangles did not close.

---

## Why it is called the Pentagon identity

The name comes from two related but distinct sources, both pointing at
the same underlying idea.

**In simplicial topology**, $\delta^2 \circ \delta^1 = 0$ is a standard
identity for the Čech coboundary operator. "Pentagon" refers to the
five-term structure of the alternating sum when written out fully for a
tetrahedron — four face terms with alternating signs, equalling zero.

**In category theory**, Mac Lane's coherence pentagon is the five-morphism
diagram that must commute for a monoidal category to be well-defined:

```text
((A⊗B)⊗C)⊗D ——————————————————→ A⊗(B⊗(C⊗D))
      |                                  ↑
      ↓                                  |
(A⊗(B⊗C))⊗D ————→ A⊗((B⊗C)⊗D) ————→ ...
```

Mac Lane's pentagon says: "two different ways of re-bracketing a
four-fold tensor product give the same result." The financial Pentagon
identity says: "two different ways of assembling four triangular
residuals into a tetrahedral boundary give the same result — namely,
zero." Both are instances of the same demand: **composing two layers of
consistency requirements must itself be consistent**.

The connection is not merely analogical. In the language of
[the Origami ISA](../why/origami_isa.md), the Pentagon identity is the
coherence condition that makes the ISA well-defined as a monoidal
structure on the category of representations. For the purely financial
(Abelian, commutative) applications of EconIAC, the categorical machinery
is not needed — the simplicial form $\delta^2 \circ \delta^1 = 0$ is
sufficient and self-contained.

---

## What the identity tells us about risk

| Situation | Pentagon identity | Meaning |
| --- | --- | --- |
| Single pricing model, four triangles | Holds automatically | $H^2 = 0$; no systemic obstruction possible |
| Multiple desks, internally consistent | May fail | $H^2 \neq 0$; systemic obstruction present |
| All bilateral exposures resolved | Does not help | H² is invisible to bilateral instruments |
| CCP novation of all triangles | Fills triangles, $H^1 \to 0$ | But does not address $H^2$ if already present |
| Government intervention | Acts at tetrahedron level | Only instrument that can resolve $H^2 \neq 0$ |

---

## The identity as a diagnostic

Because the Pentagon identity is a simple linear equation, it can be
evaluated from market data without any model:

1. Observe the H¹ residuals $c_{ABC}$, $c_{ABD}$, $c_{ACD}$, $c_{BCD}$
   from prices of triangular instruments (basket options, CDX tranches,
   correlation swaps) on all four triples.
2. Compute the alternating sum.
3. If it is non-zero, $H^2 \neq 0$: the system has a topological
   obstruction that no bilateral or triangular instrument can resolve.

This is the computational core of EconIAC's cohomological stress test
([Paper 397](https://doi.org/10.5281/zenodo.20642908)). The test
requires only prices of liquid correlation instruments — available in
real time from trade repositories and listed markets. No model. No
parameters. No assumptions about distributions.

**The Pentagon identity either holds or it does not. When it stops
holding, the system is structurally fragile — regardless of whether any
individual institution has breached a threshold.**

---

## Further reading

- Buckley (2026). Paper 396: The Unhedgeability Theorem.
  doi:[10.5281/zenodo.20635479](https://doi.org/10.5281/zenodo.20635479)
- Buckley (2026). Paper 397: Systemic Risk as H².
  doi:[10.5281/zenodo.20642908](https://doi.org/10.5281/zenodo.20642908)
- Buckley (2026). Paper 398: The Topology of Risk (Primer).
  doi:[10.5281/zenodo.20642983](https://doi.org/10.5281/zenodo.20642983)
- Mac Lane, S. (1963). Natural associativity and commutativity.
  *Rice University Studies* 49(4), 28–46. (Original Pentagon coherence paper.)
- Weibel, C. (1994). *An Introduction to Homological Algebra*. Cambridge.
  (Standard reference for $\delta^2 = 0$ in all its forms.)
