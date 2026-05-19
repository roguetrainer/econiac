# Why does EconIAC use a combinator library?

*You are a quant or economist. You already have Python. Why do you need a DSL?*

---

## The circuit analogy

In 1845, Gustav Kirchhoff stated two laws that every electrical engineer knows:

- **Kirchhoff's Current Law (KCL)**: at every junction, current in = current out.
- **Kirchhoff's Voltage Law (KVL)**: around every closed loop, voltage drops sum to zero.

KCL is a conservation law. It is necessary — a circuit that violates it is broken — but it tells you almost nothing about what the circuit *does*. A resistor, a capacitor, a transistor, and a battery all obey KCL. What distinguishes them is how they *transform* the signal that passes through them, and how they *compose* with each other.

The Pacioli Combinator Library (PCL) stands in exactly the same relation to double-entry bookkeeping. The conservation law ∂²=0 — every asset is someone else's liability, column sums of the balance sheet equal zero — is PCL's KCL. It is the substrate. What PCL adds is a vocabulary of **composable financial operations**: primitives that transform balance sheets, and combination rules that wire those primitives into arbitrarily complex financial instruments.

The question "why use a combinator library?" is the same as asking "why do electronics engineers use circuit schematics instead of writing Maxwell's equations from scratch every time?" Because composition is how you build complex things from simple parts without losing track of what the parts do.

---

## The electronic components

Every PCL computation is a function from one balance sheet to another. The primitive operations correspond to electronic components:

| Electronic component | What it does | PCL primitive |
|---|---|---|
| Wire | Passes signal unchanged | `identity()` |
| Ground | Zeroes the signal | `zero()` |
| Amplifier / attenuator | Scales the signal | `scale(α, f)` |
| Current source | Injects a fixed current at a junction | `flow(from, to, instrument, amount)` |
| Series connection | Signal passes through A then B | `sequence(f, g)` |
| Parallel connection | Signal splits, both paths active | `parallel(f, g)` |
| Transistor | β-controlled switch between two paths | `choose(β, f, g)` |
| Multiplexer | β-weighted selection among N paths | `fold(β, [f₁,…,fₙ])` |
| Oscillator / clock | Applies the same operation N times | `repeat(n, f)` |
| SPICE netlist | The full circuit specification | A composed `Computation` tree |
| Compiler | Translates netlist to silicon | `compile(computation)` |

The `flow` primitive is the double-entry atom: move `amount` of instrument `X` from sector A to sector B. The debit equals the credit; KCL is satisfied by construction. Everything else is composition.

---

## Why the transistor is the interesting part

A resistor and a capacitor are passive: they transform the signal but do not make decisions. The transistor is active: it uses one signal (the gate voltage) to control the routing of another signal (the source-drain current). This is what makes computation possible — without the transistor, a circuit can only filter; with it, a circuit can switch, amplify, and ultimately implement any Boolean function.

PCL's transistor is `choose(β, f, g)`:

```python
choose(β, f, g)(balance_sheet)
```

This evaluates both `f` and `g` on the input balance sheet, then routes weight between them according to the Gibbs distribution at inverse temperature β:

- **β = 0**: equal weight to both paths. The system explores both financial strategies simultaneously. This is the analogue limit — a linear mixer.
- **β → ∞**: all weight to whichever path produces higher value. The system commits to the better strategy. This is the digital limit — a hard switch.
- **β finite**: a smooth interpolation. The system hedges between strategies in proportion to their relative value, with the degree of hedging controlled by β.

This is not metaphor. At β → ∞, `choose` recovers the tropical (max, +) semiring exactly — the mathematical definition of digital switching. At finite β it is the Gibbs ensemble average. The Maslov-Gibbs Einsum (MGE) is precisely this interpolation, implemented as a tensor contraction.

The consequence: a PCL computation parameterised by β is **differentiable in β**. You can ask: "how does the output change if I make the system more decisive?" The answer is a gradient. You can calibrate β to data by backpropagation through the entire composed computation — including through every `choose` and `fold` gate in the circuit.

---

## Series vs parallel: order matters

In electronics, series and parallel have different meanings. A series RC circuit and a parallel RC circuit behave completely differently, even though both contain the same components.

PCL's `sequence` and `parallel` have the same distinction:

**`sequence(f, g)`** applies `f` first, then `g` to the result. The output of `f` is the input to `g`. This is the natural model for temporal financial processes: wages are paid (f), then taxes are remitted (g), then dividends are distributed (h). The order matters — in a non-Abelian gauge theory, `sequence(f, g) ≠ sequence(g, f)` in general, just as the order of gauge transformations on a fibre bundle is non-commutative.

**`parallel(f, g)`** applies both `f` and `g` to the same input and adds their effects. This models simultaneous, independent transactions: the same shock hits the solvency channel and the liquidity channel at the same time. Both cascades run in parallel; their effects compound.

The non-commutativity of `sequence` is not a quirk — it is the mathematical content of Papers 291 and 294. Policy interventions compose non-commutatively: a debt restructuring followed by a tax reform is not the same economy as the same two policies in reverse order. PCL makes this non-commutativity explicit and computable.

---

## The type system is Kirchhoff

Every PCL combinator is guaranteed — by construction — to preserve ∂²=0. This is the type system. A `flow(A, B, X, amount)` simultaneously debits A and credits B by the same amount; the conservation law cannot be violated. A `sequence(f, g)` of two conserving operations is conserving. A `choose(β, f, g)` of two conserving operations is conserving, because a Gibbs-weighted average of zero-column-sum matrices has zero column sums.

The `typecheck` function provides a runtime assertion:

```python
assert typecheck(my_computation)   # raises if ∂²=0 is violated
```

In electronic terms: the type system is the rule that no node can accumulate charge. A circuit that violates this rule has a short circuit or a floating node — it is not a valid circuit. A PCL computation that violates ∂²=0 has an unbalanced book — it is not a valid financial model.

The practical consequence: you cannot accidentally write a PCL program that creates money from nothing or destroys it. Conservation is enforced structurally, not by testing.

---

## Compilation to silicon

A circuit schematic is not a circuit. To get actual behaviour, you need to fabricate the schematic — translate it from a description to a physical implementation.

PCL's `compile(computation)` does the same thing. It takes a composed `Computation` tree and returns a JAX-jit-compiled version: the full computation is traced through XLA, optimised, and compiled to run on CPU, GPU, or TPU. After compilation, the computation executes at hardware speed with no Python overhead.

```python
quarterly_report = sequence(
    fold(β, [wages_flow, interest_flow, tax_flow]),
    choose(β, reinvest, distribute),
)
fast_report = compile(quarterly_report)   # compile once
results = [fast_report(balance_sheet_t) for t in range(1000)]   # run fast
```

The compiled computation is differentiable end-to-end. Backpropagation flows through every `choose`, `fold`, `sequence`, and `scale` gate. This is what makes calibration possible: run the compiled circuit forward on historical data, compare to observed outcomes, backpropagate the error, and update parameters — including β — by gradient descent.

---

## What PCL is not

PCL is not an accounting system. It does not produce financial statements, check regulatory compliance, or interface with ERP software. It is a mathematical substrate for building differentiable financial models — the same way a hardware description language (VHDL, Verilog) is not a product, but the language in which products are described.

PCL is not a simulation engine. It does not advance time autonomously, manage agent states, or handle events. Time is advanced by the caller: `repeat(n, step)` applies one step `n` times, but the caller decides what `n` is and what to do with the trajectory.

PCL is not a replacement for Python. Complex models will use PCL primitives inside ordinary Python logic — conditionals, loops, data loading. PCL provides the composable financial operations; Python provides the orchestration.

---

## The short version

You need a combinator library because:

1. **Conservation is the substrate, not the point.** ∂²=0 is KCL for money. It rules out broken models but does not tell you what any model does. The combinators — `flow`, `sequence`, `parallel`, `choose`, `fold` — are the components that determine behaviour.

2. **Composition is how you build complex things from simple parts.** A mortgage-backed security, a credit default swap, a central bank repo facility — all are compositions of primitive money flows. PCL makes that composition explicit, checkable, and differentiable.

3. **`choose(β, f, g)` is the transistor.** It is the component that turns a passive conservation-enforcing framework into an active, decision-making, optimisable system. Without it, PCL is a ledger. With it, PCL is a computer.
