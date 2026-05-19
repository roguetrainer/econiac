# EconIAC

**Thermodynamic information routing and economic gauge theory on the Pacioli manifold.**

EconIAC is a Python library for building differentiable macroeconomic and financial models grounded
in gauge theory, thermodynamics, and double-entry bookkeeping. It is the software companion to
the [Portfolio G papers](https://zenodo.org/communities/adelic-simplicial-architecture).

## Quick start

```python
from econiac.pcl import flow, sequence, choose, compile, typecheck

# Three sectors, one instrument
wages    = flow("firms", "households", "deposits", 1000.0)
taxes    = flow("households", "government", "deposits", 200.0)
reinvest = flow("households", "firms", "deposits", 500.0)
save     = flow("households", "banks", "deposits", 300.0)

# β=2: lean toward higher-value strategy, but hedge
quarterly = sequence(wages, sequence(taxes, choose(2.0, reinvest, save)))

assert typecheck(quarterly)
fast = compile(quarterly)
```

## Why abstract mathematics?

Start with the [Why EconIAC?](why/README.md) pages — each one motivates a single mathematical
concept from the economic problem it solves, not from the mathematics itself.

## Modules

| Module | What it does |
| --- | --- |
| `econiac.core` | `BalanceSheet`, Gibbs weights, MGE tensor contractions |
| `econiac.routing` | TIR routing geometry, thermal Shapley attribution |
| `econiac.pcl` | Pacioli Combinator Library — conservation-enforcing DSL |
