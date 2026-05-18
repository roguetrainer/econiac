# Why does econiac use thermodynamics?

*You are an economist or quant. You have never needed a partition function. Why should you start now?*

---

## The aggregation problem you already have

Every model of collective behaviour faces the same problem: you have a population of agents with different beliefs, different budgets, different risk tolerances — and you need to aggregate them into a single demand curve, a market price, or a policy response. How?

The standard answer is the **representative agent**: replace the heterogeneous population with a single fictional agent whose preferences are the average. This is mathematically convenient and almost always wrong. Representative agents cannot produce excess volatility, cannot crash, cannot generate the fat tails that every financial time series exhibits. You already know this. The question is what to do instead.

## The Gibbs distribution is the only honest answer

Suppose you know the average utility across your population — the mean, but not the distribution. You want to assign probabilities to each possible aggregate outcome. What probability distribution should you use?

The answer is not arbitrary. There is exactly one distribution that:
- matches the known mean (uses all the information you have), and
- makes no additional assumptions (uses no information you don't have).

That distribution is the **Gibbs distribution**:

```
P(state i) ∝ exp(−β · U_i)
```

where `U_i` is the utility (or cost, or energy) of state `i`, and `β` is the inverse temperature — a measure of how much the population discriminates between states. This is not a modelling choice. It is the *unique* answer consistent with maximum entropy, proved by Jaynes (1957) and re-derived independently by eight separate research programmes across economics, neuroscience, statistical mechanics, and computer science.

## What β means for economists

β is not a physicist's concept borrowed awkwardly into economics. It is the **rationality parameter** — how sharply agents discriminate between options.

- **β → 0** (high temperature): agents are indifferent, choose randomly. This is the noise trader limit.
- **β → ∞** (low temperature): agents always choose the best option. This is the perfectly rational limit of standard economics.
- **β finite**: agents are partially rational, with the degree of rationality calibrated to data.

The standard representative agent model implicitly assumes β → ∞. The Gibbs distribution makes β explicit and calibratable. McKelvey and Palfrey (1995) derived exactly this framework for game theory — they called it the Quantal Response Equilibrium. Sims (2003) derived it from information constraints — he called it Rational Inattention. Friston (2010) derived it from variational principles in neuroscience — he called it Active Inference. They all got the same formula.

## What this means for your models

econiac uses β in three ways:

**1. Aggregation**: instead of averaging agent behaviour, weight each outcome by `exp(−β · cost)`. At β calibrated to data, this correctly reproduces fat tails, excess volatility, and crash dynamics that representative agents cannot.

**2. Calibration**: β is differentiable. Given observed aggregate outcomes, gradient descent finds the β — and the underlying parameters — that best fit the data. This is impossible with discrete if/then agent rules.

**3. The β-schedule**: in the Maslov-Gibbs Einsum (MGE), β starts low (analogue exploration) and rises to high (digital commitment). This is the same trajectory as simulated annealing, but with a mathematical guarantee: the free energy is minimised at every step. econiac's optimiser is thermodynamically grounded; most optimisers are not.

## The conservation law connection

Thermodynamics is not just about temperature. The deeper result — proved in Paper 201 — is that the Gibbs routing weights are the *unique* weights that simultaneously preserve three geometric conservation laws: conformal invariance (scale-free in utility units), symplectic conservation (no information is destroyed in routing), and adiabatic invariance (the system tracks the free energy minimum as parameters change).

This is why transformer attention — which uses a fixed temperature β = 1/√d — breaks under distribution shift: it violates adiabatic invariance. econiac's β-schedule does not.

## The short version

You need thermodynamics because:

1. The Gibbs distribution is the only aggregation rule consistent with having partial information about a heterogeneous population.
2. β is the rationality parameter — calibratable, differentiable, and already implicit in your models.
3. The thermodynamic conservation laws explain *why* the Gibbs form is universal, not just *that* it is.

If your model aggregates heterogeneous agents, it is already implicitly thermodynamic. econiac makes the thermodynamics explicit — and correct.
