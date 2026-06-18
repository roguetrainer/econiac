# Rationality is Temperature

> *"The question is not whether agents are rational.
> The question is: how rational, at what cost, and measurable from what data?"*

Standard economic models treat agents as perfectly rational: they maximise
utility exactly, choose the unique best option, and never make mistakes. This
is the limit $\beta \to \infty$ — infinite inverse temperature, zero decision
noise, pure argmax.

EconIAC replaces argmax with a single parameter:

$$Z_\beta = \frac{1}{\beta} \ln \sum_i e^{\beta U_i}$$

$\beta$ is the **inverse temperature of rationality** — the inverse of decision
noise. High $\beta$: agents behave nearly classically. Low $\beta$: agents
choose more randomly, weighted by utility but not determined by it.

---

## What this one substitution buys

The classical argmax is a step function. It is not differentiable, not
calibratable from data, and not continuous in its parameters. The Gibbs
softmax $e^{\beta U_i} / Z_\beta$ is all three.

| Classical model | EconIAC equivalent | What changes |
| --- | --- | --- |
| Argmax (perfect choice) | Gibbs softmax | Smooth, differentiable, calibratable |
| Hard threshold (VaR breach) | Sigmoid | Continuous tipping point; early-warning signal |
| Leontief minimum (production) | SoftMin | Differentiable supply chain; exact policy gradient |
| Nash equilibrium | Quantal response equilibrium (QRE) | Calibratable from observed choice variance |
| Shapley value | Thermal Shapley | One backward pass; exact attribution |

The substitution is not an approximation to the classical model — it is a
one-parameter family that **contains** the classical model as its limiting
case. At $\beta \to \infty$ you recover Nash equilibria, Leontief multipliers,
and Shapley values exactly. At finite $\beta$ you gain differentiability.

---

## Calibrating β from data

$\beta$ is not a free parameter to be tuned by hand. It is a physical quantity
measurable from the variance of observed choices:

$$\beta^* = \frac{1}{\mathrm{Var}(\text{observed choices})}$$

More precisely: in a Gibbs ensemble, the variance of a utility-weighted
observable is $\partial^2 \ln Z_\beta / \partial \beta^2$. Inverting this
gives $\beta^*$ from any dataset of observed decisions — bids in an auction,
portfolio allocations, lending choices, voting records.

This calibration has a physical interpretation: $\beta^*$ is the point at
which the system is at its empirical operating temperature. It is not the
temperature at which agents are "approximately rational" — it is the
temperature at which their observed behaviour is thermodynamically consistent.

---

## The tipping point connection

Near a phase transition — a bank run, a repo freeze, a cascading default —
the susceptibility

$$\chi(\beta) = \frac{\partial^2 \ln Z_\beta}{\partial \beta^2}$$

diverges. This is a computable early-warning signal: as the system approaches
a tipping point, $\chi(\beta)$ rises sharply before any individual threshold
is breached.

This is the EconIAC version of the standard early-warning literature (Scheffer
et al. 2009), but derived from first principles rather than fitted to historical
data. The signal is exact, not heuristic, because it follows from the analytic
structure of $Z_\beta$.

---

## Why temperature, not noise

It is tempting to read $\beta$ as "how much noise agents have" — a behavioural
friction. This misses the point.

In statistical mechanics, temperature is not noise — it is the parameter that
controls the trade-off between energy (utility) and entropy (diversity of
choices). A system at high temperature explores many states; a system at low
temperature concentrates on the ground state. The ground state of an economic
system is the Nash equilibrium or Pareto optimum; the high-temperature states
are the heterogeneous, partially-coordinated behaviours actually observed in
markets.

The reason EconIAC uses temperature rather than noise is that temperature is
**extensive and measurable**: it is the same parameter that controls
differentiability, calibration, tipping-point detection, and Shapley
attribution. A noise parameter would give you differentiability but not the
thermodynamic structure that makes $\chi(\beta)$ an early-warning signal.

---

## Connection to the other core ideas

Rationality-as-temperature is the foundation that makes the rest of EconIAC
computable:

- The [three levels of risk](../why/cohomology.md) (H⁰/H¹/H²) are measurable
  because $\beta$ makes the consistency conditions differentiable.
- The [Pentagon identity](pentagon_identity.md) can be tested empirically because
  the Gibbs lift turns hard thresholds into smooth residuals.
- [Clearing and netting](clearing.md) topology changes the phase structure of the
  Gibbs ensemble — CCP novation changes which configurations are reachable at a
  given $\beta$.

---

## Further reading

- Buckley (2026). Paper 289: The Temperature of Rationality.
  doi:[10.5281/zenodo.20234841](https://doi.org/10.5281/zenodo.20234841)
- Buckley (2026). Paper 315: Differentiable Nash.
  doi:[10.5281/zenodo.20318527](https://doi.org/10.5281/zenodo.20318527)
- Buckley (2026). Paper 316: EconIAC / MONIAC.
  doi:[10.5281/zenodo.20315689](https://doi.org/10.5281/zenodo.20315689)
- Jaynes, E.T. (1957). Information theory and statistical mechanics.
  *Physical Review* 106(4), 620–630. (The maximum-entropy foundation.)
- McKelvey, R. & Palfrey, T. (1995). Quantal response equilibria for normal
  form games. *Games and Economic Behavior* 10(1), 6–38. (QRE as finite-β Nash.)
