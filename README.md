# EconIAC

**Thermodynamic information routing and economic gauge theory on the Pacioli manifold.**

**Econ**omic **I**ntegrator **A**nd **C**omputer — named after [MONIAC](https://en.wikipedia.org/wiki/MONIAC) (1949), Bill Phillips's hydraulic computer that modelled the economy as a conserved flow system using tanks, pipes, and valves. `Econiac` does the same with differential geometry, the Maslov-Gibbs partition function, and JAX.

> *"The economy is a flow network. Conservation laws are not optional."*

---

## What is EconIAC?

`Econiac` is the Python implementation of the **Adelic Simplicial Architecture (ASA)** economic and financial frameworks:

- **Thermodynamic Information Routing (TIR)** — the Gibbs distribution as a universal routing primitive across economics, game theory, neuroscience, and computation ([Paper 294](https://doi.org/10.5281/zenodo.20237288))
- **Maslov-Gibbs Einsum (MGE)** — the active tensor contraction operator that executes Gibbs-weighted routing; in the tropical limit (β→∞) it dequantizes to a max-plus semiring, recovering discrete argmax from a continuous sum-product ([Paper 201](https://doi.org/10.5281/zenodo.17981393))
- **Economic Gauge Theory (EGT)** — stock-flow consistency as a discrete gauge theory on the Pacioli manifold ([Paper 300](https://doi.org/10.5281/zenodo.20259495))
- **Financial Gauge Theory (FGT)** — exchange rates, yield curves, and credit spreads as connections and curvature ([Papers 295–299](https://doi.org/10.5281/zenodo.20242355))
- **Pacioli Combinator Library (PCL)** — a domain-specific language for financial computation that enforces conservation by construction ([Paper 306](https://doi.org/10.5281/zenodo.20262070))
- **Differentiable Agent-Based Macroeconomics (DABM)** — fully differentiable macroeconomic digital twins calibratable by gradient descent ([Paper 305](https://doi.org/10.5281/zenodo.20261945))

## Status

**0.0.1 — placeholder release.** Full implementation in progress.

## Installation

```bash
pip install econiac
```

## The MONIAC connection

In 1949, the New Zealand economist Bill Phillips built [MONIAC](https://en.wikipedia.org/wiki/MONIAC): a hydraulic computer that modelled the British economy as a system of tanks, pipes, and valves, with coloured water representing money flows. Conservation was enforced physically — what flowed in had to flow out. It was a working analogue computer that correctly predicted macroeconomic dynamics.

MONIAC stands for **Monetary National Income Analogue Computer** — and is simultaneously a deliberate portmanteau of [ENIAC](https://en.wikipedia.org/wiki/ENIAC) (the 1945 electronic general-purpose computer) and "money". Phillips coined it to signal that economic computation deserved the same ambition as the cutting-edge computing of his day. `Econiac` continues that chain: ENIAC → MONIAC → EconIAC.

`Econiac` also backronyms cleanly: **Econ**omic **I**ntegrator **A**nd **C**omputer — echoing ENIAC's own expansion (*Electronic Numerical Integrator And Computer*), with "Integrator" shared verbatim. The word is doubly apt: `Econiac` literally integrates ODEs (Keen debt dynamics, pysd/Stella/Vensim models) and integrates in the measure-theoretic sense (partition functions Z(β), path integrals over the Pacioli manifold).

**What does it compute?** All of the following are special cases of the same underlying computation — the free energy of a flow network at inverse temperature β:

| Output | Description |
| --- | --- |
| Accounting measures | Sectoral balances, XVA — integrals of curvature on the Pacioli manifold |
| Entropies & free energies | The partition function Z(β) at the core of TIR and the Maslov-Gibbs Einsum (MGE) |
| Sensitivities | Greeks, thermal attribution — automatic differentiation via JAX |
| Optima & equilibria | The high-β (low-temperature) limit of the Gibbs distribution |
| Calibration weights | The β-schedule trajectory from analogue exploration to discrete commitment |

`Econiac` is MONIAC for the 21st century: the same conservation laws, the same flow network, but implemented with differential geometry (the Pacioli manifold), thermodynamic routing (the Maslov-Gibbs partition function), and automatic differentiation (JAX).

## The Pacioli manifold

The **Pacioli manifold** is named after Luca Pacioli (1447–1517), the Franciscan friar who documented double-entry bookkeeping in his 1494 *Summa de arithmetica* — the first printed treatment of the system in Europe. The name honours the mathematical structure he documented, not its invention.

Double-entry bookkeeping was not a European invention. The practice predates Pacioli by centuries and likely arrived in Europe via multiple routes:

- **India**: the *Bahi-Khata* system of double-entry ledger accounting, used by merchants in Rajasthan and Gujarat, dates to at least the 12th century and possibly earlier.
- **Islamic world**: Abbasid-era *hawala* (8th–13th century) and the sophisticated accounting practices of medieval Islamic merchants documented in the Cairo Geniza records show conservation-of-value principles structurally equivalent to double-entry.
- **China**: the *四柱清册* (four-column account) system used by Chinese merchants from at least the Tang dynasty (7th century) encodes the same debit-credit conservation law.
- **Transmission to Europe**: Italian merchants — particularly in Venice, Genoa, and Florence — encountered these systems through the spice and silk trades. The Florentine merchant Francesco Datini's ledgers (c. 1299) predate Pacioli by nearly two centuries.

Pacioli's contribution was to recognise the mathematical structure and give it a systematic written treatment. By Stigler's Law of Eponymy (*no scientific discovery is named after its original discoverer*), "Pacioli manifold" is in good historical company. The conservation law ∂²=0 that defines the manifold belongs to no single culture — it is a universal consequence of the mathematics of flows.

## FAQ

**Is EconIAC analogue or digital?**

All three, in historical order:

- **ENIAC (1945)** was *digital* — discrete voltage states, Boolean logic, stored program.
- **MONIAC (1949)** was *analogue* — continuous water flows, physical conservation enforced by hydraulics.
- **EconIAC** is *analogue emulated on a digital computer* — continuous probability flows (the Gibbs distribution) running on discrete hardware.

There is a precise sense in which EconIAC is also digital: in the high-β (low-temperature) limit, the Gibbs weights collapse to a hard argmax — one route takes all the flow, all others get zero. This is winner-takes-all, i.e. digital switching. At low β the weights are a soft mixture over all routes: fully analogue. The β-schedule therefore interpolates between analogue (exploration) and digital (commitment) behaviour. The SNAP phase of the MGE (Maslov-Gibbs Einsum) algorithm is literally this transition — analogue search crystallising into a discrete decision.

So EconIAC is analogue at heart, digital at the boundary, and the boundary is where the answer lives.

## References

- Buckley, I. R. C. (2026). Thermodynamic Information Routing. [doi:10.5281/zenodo.20237288](https://doi.org/10.5281/zenodo.20237288)
- Buckley, I. R. C. (2026). Economic Gauge Theory. [doi:10.5281/zenodo.20259495](https://doi.org/10.5281/zenodo.20259495)
- Buckley, I. R. C. (2026). The Pacioli Combinator Library. [doi:10.5281/zenodo.20262070](https://doi.org/10.5281/zenodo.20262070)

## License

MIT
