# econiac

**Thermodynamic information routing and economic gauge theory on the Pacioli manifold.**

Named after [MONIAC](https://en.wikipedia.org/wiki/MONIAC) (1949) — Bill Phillips's hydraulic computer that modelled the economy as a conserved flow system using tanks, pipes, and valves. `econiac` does the same with differential geometry, the Maslov-Gibbs partition function, and JAX.

> *"The economy is a flow network. Conservation laws are not optional."*

---

## What is econiac?

`econiac` is the Python implementation of the **Adelic Simplicial Architecture (ASA)** economic and financial frameworks:

- **Thermodynamic Information Routing (TIR)** — the Gibbs distribution as a universal routing primitive across economics, game theory, neuroscience, and computation ([Paper 294](https://doi.org/10.5281/zenodo.20237288))
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

In 1949, the New Zealand economist Bill Phillips built MONIAC: a hydraulic computer that modelled the British economy as a system of tanks, pipes, and valves, with coloured water representing money flows. Conservation was enforced physically — what flowed in had to flow out. It was a working analogue computer that correctly predicted macroeconomic dynamics.

`econiac` is MONIAC for the 21st century: the same conservation laws, the same flow network, but implemented with differential geometry (the Pacioli manifold), thermodynamic routing (the Maslov-Gibbs partition function), and automatic differentiation (JAX).

## The Pacioli manifold

The **Pacioli manifold** is named after Luca Pacioli (1447–1517), the Franciscan friar who documented double-entry bookkeeping in his 1494 *Summa de arithmetica* — the first printed treatment of the system in Europe. The name honours the mathematical structure he documented, not its invention.

Double-entry bookkeeping was not a European invention. The practice predates Pacioli by centuries and likely arrived in Europe via multiple routes:

- **India**: the *Bahi-Khata* system of double-entry ledger accounting, used by merchants in Rajasthan and Gujarat, dates to at least the 12th century and possibly earlier.
- **Islamic world**: Abbasid-era *hawala* (8th–13th century) and the sophisticated accounting practices of medieval Islamic merchants documented in the Cairo Geniza records show conservation-of-value principles structurally equivalent to double-entry.
- **China**: the *四柱清册* (four-column account) system used by Chinese merchants from at least the Tang dynasty (7th century) encodes the same debit-credit conservation law.
- **Transmission to Europe**: Italian merchants — particularly in Venice, Genoa, and Florence — encountered these systems through the spice and silk trades. The Florentine merchant Francesco Datini's ledgers (c. 1299) predate Pacioli by nearly two centuries.

Pacioli's contribution was to recognise the mathematical structure and give it a systematic written treatment. By Stigler's Law of Eponymy (*no scientific discovery is named after its original discoverer*), "Pacioli manifold" is in good historical company. The conservation law ∂²=0 that defines the manifold belongs to no single culture — it is a universal consequence of the mathematics of flows.

## References

- Buckley, I. R. C. (2026). Thermodynamic Information Routing. [doi:10.5281/zenodo.20237288](https://doi.org/10.5281/zenodo.20237288)
- Buckley, I. R. C. (2026). Economic Gauge Theory. [doi:10.5281/zenodo.20259495](https://doi.org/10.5281/zenodo.20259495)
- Buckley, I. R. C. (2026). The Pacioli Combinator Library. [doi:10.5281/zenodo.20262070](https://doi.org/10.5281/zenodo.20262070)

## License

MIT
