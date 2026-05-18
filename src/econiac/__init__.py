"""
econiac (Economic Integrator And Computer) — Thermodynamic Information Routing
and Economic Gauge Theory on the Pacioli manifold.

Named after MONIAC (1949), Bill Phillips's hydraulic computer that modelled
the economy as a conserved flow system. econiac does the same with
differential geometry, the Maslov-Gibbs partition function, and JAX.

What does econiac compute?
  - Accounting measures    sectoral balances, XVA — integrals of curvature
  - Entropies & free energies  the partition function Z(β) at the core of TIR
  - Sensitivities          Greeks, thermal attribution — autograd of the above
  - Optima & equilibria    the high-β limit of the Gibbs distribution
  - Calibration weights    the β-schedule from analogue exploration to decision

"Integrator" is doubly apt: econiac literally integrates ODEs (Keen dynamics,
pysd backend) and integrates in the measure-theoretic sense (partition
functions, path integrals over the Pacioli manifold).

The lineage: ENIAC (1945, digital) → MONIAC (1949, analogue) → econiac
(analogue emulated on a digital computer — and digital in the high-β limit).

See: https://doi.org/10.5281/zenodo.20237288 (Thermodynamic Information Routing)
     https://doi.org/10.5281/zenodo.20259495 (Economic Gauge Theory)
     https://doi.org/10.5281/zenodo.20262070 (Pacioli Combinator Library)
"""

__version__ = "0.0.2"
__author__ = "Ian R. C. Buckley"
