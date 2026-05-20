"""
Example: Supply Chain Reverse Stress Testing via Differentiable ADTs.

Supply chain risk and financial contagion are DUAL algebraic types:

  Financial risk:    OR  logic  (TensorSumStep) → losses accumulate
  Supply chain risk: AND logic  (TensorMinStep) → bottlenecks bind

This example demonstrates:
  1. The copper supply chain topology and its Laplacian spectrum
  2. Forward simulation: capacity propagation through a 4-tier chain
  3. Shock scenarios: what happens to OEM output when CHL_Mine is disrupted
  4. Reverse Stress Testing (RST): find minimal buffers to survive the shock
  5. The ADT / PCL connection: how the network maps to PCL combinator trees

References:
    Smith et al. (2025) Reverse Stress Testing for Supply Chain Resilience.
        arXiv:2511.07289
    Buckley (2026) PCL. doi:10.5281/zenodo.20262070
"""

import numpy as np
from econiac.economics.supply_chain import (
    COPPER_CHAIN,
    COPPER_NODES,
    simulate_chain,
    reverse_stress_test,
    laplacian_spectrum,
    apply_shock,
    to_pcl_description,
)


# ---------------------------------------------------------------------------
# 1. Network topology and PCL structure
# ---------------------------------------------------------------------------

print("=" * 70)
print("1. Copper supply chain topology")
print("=" * 70)
print()
print(to_pcl_description(COPPER_CHAIN))

spec = laplacian_spectrum(COPPER_CHAIN)
print(f"  Laplacian eigenvalues: {np.round(spec['eigenvalues'], 3)}")
print(f"  Fiedler value λ₂ = {spec['fiedler_value']:.4f}")
print(f"  Connected components: {spec['n_components']}")
print()
print("  Interpretation:")
print(f"  λ₂ = {spec['fiedler_value']:.4f} — algebraic connectivity.")
print("  Small λ₂ → fragile: a shock at any tier reaches the OEM quickly.")
print()

fv = spec['fiedler_vector']
fiedler_sorted = sorted(zip(COPPER_NODES, fv), key=lambda x: x[1])
print("  Fiedler vector (cuts the graph at bottleneck):")
for name, val in fiedler_sorted:
    bar = "█" * int(abs(val) * 30)
    sign = "−" if val < 0 else "+"
    print(f"    {name:20s}  {sign}{abs(val):.4f}  {bar}")
print()
print("  Nodes with negative Fiedler values are on the 'upstream' side")
print("  of the most vulnerable cut. A disruption here propagates farthest.")
print()


# ---------------------------------------------------------------------------
# 2. Baseline simulation (no shock)
# ---------------------------------------------------------------------------

print("=" * 70)
print("2. Baseline forward simulation (full capacity, no buffers)")
print("=" * 70)
print()

baseline_capacity = np.ones(len(COPPER_NODES))
result_base = simulate_chain(COPPER_CHAIN, baseline_capacity)

print(f"  {'Node':22s}  {'Final cap':>10}")
print(f"  {'-'*35}")
for name, cap in zip(COPPER_NODES, result_base['final_capacity']):
    bar = "█" * int(cap * 20)
    print(f"  {name:22s}  {cap:9.3f}  {bar}")
print()


# ---------------------------------------------------------------------------
# 3. Shock scenarios
# ---------------------------------------------------------------------------

print("=" * 70)
print("3. Shock scenarios: disruption at CHL_Mine (50% / 20% / 0% capacity)")
print("=" * 70)
print()

print(f"  {'Shock':10s}  {'OEM output':>11}  {'Impact':>8}  Key bottleneck")
print(f"  {'-'*55}")

for shock_level in [1.0, 0.5, 0.2, 0.0]:
    shocked = apply_shock(
        COPPER_CHAIN,
        shocked_nodes=["CHL_Mine"],
        shock_fraction=shock_level,
    )
    result = simulate_chain(COPPER_CHAIN, shocked)
    oem_out = result['final_capacity'][0]  # US_OEM is index 0
    impact = 1.0 - oem_out
    bottleneck = COPPER_NODES[result['bottleneck_nodes'][0]]
    label = f"{int((1 - shock_level)*100):3d}% loss"
    print(f"  {label:10s}  {oem_out:10.3f}  {impact:+7.1%}  {bottleneck}")
print()

# Multi-node shock: CHL_Mine + DRC_Mine (dual mining disruption)
print("  Multi-node shock: both CHL_Mine AND DRC_Mine at 20%")
shocked_dual = apply_shock(
    COPPER_CHAIN,
    shocked_nodes=["CHL_Mine", "DRC_Mine"],
    shock_fraction=0.2,
)
result_dual = simulate_chain(COPPER_CHAIN, shocked_dual)
oem_dual = result_dual['final_capacity'][0]
print(f"  OEM output: {oem_dual:.3f}  ({(1-oem_dual):.1%} loss)")
print()


# ---------------------------------------------------------------------------
# 4. Reverse Stress Testing: find minimum buffers to survive shock
# ---------------------------------------------------------------------------

print("=" * 70)
print("4. Reverse Stress Testing: minimum buffers to survive CHL_Mine shock")
print("=" * 70)
print()

shock_50 = apply_shock(COPPER_CHAIN, shocked_nodes=["CHL_Mine"], shock_fraction=0.5)

for required in [0.7, 0.85, 0.95]:
    rst = reverse_stress_test(
        COPPER_CHAIN,
        shock_capacity=shock_50,
        required_output=required,
        output_node=0,
        budget_weight=0.1,
        n_epochs=300,
    )

    print(f"  Required OEM output ≥ {required:.0%}:")
    print(f"    Final output achieved: {rst['final_output']:.3f}  "
          f"({'✓ survived' if rst['survived'] else '✗ failed'})")
    print(f"    Total buffer cost:     {rst['buffers'].sum():.4f}")
    print()

    # Show which nodes need buffering
    print(f"    {'Node':22s}  {'Buffer':>8}  {'Criticality':>12}")
    print(f"    {'-'*46}")
    order = np.argsort(-rst['criticality'])
    for i in order:
        buf = rst['buffers'][i]
        crit = rst['criticality'][i]
        if buf > 0.001 or crit > 0.001:
            bar = "█" * int(crit * 40)
            print(f"    {COPPER_NODES[i]:22s}  {buf:8.4f}  {crit:11.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# 5. Dual shock RST: both mines disrupted
# ---------------------------------------------------------------------------

print("=" * 70)
print("5. RST: dual mining shock — CHL_Mine + DRC_Mine at 20%")
print("=" * 70)
print()

rst_dual = reverse_stress_test(
    COPPER_CHAIN,
    shock_capacity=shocked_dual,
    required_output=0.80,
    output_node=0,
    budget_weight=0.05,
    n_epochs=400,
)

print(f"  Required: OEM ≥ 80% after both mines disrupted")
print(f"  Result:   {rst_dual['final_output']:.3f}  ({'✓' if rst_dual['survived'] else '✗'})")
print(f"  Total buffer cost: {rst_dual['buffers'].sum():.4f}")
print()
print(f"  {'Node':22s}  {'Buffer':>8}  {'Criticality':>12}")
print(f"  {'-'*46}")
order = np.argsort(-rst_dual['criticality'])
for i in order[:5]:
    buf = rst_dual['buffers'][i]
    crit = rst_dual['criticality'][i]
    bar = "█" * int(crit * 40)
    print(f"  {COPPER_NODES[i]:22s}  {buf:8.4f}  {crit:11.4f}  {bar}")
print()


# ---------------------------------------------------------------------------
# 6. The ADT duality
# ---------------------------------------------------------------------------

print("=" * 70)
print("6. Algebraic type duality: supply chain vs. financial contagion")
print("=" * 70)
print("""
  Supply chain (this model):        Financial contagion (DebtRank):
  ─────────────────────────────     ──────────────────────────────────
  AND logic  — bottleneck           OR logic   — accumulation
  TensorMinStep(BOM, cap)           TensorSumStep(adj, exposure)
  Product type  (Record)            Sum type  (Union)
  PCL: fold(β→∞, [t₁, t₂, …])      PCL: parallel(flow_A, flow_B, …)
  Neutral element: 1  (none)        Neutral element: 0  (no exposure)
  Failure mode: starvation          Failure mode: contagion

  Both are differentiable — RST uses JAX grad() on the same loss:
    shortfall = relu(required - final_output)
    cost      = sum(buffers)
    loss      = 10 · shortfall + λ · cost

  The gradient ∂loss/∂shock_capacity is the criticality vector:
  it tells you which nodes, if shocked, cause the largest downstream
  loss — the 'straw that breaks the camel's back' (Smith et al. 2025).

  Curry-Howard correspondence (ADT ↔ Logic ↔ Type):
    Supply chain AND ↔ Product type  ↔ Leontief BOM  ↔ PCL fold
    Financial    OR  ↔ Sum type      ↔ DebtRank adj  ↔ PCL parallel
""")

print("=" * 70)
print("Supply chain RST example complete.")
print("=" * 70)
