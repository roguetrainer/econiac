"""econiac.routing — TIR framework and thermal Shapley attribution."""

from econiac.routing.tir import (
    TIRInstance,
    route,
    free_energy,
    escape_arrow,
    admissible_count,
    routing_entropy,
    social_multiplier,
    tir_from_scores,
)

from econiac.routing.attribution import (
    thermal_shapley,
    bottleneck_index,
    tropical_limit,
    nonassociative_shapley,
    pacioli_attribution,
    ShapleyResult,
    full_shapley_analysis,
)
