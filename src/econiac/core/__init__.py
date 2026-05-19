"""econiac.core — statistical ensemble, Pacioli manifold, connections, geometry."""

from econiac.core.connections import (
    Connection,
    parallel_transport,
    wilson_loop,
    log_holonomy,
    curvature,
    is_flat,
    max_curvature,
    curvature_matrix,
    gauge_transform,
    flat_gauge,
    fx_connection,
    discount_connection,
)

from econiac.core.geometry import (
    GeometryType,
    AbelianGeometry,
    FanoGeometry,
    FANO_LINES,
    G2Geometry,
    CatalanGeometry,
    geometry_type_of,
)

from econiac.core.manifold import (
    BalanceSheet,
    GodleyTable,
    PacioliManifold,
    HomologyGroups,
    CurvedBalanceSheet,
    holonomy,
    add_residual_sector,
    add_float_sector,
    residual_magnitude,
    RESIDUAL_SECTOR,
    FLOAT_SECTOR,
    three_sector_sfc,
)

from econiac.core.ensemble import (
    partition_function,
    log_partition,
    gibbs_weights,
    free_energy,
    entropy,
    mean_utility,
    choose,
    beta_schedule,
    ensemble_sweep,
    summarise,
    EnsembleSummary,
)
