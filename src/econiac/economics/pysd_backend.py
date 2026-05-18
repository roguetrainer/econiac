"""
pysd JAX backend: execute Stella/Vensim system dynamics models via JAX.

pysd reads Stella (.stmx) and Vensim (.mdl) files and executes them in Python.
This backend replaces pysd's numpy executor with a JAX executor, making every
existing system dynamics model in academia differentiable and GPU-accelerated.

Adoption path: PR to pysd upstream adding 'backend=jax' option.

Usage (planned):
    import pysd
    model = pysd.read_stella('my_model.stmx', backend='jax')
    result = model.run(params={'birth_rate': 0.02})
    # result is a JAX array — differentiable, GPU-accelerated

Reference: Buckley (2026) DABM, doi:10.5281/zenodo.20261945
"""

# TODO Phase 3: implement
# Requires: pip install pysd
#
# - JAXBackend — pysd-compatible execution backend using JAX
# - compile_to_jax(pysd_model) -> JAX-compiled forward function
# - calibrate_stella(model_path, data, params_to_fit) -> fitted_params
