"""Smoke tests: verify package imports and version."""

import econiac

def test_version():
    assert econiac.__version__ == "0.0.2"

def test_imports():
    from econiac.core import ensemble, manifold, connections, geometry
    from econiac.finance import curves, fx, credit, quantlib
    from econiac.economics import sfc, agents, minsky, pysd_backend
    from econiac.routing import tir, attribution
    from econiac.pcl import combinators
