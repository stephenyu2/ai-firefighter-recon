"""
Smoke test: a minimal check that the project and its core dependencies are
importable and the test runner works. This is intentionally shallow. Its job
is to catch a fundamentally broken repo (bad install, missing dependency,
syntax error) before the real test suite runs. Expand this with real tests as
the pipeline is built out.
"""


def test_python_sanity():
    assert 1 + 1 == 2


def test_core_dependencies_import():
    import numpy  # noqa: F401
    import cv2  # noqa: F401  (opencv-python-headless)
