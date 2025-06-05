import simlearn


def test_version_string() -> None:
    assert hasattr(simlearn, "__version__")
    assert isinstance(simlearn.__version__, str)
