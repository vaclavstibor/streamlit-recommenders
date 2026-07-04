from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_polished_examples_exist():
    assert (EXAMPLES / "baseline_comparison_demo.py").exists()
    assert (EXAMPLES / "appendix_demo.py").exists()
    assert (EXAMPLES / "sequence_cf_demo.py").exists()


def test_legacy_examples_removed():
    assert not (EXAMPLES / "_bootstrap.py").exists()
    assert not (EXAMPLES / "minimal_demo.py").exists()
    assert not (EXAMPLES / "pickle_demo.py").exists()
    assert not (EXAMPLES / "matrix_demo.py").exists()
    assert not (EXAMPLES / "showcase_demo.py").exists()
