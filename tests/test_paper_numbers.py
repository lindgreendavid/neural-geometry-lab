import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEX = (ROOT / "paper" / "paper.tex").read_text(encoding="utf-8")
AUDIT = json.loads((ROOT / "reports" / "post-release-within-condition.json").read_text())
FROZEN = json.loads((ROOT / "reports" / "results-v1.0.json").read_text())


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def test_pooled_correlations_in_paper_match_the_frozen_results():
    corr = FROZEN["evaluation"]["descriptive_correlations"]
    assert f"{corr['spearman_final_nc1_vs_test_accuracy']:.3f}" in TEX
    assert f"{abs(corr['spearman_final_nc2_vs_test_accuracy']):.3f}" in TEX


def test_within_condition_table_matches_the_audit_file():
    for coord in ("nc1", "nc2", "nc3", "nc4"):
        block = AUDIT["coordinates"][coord]
        assert f"{abs(block['pooled']['spearman']):.3f}" in TEX
        assert f"{abs(block['pooled_within_condition']['spearman']):.3f}" in TEX
        for condition in ("clean", "long_tail", "label_noise_20"):
            value = block["per_condition"][condition]["spearman"]
            assert f"{abs(value):.2f}" in TEX, (coord, condition)


def test_gate_counts_in_paper_match_the_frozen_results():
    gates = FROZEN["evaluation"]["gates"]
    assert gates["clean_nc1_after_zero"]["count"] == 7 and "7/10" in TEX
    assert gates["clean_nc4_after_zero"]["count"] == 10 and "10/10" in TEX
