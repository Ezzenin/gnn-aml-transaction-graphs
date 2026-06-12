"""Тест разделения Elliptic/IBM в сводках (P1.4)."""
import json

from src.compare import collect, collect_ibm


def _write(path, model_type, auc):
    path.write_text(json.dumps({
        "model_type": model_type,
        "test_metrics": {"auc_pr": auc, "f1": auc},
    }), encoding="utf-8")


def test_collect_elliptic_only(tmp_path):
    # Кладём и Elliptic-, и IBM-результаты в одну директорию.
    _write(tmp_path / "elliptic_gcn_metrics.json", "gcn", 0.66)
    _write(tmp_path / "ibm_gine_metrics.json", "gine", 0.05)
    _write(tmp_path / "ibm_xgboost_metrics.json", "xgboost", 0.13)

    elliptic = collect(str(tmp_path))
    names = {r["file"] for r in elliptic}
    assert names == {"elliptic_gcn_metrics.json"}, "IBM просочился в Elliptic-сводку"

    ibm = collect_ibm(str(tmp_path))
    ibm_variants = {r["variant"] for r in ibm}
    assert "XGBoost" in ibm_variants and "GINe (base)" in ibm_variants
