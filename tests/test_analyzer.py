import json
import sqlite3
import os
import pytest
import analyzer


def test_queue_and_benchmark(tmp_path):
    db = tmp_path/"review.db"
    report = analyzer.analyze(rows=900, seed=42, database=db, tabnet=os.getenv("TEST_TABNET") == "1")
    assert report["train"] + report["validation"] + report["test"] == 900
    expected = {"gradient_boosting","xgboost"}
    if os.getenv("TEST_TABNET") == "1":
        expected.add("tabnet")
    assert set(report["benchmark"]) == expected
    for result in report["benchmark"].values():
        assert all(0 <= result[k] <= 1 for k in ("precision","recall","roc_auc"))
    with sqlite3.connect(db) as con:
        flags = con.execute("SELECT transaction_no, score, evidence, explanation FROM flags").fetchall()
    assert len(flags) == report["flagged"] > 0
    for number, score, evidence, summary in flags:
        assert number > 720 and score >= report["benchmark"]["xgboost"]["threshold"]
        assert len(json.loads(evidence)) == 3 and summary.startswith("Model review flag.")


def test_ollama_payload(monkeypatch):
    from io import BytesIO
    def fake(request, timeout):
        body = json.loads(request.data)
        assert body["model"] == "llama2" and not body["stream"]
        assert "amount_ratio" in body["prompt"]
        return BytesIO(b'{"response":"The amount ratio raised the score."}')
    monkeypatch.setattr(analyzer, "urlopen", fake)
    assert "raised" in analyzer.ollama_summary([{"feature":"amount_ratio"}], "llama2")


def test_small_sample():
    with pytest.raises(ValueError):
        analyzer.simulate(8, 1)
