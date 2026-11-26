"""Benchmark synthetic transaction flags and explain a fixed XGBoost review policy."""
import argparse
import json
import sqlite3
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from xgboost import DMatrix, XGBClassifier

FEATURES = ("amount_ratio", "days_since_bank_change", "payments_7d", "new_recipient", "after_hours")
LABELS = ("amount relative to usual", "days since bank details changed", "payments in seven days",
          "first-time recipient indicator", "after-hours indicator")


def simulate(n, seed):
    if n < 600:
        raise ValueError("Choose at least 600 transactions for the time-ordered split.")
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.lognormal(0, .65, n), rng.exponential(55, n).round(),
                         rng.poisson(2, n), rng.binomial(1, .22, n), rng.binomial(1, .15, n)]).astype("float32")
    log_odds = (-3.7 + 1.7*(X[:, 0] > 1.8) + 1.5*(X[:, 1] < 9) + 1.2*(X[:, 2] > 3)
                + 1.5*X[:, 3] + 1.0*X[:, 4])
    y = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))
    return X, y


def choose_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def explain(row, contributions):
    top = np.argsort(-np.abs(contributions))[:3]
    evidence = [{"feature": FEATURES[i], "label": LABELS[i], "value": round(float(row[i]), 3),
                 "log_odds_effect": round(float(contributions[i]), 5)} for i in top]
    pieces = [f"{e['label']} ({e['value']:g}) {'raised' if e['log_odds_effect'] >= 0 else 'lowered'} the score"
              for e in evidence]
    return evidence, "Model review flag. " + "; ".join(pieces) + "."


def ollama_summary(evidence, model):
    prompt = ("Describe only the listed model factors in one plain-English sentence for a reviewer. "
              "Positive effects raise the score. Negative effects lower it. Avoid attributing intent. "
              + json.dumps(evidence))
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0}}).encode()
    req = Request("http://127.0.0.1:11434/api/generate", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=90) as response:
        result = json.load(response)["response"].strip()
    if not result:
        raise ValueError("Ollama returned an empty explanation")
    return result


def analyze(*, rows, seed, database, tabnet=False, ollama=None):
    X, y = simulate(rows, seed)
    a, b = int(.6*rows), int(.8*rows)
    if any(len(np.unique(part)) < 2 for part in (y[:a], y[a:b], y[b:])):
        raise ValueError("A partition has only one label. Increase --rows or use another seed.")
    tree = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=.07, n_jobs=1, random_state=seed)
    models = {"xgboost": tree, "gradient_boosting": GradientBoostingClassifier(random_state=seed)}
    if tabnet:
        from pytorch_tabnet.tab_model import TabNetClassifier
        models["tabnet"] = TabNetClassifier(n_d=8, n_a=8, n_steps=3, seed=seed, verbose=0, device_name="cpu")
    metrics = {}
    for name, model in models.items():
        kwargs = {}
        if name == "tabnet":
            kwargs = dict(eval_set=[(X[a:b], y[a:b])], eval_metric=["auc"], max_epochs=16, patience=4,
                          batch_size=128, virtual_batch_size=32, num_workers=0)
        model.fit(X[:a], y[:a], **kwargs)
        threshold = choose_threshold(y[a:b], model.predict_proba(X[a:b])[:, 1])
        score = model.predict_proba(X[b:])[:, 1]
        metrics[name] = {"threshold": threshold, "precision": float(precision_score(y[b:], score >= threshold, zero_division=0)),
                         "recall": float(recall_score(y[b:], score >= threshold, zero_division=0)),
                         "roc_auc": float(roc_auc_score(y[b:], score))}
    prob = tree.predict_proba(X[b:])[:, 1]
    flagged = np.flatnonzero(prob >= metrics["xgboost"]["threshold"])
    effects = tree.get_booster().predict(DMatrix(X[b:]), pred_contribs=True)
    run_id = uuid4().hex
    queue = []
    for i in flagged:
        evidence, summary = explain(X[b+i], effects[i, :-1])
        if ollama:
            summary = ollama_summary(evidence, ollama)
        queue.append((run_id, b+int(i)+1, float(prob[i]), json.dumps(evidence), summary))
    db = sqlite3.connect(database)
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, seed INTEGER, rows INTEGER, metrics TEXT, narrator TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS flags (run_id TEXT REFERENCES runs(id), transaction_no INTEGER, "
                       "score REAL, evidence TEXT, explanation TEXT, PRIMARY KEY(run_id, transaction_no))")
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, seed, rows, json.dumps(metrics), ollama or "template"))
            db.executemany("INSERT INTO flags VALUES (?,?,?,?,?)", queue)
    finally:
        db.close()
    return {"run_id": run_id, "train": a, "validation": b-a, "test": rows-b, "benchmark": metrics,
            "flagged": len(queue), "explanation_mode": ollama or "template"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--database", default="review.db")
    p.add_argument("--tabnet", action="store_true", help="Include the third model (optional dependencies)")
    p.add_argument("--ollama", metavar="MODEL", help="Generate narratives locally, e.g. llama2")
    print(json.dumps(analyze(**vars(p.parse_args())), indent=2))
