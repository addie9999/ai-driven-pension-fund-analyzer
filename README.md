# AI-Driven Pension Fund Analyzer

Which simulated pension payments should a reviewer inspect first, and which transaction features pushed the model toward a flag? This Python project compares three supervised classifiers on synthetic transaction events and writes an explainable review queue to SQLite.

## Run the demo

Use Python 3.11 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python analyzer.py --database review.db
python -m pytest -q
```

The quick demo trains XGBoost and scikit-learn Gradient Boosting. For the complete three-model comparison, install the TabNet dependencies and run:

```sh
python -m pip install -r requirements-tabnet.txt
python analyzer.py --tabnet --database review.db
```

The JSON output gives each model's classification threshold, held-out precision, recall and ROC-AUC, plus a run ID. Use that ID to inspect the highest-scoring cases:

```sql
SELECT transaction_no, ROUND(score, 3) AS model_score, explanation, evidence
FROM flags WHERE run_id = :run_id ORDER BY score DESC LIMIT 10;
```

## How the evaluation works

The generator creates a seeded sequence of payments using five features, including amount relative to usual, recent bank-detail changes, and payment frequency. Labels are sampled probabilistically. All models train on the first 60% of events. The next 20% sets each model's threshold by validation F1. The last 20% is scored once for the reported metrics. The simulation has no real members, account identifiers or institution records.

XGBoost supplies the review queue regardless of which model scores highest. Its native TreeSHAP contributions identify the three largest absolute feature effects for each flag. The stored evidence records the observed value, signed contribution on the log-odds scale, and a readable description. Every flagged event gets a deterministic explanation by default.

## Local LLaMA 2 narratives

With Ollama running locally and the model installed, use `python analyzer.py --ollama llama2`. The program sends the three model factors to the local model, keeps the structured evidence alongside the narrative, and records the narrator in the run table. A failed generation aborts that run before database writes. The tests simulate the local model response and exercise the full three-model benchmark in GitHub Actions.

This is a portfolio demonstration of model evaluation and review tooling. Results on simulated labels measure how well a model recovers the generator's patterns. They do not establish real-world fraud detection performance.

[MIT License](LICENSE)
