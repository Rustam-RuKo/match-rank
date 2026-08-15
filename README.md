# MatchRank

MatchRank is an explainable university recommendation engine. It learns relevance from
applicant–university interactions and ranks a candidate set using academic, financial,
geographic, program, language, scholarship, and admissions features.

It includes a transparent weighted baseline, a scikit-learn pointwise learning-to-rank model,
cold-start fallback, local counterfactual explanations, offline ranking metrics, a versioned
FastAPI service, Prometheus-compatible latency metrics, PSI data-drift monitoring, and optional
MLflow experiment logging.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,mlflow]"
matchrank-train --output-dir artifacts --profiles 300 --k 10
matchrank-api
```

Open `http://localhost:8000/docs` for the interactive API. The server can also run before
training; it safely falls back to the rule-based cold-start ranker.

## Train and evaluate

The bundled generator creates deterministic demo data so the entire workflow can be reproduced:

```powershell
matchrank-train --profiles 300 --k 10
Get-Content artifacts/metrics.json
```

Training uses an applicant-level 80/20 holdout, preventing university pairs belonging to one
applicant from leaking across splits. Labels are graded relevance values. Precision and recall
treat relevance >= 2 as positive; NDCG retains the full graded signal. For a real deployment,
replace `generate_labels` with outcomes such as shortlist, application, offer, and enrollment,
while keeping the same grouped split and evaluation contract.

### Reproducible demo result

The default seed and 300 generated profiles produce a 240-profile training set and a 60-profile
applicant-level test set:

| Ranker | Precision@10 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Weighted-filter baseline | 1.0000 | 0.2282 | 0.9716 |
| Learned ranker | 1.0000 | 0.2282 | **0.9921** |

The relevance threshold makes the demo's top-ten precision easy to saturate; graded NDCG is the
more informative result. These are synthetic benchmark values, not claims about real admissions
outcomes.

## API example

```powershell
$body = @{
  applicant_id = "candidate-1"; grade = 88; budget_usd = 35000
  preferred_majors = @("computer science"); preferred_countries = @("Canada")
  languages = @("English"); needs_scholarship = $true; test_score = 1350; top_k = 10
} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/v1/recommendations -Method Post -Body $body -ContentType application/json
```

Operational endpoints are `/health`, `/metrics`, and `/monitoring/drift`. Drift is reported per
engineered feature with Population Stability Index (PSI); 0.2 is the alert threshold after at
least ten observed values. Explanations are model-driven local ablations: each displayed factor
is replaced with its training reference mean and the change in predicted relevance is measured.

## Project layout

- `features.py`: pairwise feature-engineering contract and synthetic relevance function
- `baseline.py`: interpretable weighted-filter baseline
- `ranker.py`: learned ranker, persistence, cold start, and local explanations
- `evaluation.py`: Precision@K, Recall@K, and NDCG@K
- `monitoring.py`: prediction latency and PSI drift
- `api.py`: validated versioned serving API
- `train.py`: reproducible train/evaluate/persist workflow and optional MLflow logging

## Production considerations

The demo intentionally keeps infrastructure local. A production iteration should train from
time-stamped interaction data, split chronologically, audit ranking quality by demographic and
geographic cohorts, calibrate scholarship and admission claims against official data, and never
treat recommendations as admission guarantees.

## Résumé-ready description

**MatchRank — University Recommendation Engine | Python, scikit-learn, FastAPI**

- Developed a learning-to-rank system matching applicants with universities using academic,
  financial, geographic, program, language, scholarship, and admissions features.
- Improved NDCG@10 from 0.9716 to 0.9921 against a weighted-filter baseline across 60 held-out
  applicant profiles in a reproducible synthetic benchmark.
- Served a versioned FastAPI recommendation API with Pydantic validation, cold-start fallback,
  local feature-ablation explanations, latency telemetry, and PSI drift monitoring.
