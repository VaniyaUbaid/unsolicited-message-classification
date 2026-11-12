# MessageGuard

**Intelligent Message Classification & Analysis**

MessageGuard is a local Streamlit application for classifying SMS messages with
the project's existing baseline and balanced TF-IDF pipelines. It includes
model-specific term evidence, separate rule-based pattern indicators,
held-out evaluation, optional SQLite history, CSV batch classification, and
small input-variation checks.

## Run locally

From the project root, activate the existing Python environment (or create one
and install the pinned packages in `requirements.txt`), then run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The application does not call cloud APIs or retrain models. It loads the six
pipelines from `models/` and reads the reported test metrics from
`outputs/final_model_comparison.csv`. Test-set confusion matrices and ROC
curves are calculated from the existing `dataset/test.csv`.

## Pages

- **Message Classification** — select Baseline or Balanced, classify with an
  individual model or compare the three models, and optionally save a result.
- **Explainable Analysis** — inspect observed TF-IDF terms weighted by Naive
  Bayes log-likelihood differences or linear classifier coefficients.
- **Suspicious Patterns** — view heuristic indicators for common pressure,
  reward, URL, sensitive-information, and promotional language.
- **Performance Laboratory** — compare the recorded test metrics, find the
  best model by the selected metric, and inspect held-out confusion matrices
  and ROC curves.
- **Prediction History** — search/filter local records, delete rows, clear the
  history, or export the filtered view.
- **Batch Analysis** — upload a CSV, select the message column, classify rows,
  and download the results.
- **Robustness Testing** — compare predictions on case, whitespace,
  punctuation, spelling, and character-substitution variations.

History is saved in `outputs/messageguard_history.sqlite3`; uncheck the option
on the classification page to avoid saving an individual message. Keep
uploaded messages and history files private if they contain personal data.

Model explanations show associations represented in the fitted model and are
not causal explanations. Rule-based indicators are supplementary checks, not
proof that a message is fraudulent. A model vote is only a suggestion; review
the message and verify requests through trusted channels.

## Tests

Run the focused checks against the saved models and existing data:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```