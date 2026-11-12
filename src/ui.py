from pathlib import Path
import hashlib
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.batch_analysis import classify_csv, read_message_csv
from src.database import (
    clear_history,
    delete_prediction,
    get_history,
    save_prediction,
)
from src.evaluation import evaluate_test_models, load_comparison
from src.explainability import explain_message
from src.prediction import (
    MODEL_FILES,
    label_name,
    majority_prediction,
    predict_message,
    predict_models,
)
from src.robustness import test_robustness
from src.suspicious_patterns import analyze_suspicious_patterns


HISTORY_PATH = Path("outputs") / "messageguard_history.sqlite3"
MESSAGE_KEYS = {
    "Message Classification": "home_message",
    "Explainable Analysis": "analysis_message",
    "Suspicious Patterns": "patterns_message",
    "Robustness Testing": "robustness_message",
}


def _sync_message(key: str) -> None:
    st.session_state["current_message"] = st.session_state[key]


def _message_input(label: str, key: str, height: int = 170) -> str:
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get("current_message", "")
    return st.text_area(
        label,
        key=key,
        height=height,
        placeholder="Paste or type an SMS message here…",
        on_change=_sync_message,
        args=(key,),
    )


def _title(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted">{description}</p>', unsafe_allow_html=True)


def _available_models(
    models: dict[str, dict[str, Any]], experiment: str
) -> dict[str, Any]:
    return models.get(experiment, {})


def _classify_page(
    models: dict[str, dict[str, Any]], base_dir: Path
) -> None:
    _title(
        "MessageGuard · Classifier",
        "Message classification",
        "Classify a message with an existing model, or compare the three classifiers in one experiment.",
    )
    first, second = st.columns(2)
    with first:
        experiment = st.selectbox(
            "Model experiment",
            list(MODEL_FILES),
            key="home_experiment",
            help="Baseline uses the original training distribution; Balanced uses oversampled training data.",
        )
    with second:
        selected = st.selectbox(
            "Classifier",
            ["Compare All Models", *MODEL_FILES[experiment].keys()],
            key="home_classifier",
        )

    message = _message_input("Message to classify", "home_message")
    save_to_history = st.checkbox(
        "Save this prediction to local history", value=True
    )
    classify_col, clear_col, _ = st.columns([1.2, 1, 5])
    with classify_col:
        run = st.button("Classify message", type="primary", width="stretch")
    with clear_col:
        clear = st.button(
            "Clear input",
            width="stretch",
            on_click=lambda: st.session_state.update(
                {"home_message": "", "current_message": "", "last_result": None}
            ),
        )
    if clear:
        st.info("Message input cleared.")
    if run:
        if not message.strip():
            st.session_state["last_result"] = None
            st.warning("Enter a message before starting classification.")
        else:
            selected_models = _available_models(models, experiment)
            if selected != "Compare All Models":
                selected_models = (
                    {selected: selected_models[selected]}
                    if selected in selected_models
                    else {}
                )
            if not selected_models:
                st.session_state["last_result"] = None
                st.error("No saved models are available for this selection.")
            else:
                results = predict_models(selected_models, message)
                majority = majority_prediction(results) if len(results) > 1 else None
                st.session_state["current_message"] = message
                st.session_state["last_result"] = {
                    "experiment": experiment,
                    "selected": selected,
                    "results": results,
                    "majority": majority,
                }
                if save_to_history and (
                    majority is not None or len(results) == 1
                ):
                    history_label = (
                        majority if majority is not None else next(iter(results.values()))
                    )
                    save_prediction(
                        message,
                        history_label,
                        selected,
                        experiment,
                        base_dir / HISTORY_PATH,
                    )
                elif save_to_history:
                    st.warning(
                        "There was no majority result to save; this comparison was not added to history."
                    )
    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Result")
        st.caption(
            f"{result['experiment']} experiment · "
            f"{'All classifiers' if result['selected'] == 'Compare All Models' else result['selected']}"
        )
        columns = st.columns(len(result["results"]))
        for column, (name, prediction) in zip(columns, result["results"].items()):
            with column:
                st.metric(name, prediction)
        if len(result["results"]) > 1:
            labels = list(result["results"].values())
            if len(set(labels)) == 1:
                st.success("All available models agree.")
            else:
                st.warning("The models disagree; treat the outputs as uncertain.")
            if result["majority"] is None:
                st.info("The available models are tied; no majority suggestion is available.")
            else:
                st.info(
                    f"Majority suggestion: **{result['majority']}**. "
                    "This is a vote among these models, not a guaranteed truth."
                )
        st.caption(
            "Automated predictions can be wrong. Verify suspicious requests using a trusted channel."
        )


def _explain_page(models: dict[str, dict[str, Any]]) -> None:
    _title(
        "MessageGuard · Model evidence",
        "Explainable message analysis",
        "Inspect how observed TF-IDF terms align with the trained classifier's learned parameters.",
    )
    experiment = st.selectbox("Experiment", list(MODEL_FILES), key="explain_experiment")
    options = list(_available_models(models, experiment))
    if not options:
        st.error("No saved models are available for this experiment.")
        return
    model_name = st.selectbox("Classifier", options, key="explain_model")
    message = _message_input("Message to explain", "analysis_message")
    if st.button("Analyze model evidence", type="primary"):
        if not message.strip():
            st.warning("Enter a message to explain.")
        else:
            model = models[experiment][model_name]
            result = explain_message(model, message)
            st.session_state["current_message"] = message
            st.subheader("Prediction")
            st.metric(model_name, label_name(predict_message(model, message)))
            st.caption(f"Evidence method: {result['method']}")
            if not result["terms"]:
                st.info(
                    "No message terms are present in this model's fitted TF-IDF vocabulary. "
                    "The prediction is based on the model's remaining learned behavior, "
                    "not term evidence shown here."
                )
                return
            spam_col, ham_col = st.columns(2)
            with spam_col:
                st.markdown("#### Terms aligned with Spam")
                if result["spam_terms"]:
                    st.dataframe(
                        pd.DataFrame(result["spam_terms"]).rename(
                            columns={"term": "Observed term", "contribution": "Contribution"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("No observed terms with positive Spam contribution.")
            with ham_col:
                st.markdown("#### Terms aligned with Ham")
                if result["ham_terms"]:
                    st.dataframe(
                        pd.DataFrame(result["ham_terms"]).rename(
                            columns={"term": "Observed term", "contribution": "Contribution"}
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("No observed terms with negative Spam contribution.")
            st.caption(
                "Contributions are model-specific associations for observed terms, not causal proof. "
                "They do not explain out-of-vocabulary terms or guarantee why a message was classified."
            )


def _patterns_page() -> None:
    _title(
        "MessageGuard · Supplementary checks",
        "Suspicious pattern analysis",
        "Rule-based indicators provide context alongside—not instead of—the machine-learning result.",
    )
    message = _message_input("Message to inspect", "patterns_message")
    if st.button("Check for indicators", type="primary"):
        if not message.strip():
            st.warning("Enter a message to inspect.")
        else:
            findings = analyze_suspicious_patterns(message)
            st.session_state["current_message"] = message
            if findings:
                for finding in findings:
                    st.markdown(
                        f"""
                        <div class="panel">
                          <strong>{finding['indicator']}</strong><br>
                          <span class="muted">Matched: {finding['matched_text']}</span><br>
                          {finding['explanation']}
                        </div><br>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No indicators from this rule set were detected.")
            st.info(
                "These indicators are heuristics, not proof of fraud. A message without "
                "these patterns may still be unsafe."
            )


def _performance_page(
    models: dict[str, dict[str, Any]], base_dir: Path
) -> None:
    _title(
        "MessageGuard · Model evaluation",
        "Performance laboratory",
        "Compare reported held-out test metrics and inspect test-set errors and ranking behavior.",
    )
    comparison_path = base_dir / "outputs" / "final_model_comparison.csv"
    try:
        comparison = load_comparison(comparison_path)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(str(error))
        return
    test_metrics = comparison[comparison["Dataset"].str.lower() == "test"].copy()
    experiments = list(MODEL_FILES)
    experiment = st.selectbox("Experiment", experiments, key="lab_experiment")
    experiment_data = test_metrics[
        test_metrics["Experiment"].str.lower() == experiment.lower()
    ].copy()
    if experiment_data.empty:
        st.warning("No test metrics are recorded for this experiment.")
        return
    metric = st.selectbox(
        "Compare models by",
        ["F1 Score", "Accuracy", "Precision", "Recall"],
        key="lab_metric",
    )
    best = experiment_data.loc[experiment_data[metric].idxmax()]
    st.markdown(
        f'<div class="panel"><span class="eyebrow">Best by test {metric}</span>'
        f'<h3>{best["Model"]}</h3><span class="muted">{metric}: '
        f'{best[metric]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    for column, metric_name in zip(
        metric_cols, ["Accuracy", "Precision", "Recall", "F1 Score"]
    ):
        with column:
            st.metric(metric_name, f"{best[metric_name]:.3f}")

    chart_data = experiment_data.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]
    st.subheader("Test-set metrics")
    st.bar_chart(chart_data, y=metric, color="#7862c8")
    st.dataframe(
        experiment_data[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
        .sort_values(metric, ascending=False)
        .style.format(
            {name: "{:.3f}" for name in ["Accuracy", "Precision", "Recall", "F1 Score"]}
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Confusion matrices and ROC")
    available = _available_models(models, experiment)
    if not available:
        st.warning("Saved model files are unavailable; test-set plots cannot be generated.")
        return
    try:
        evaluated = evaluate_test_models(
            available, base_dir / "dataset" / "test.csv"
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(str(error))
        return

    figure, axes = plt.subplots(1, len(evaluated), figsize=(5 * len(evaluated), 4))
    if len(evaluated) == 1:
        axes = [axes]
    for axis, (name, result) in zip(axes, evaluated.items()):
        matrix = result["confusion_matrix"]
        axis.imshow(matrix, cmap="Purples")
        axis.set_title(name)
        axis.set_xticks([0, 1], labels=["Ham", "Spam"])
        axis.set_yticks([0, 1], labels=["Ham", "Spam"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption("Confusion matrices are computed from the provided held-out test.csv.")

    roc_figure, roc_axis = plt.subplots(figsize=(7, 4))
    has_roc = False
    for name, result in evaluated.items():
        if "roc" in result:
            curve = result["roc"]
            roc_axis.plot(
                curve["fpr"],
                curve["tpr"],
                label=f"{name} (AUC {curve['auc']:.3f})",
            )
            has_roc = True
    if has_roc:
        roc_axis.plot([0, 1], [0, 1], linestyle="--", color="#aaa")
        roc_axis.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="ROC comparison on held-out test data",
        )
        roc_axis.legend()
        st.pyplot(roc_figure, width="stretch")
    else:
        st.info("The selected models do not expose score outputs for an ROC curve.")
    plt.close(roc_figure)

    st.divider()
    st.subheader("Baseline vs Balanced")
    baseline = test_metrics[test_metrics["Experiment"].str.lower() == "baseline"]
    balanced = test_metrics[test_metrics["Experiment"].str.lower() == "balanced"]
    cross = baseline.merge(balanced, on="Model", suffixes=(" (Baseline)", " (Balanced)"))
    if not cross.empty:
        st.dataframe(cross, width="stretch", hide_index=True)


def _history_page(base_dir: Path) -> None:
    _title(
        "MessageGuard · Local records",
        "Prediction history",
        "Locally stored predictions are optional and remain in this project's SQLite database.",
    )
    database_path = base_dir / HISTORY_PATH
    filters = st.columns([2, 1, 1])
    search = filters[0].text_input("Search message", key="history_search")
    label = filters[1].selectbox("Label", ["All", "Spam", "Ham"], key="history_label")
    model = filters[2].selectbox(
        "Model", ["All", "Naive Bayes", "Logistic Regression", "SVM", "Compare All Models"],
        key="history_model",
    )
    history = get_history(database_path, search, label, model)
    if history.empty:
        st.info("No predictions match these filters.")
    else:
        st.dataframe(history, width="stretch", hide_index=True)
        st.download_button(
            "Export filtered history as CSV",
            history.to_csv(index=False).encode("utf-8"),
            file_name="messageguard_history.csv",
            mime="text/csv",
        )
        records = {
            f"#{row.id} · {row.predicted_label} · {row.message[:70]}": int(row.id)
            for row in history.itertuples()
        }
        selected_record = st.selectbox(
            "Record to delete", ["Select a record", *records], key="delete_history_record"
        )
        delete_col, clear_col = st.columns(2)
        with delete_col:
            if st.button(
                "Delete selected record",
                disabled=selected_record == "Select a record",
            ):
                if delete_prediction(records[selected_record], database_path):
                    st.success("Record deleted.")
                    st.rerun()
        with clear_col:
            if st.button("Clear all history"):
                removed = clear_history(database_path)
                st.success(f"Cleared {removed} history record(s).")
                st.rerun()


def _batch_page(models: dict[str, dict[str, Any]]) -> None:
    _title(
        "MessageGuard · Bulk workflow",
        "Batch message analysis",
        "Upload a CSV and classify its messages locally. Rows without message text are retained as not classified.",
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.caption("Choose a CSV with a header row and a column containing message text.")
        return
    upload_bytes = uploaded.getvalue()
    try:
        frame = read_message_csv(upload_bytes)
    except ValueError as error:
        st.error(str(error))
        return
    message_column = st.selectbox("Message column", list(frame.columns))
    first, second = st.columns(2)
    experiment = first.selectbox("Experiment", list(MODEL_FILES), key="batch_experiment")
    selected = second.selectbox(
        "Classifier",
        ["Compare All Models", *MODEL_FILES[experiment].keys()],
        key="batch_classifier",
    )
    signature = (
        hashlib.sha256(upload_bytes).hexdigest(),
        message_column,
        experiment,
        selected,
    )
    if st.button("Classify CSV", type="primary"):
        selected_models = _available_models(models, experiment)
        if selected != "Compare All Models":
            selected_models = (
                {selected: selected_models[selected]}
                if selected in selected_models
                else {}
            )
        try:
            result = classify_csv(frame, message_column, selected_models)
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["batch_result"] = {
            "signature": signature,
            "data": result,
        }
    batch_result = st.session_state.get("batch_result")
    if batch_result and batch_result["signature"] == signature:
        result = batch_result["data"]
        st.subheader("Classified messages")
        st.dataframe(result, width="stretch", hide_index=True)
        if "Prediction" in result.columns:
            counts = result["Prediction"].value_counts()
            spam_col, ham_col, other_col = st.columns(3)
            spam_col.metric("Spam", int(counts.get("Spam", 0)))
            ham_col.metric("Ham", int(counts.get("Ham", 0)))
            other_col.metric("Not classified", int(counts.get("Not classified", 0)))
        st.download_button(
            "Download classified results",
            result.to_csv(index=False).encode("utf-8"),
            file_name="messageguard_classified.csv",
            mime="text/csv",
        )


def _robustness_page(models: dict[str, dict[str, Any]]) -> None:
    _title(
        "MessageGuard · Input sensitivity",
        "Robustness testing",
        "Compare one model's output on controlled text variations. The saved model and test dataset are not modified.",
    )
    experiment = st.selectbox(
        "Experiment", list(MODEL_FILES), key="robustness_experiment"
    )
    options = list(_available_models(models, experiment))
    if not options:
        st.error("No saved models are available for this experiment.")
        return
    model_name = st.selectbox("Classifier", options, key="robustness_model")
    message = _message_input("Message to vary", "robustness_message")
    if st.button("Run robustness checks", type="primary"):
        if not message.strip():
            st.warning("Enter a message to test.")
        else:
            result = test_robustness(models[experiment][model_name], message)
            st.session_state["current_message"] = message
            st.dataframe(result, width="stretch", hide_index=True)
            stable = result["Matches original"].mean()
            st.metric("Variations matching the original prediction", f"{stable:.0%}")
            st.caption(
                "Text transformations are diagnostic examples, not a formal robustness guarantee."
            )


def render_page(
    page: str,
    models: dict[str, dict[str, Any]],
    base_dir: Path,
) -> None:
    if page == "Message Classification":
        _classify_page(models, base_dir)
    elif page == "Explainable Analysis":
        _explain_page(models)
    elif page == "Suspicious Patterns":
        _patterns_page()
    elif page == "Performance Laboratory":
        _performance_page(models, base_dir)
    elif page == "Prediction History":
        _history_page(base_dir)
    elif page == "Batch Analysis":
        _batch_page(models)
    elif page == "Robustness Testing":
        _robustness_page(models)
