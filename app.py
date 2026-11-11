import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Page configuration
st.set_page_config(
    page_title="Unsolicited Message Detection",
    page_icon="📩",
    layout="wide"
)

# Load models once and cache them
@st.cache_resource
def load_models():
    models = {}

    model_files = {
        "Baseline": {
            "Naive Bayes": "naive_bayes.joblib",
            "Logistic Regression": "logistic_regression.joblib",
            "SVM": "svm.joblib"
        },
        "Balanced": {
            "Naive Bayes": "naive_bayes_balanced.joblib",
            "Logistic Regression": "logistic_regression_balanced.joblib",
            "SVM": "svm_balanced.joblib"
        }
    }

    for version, model_set in model_files.items():
        models[version] = {}

        for name, filename in model_set.items():
            path = MODEL_DIR / filename

            if path.exists():
                models[version][name] = joblib.load(path)

    return models


models = load_models()

# Header
st.title("Unsolicited Message Detection")
st.write(
    "Detect unsolicited messages using machine learning "
    "and compare different classification algorithms."
)

st.divider()

# Model configuration
st.subheader("Model Configuration")

col1, col2 = st.columns(2)

with col1:
    version = st.selectbox(
        "Choose model experiment",
        ["Baseline", "Balanced"]
    )

with col2:
    selected_model = st.selectbox(
        "Choose classifier",
        [
            "Compare All Models",
            "Naive Bayes",
            "Logistic Regression",
            "SVM"
        ]
    )

# Message input
st.subheader("Enter Your Message")

message = st.text_area(
    "SMS Message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

# Prediction
if st.button("Classify Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message first.")

    elif version not in models or not models[version]:
        st.error("No trained models were found. Check your models folder.")

    else:
        if selected_model == "Compare All Models":
            selected_models = models[version]
        else:
            selected_models = {
                selected_model: models[version].get(selected_model)
            }

        missing = [
            name for name, model in selected_models.items()
            if model is None
        ]

        if missing:
            st.error(
                "Missing model files for: " + ", ".join(missing)
            )

        else:
            results = []

            for name, model in selected_models.items():
                prediction = model.predict([message])[0]

                label = "Spam" if int(prediction) == 1 else "Ham"

                results.append({
                    "Model": name,
                    "Prediction": label
                })

            st.divider()
            st.subheader("Classification Results")

            if selected_model == "Compare All Models":
                cols = st.columns(len(results))

                for col, result in zip(cols, results):
                    with col:
                        st.markdown(f"**{result['Model']}**")

                        if result["Prediction"] == "Spam":
                            st.error("🚨 SPAM")
                        else:
                            st.success("✅ HAM")

                st.subheader("Prediction Comparison")

                result_df = pd.DataFrame(results)
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                spam_count = sum(
                    r["Prediction"] == "Spam" for r in results
                )

                st.info(
                    f"{spam_count} out of {len(results)} models "
                    "classified this message as Spam."
                )

            else:
                result = results[0]

                if result["Prediction"] == "Spam":
                    st.error("This message is classified as SPAM.")
                else:
                    st.success("This message is classified as HAM.")

                st.write(f"Model used: **{result['Model']}**")
                st.write(f"Experiment: **{version}**")