# MessageGuard 📩
### Intelligent Message Classification & Analysis Using Machine Learning

MessageGuard is a machine learning-based SMS classification and analysis platform designed to identify unsolicited messages and distinguish them from legitimate messages.

The application uses three machine learning algorithms — Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM) — to classify SMS messages as **Spam** or **Ham**.

It also provides model comparison, explainable message analysis, suspicious pattern detection, performance evaluation, prediction history, batch classification, and robustness testing through an interactive Streamlit interface.

---

## 🎯 Project Objectives

- Develop a machine learning system for SMS spam detection.
- Apply Natural Language Processing techniques to process textual data.
- Compare multiple classification algorithms.
- Address class imbalance and evaluate its impact on model performance.
- Provide interpretable predictions and model comparisons.
- Build an interactive application for individual and batch message classification.

## ✨ Key Features

### 1. Message Classification
- Classifies messages as Spam or Ham.
- Supports three machine learning algorithms.
- Allows users to select Baseline or Balanced models.
- Provides a comparison mode to evaluate predictions from all three models.

### 2. Explainable Message Analysis
- Analyses model evidence behind predictions.
- Highlights important words and phrases.
- Provides insights into features contributing towards classification.

### 3. Suspicious Pattern Analysis
- Identifies potentially suspicious message patterns.
- Detects reward-related language, urgency, financial claims, suspicious links, and requests for personal information.
- Provides supplementary rule-based analysis alongside ML predictions.

### 4. Model Performance Laboratory
- Displays Accuracy, Precision, Recall, and F1-score.
- Compares Baseline and Balanced experiments.
- Provides visual model performance comparisons.
- Supports evaluation of different classification algorithms.

### 5. Prediction History
- Stores previous classification results locally using SQLite.
- Allows users to search and filter prediction records.
- Supports reviewing previous classifications.

### 6. Batch Message Analysis
- Accepts CSV files containing multiple SMS messages.
- Classifies messages in batches.
- Provides prediction summaries and downloadable results.

### 7. Robustness Testing
- Evaluates model behaviour under variations in message formatting.
- Tests uppercase/lowercase changes, extra spaces, punctuation, and common character substitutions.

---

## 🧠 Machine Learning Methodology

The project follows a supervised machine learning classification pipeline.

### Dataset

The project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Original dataset statistics:

| Description | Count |
|---|---:|
| Total messages | 5,572 |
| Ham messages | 4,825 |
| Spam messages | 747 |
| Exact duplicate rows | 403 |
| Messages after duplicate removal | 5,169 |

After preprocessing:

| Class | Count |
|---|---:|
| Ham | 4,516 |
| Spam | 653 |
| Total | 5,169 |

### Data Preprocessing

The following preprocessing operations were performed:

- Removed exact duplicate records.
- Converted text to lowercase.
- Removed punctuation and unnecessary special characters.
- Normalized whitespace.
- Encoded labels:
  - Ham = 0
  - Spam = 1

### Exploratory Data Analysis

EDA was performed to understand:

- Class distribution and imbalance.
- Message length distribution.
- Frequently occurring words in Spam messages.
- Frequently occurring words in Ham messages.

### Train-Test Split

The dataset was divided using an 80:20 stratified train-test split.

| Dataset | Records |
|---|---:|
| Training | 4,135 |
| Testing | 1,034 |
| Total | 5,169 |

Stratification was used to preserve the class distribution across both datasets.

### TF-IDF Vectorization

Term Frequency–Inverse Document Frequency (TF-IDF) was used to convert textual messages into numerical feature vectors.

Configuration:

- `ngram_range=(1, 2)`
- `max_features=5000`
- `min_df=2`

The vectorizer was fitted on training data and applied to testing data to avoid data leakage.

### Class Imbalance Handling

The original dataset contains substantially more Ham messages than Spam messages.

Random oversampling was applied to the training dataset to balance the two classes.

| Class | Original Training Data | Balanced Training Data |
|---|---:|---:|
| Ham | 3,613 | 3,613 |
| Spam | 522 | 3,613 |
| Total | 4,135 | 7,226 |

The testing dataset was kept unchanged to ensure fair evaluation.

---

## 🤖 Machine Learning Algorithms

Three classification algorithms were implemented and evaluated.

### 1. Multinomial Naive Bayes
A probabilistic classification algorithm based on Bayes' theorem. It is commonly used for text classification tasks and works with non-negative feature representations such as TF-IDF.

### 2. Logistic Regression
A supervised classification algorithm that estimates the probability of a message belonging to a particular class using a learned linear decision function.

### 3. Linear Support Vector Machine (SVM)
A supervised learning algorithm that identifies a decision boundary separating different classes by maximizing the margin between them.

Both Baseline and Balanced versions were trained for all three algorithms, resulting in six trained models.

---

## 📊 Model Evaluation Results

The following results were obtained on the held-out test dataset.

| Model | Experiment | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Naive Bayes | Baseline | 96.81% | 100.00% | 74.81% | 85.59% |
| Naive Bayes | Balanced | 97.68% | 93.50% | 87.79% | 90.55% |
| Logistic Regression | Baseline | 96.71% | 98.99% | 74.81% | 85.22% |
| Logistic Regression | Balanced | 97.49% | 88.89% | 91.60% | 90.23% |
| Linear SVM | Baseline | 97.97% | 97.41% | 86.26% | 91.50% |
| Linear SVM | Balanced | **98.36%** | **95.97%** | **90.84%** | **93.33%** |

### Best Performing Model

The Balanced Linear SVM achieved the highest test accuracy and F1-score among the six model configurations.

It achieved:

- Accuracy: 98.36%
- Precision: 95.97%
- Recall: 90.84%
- F1-score: 93.33%

The results demonstrate the impact of handling class imbalance, particularly on Spam detection recall.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF Vectorization |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib / Plotly |
| Model Persistence | Joblib |
| Local Database | SQLite |
| Version Control | Git and GitHub |

---

## 📁 Project Structure

```text
unsolicited-message-classification/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── SMSSpamCollection
│   ├── processed_sms.csv
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── inspect_dataset.py
│   ├── eda.py
│   ├── analyze_duplicates.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── vectorization.py
│   ├── train_models.py
│   ├── experiment_balanced.py
│   └── compare_experiments.py
│
├── models/
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── naive_bayes_balanced.joblib
│   ├── logistic_regression_balanced.joblib
│   └── svm_balanced.joblib
│
└── outputs/
    ├── class_distribution.png
    ├── message_length.png
    ├── spam_word_frequency.png
    ├── ham_word_frequency.png
    ├── model_comparison.csv
    ├── balanced_model_comparison.csv
    ├── final_model_comparison.csv
    ├── model_comparison.png
    └── roc_comparison.png
```

*Note: The structure above highlights the main project files. Additional application modules and output files may also be present.*

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- VS Code or another Python-compatible IDE.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd unsolicited-message-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

---

## 🚀 Usage

1. Launch MessageGuard.
2. Navigate to the Message Classification page.
3. Enter an SMS message.
4. Select Baseline or Balanced experiment.
5. Choose an individual classifier or Compare All Models.
6. Run the classification.
7. Explore explainable analysis and suspicious pattern detection.
8. Visit the Performance Laboratory to compare model metrics.
9. Use Prediction History to review saved predictions.
10. Upload CSV files for batch message classification.
11. Explore Robustness Testing to examine model behaviour under text variations.

---

## 🔬 Future Scope

Potential future improvements include:

- Support for additional languages.
- Advanced NLP and transformer-based classification models.
- Improved handling of obfuscated and misspelled spam messages.
- Enhanced explainability techniques.
- Real-time message analysis integrations.
- Further evaluation on larger and more diverse datasets.

---

## 🎓 Academic Project Information

**Project Title:** Unsolicited Message Detection and Classification Using Machine Learning

**Application Name:** MessageGuard

**Course:** Machine Learning (AI 304 / CS 304)

**Project Type:** Course-Based Machine Learning Mini Project

---

## 👩‍💻 Contributors

**Vaniya Ubaid:** 24STUCHH010251
**Tanushree Tanirika:** 24STUCHH010215

---

## 📜 License

This project is developed for academic and educational purposes.

The SMS Spam Collection dataset is provided by the UCI Machine Learning Repository. Refer to its original source for dataset usage and licensing information.