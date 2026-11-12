from typing import Any

def explain_message(model: Any, message: str, limit: int = 8) -> dict[str, Any]:
    """Rank observed TF-IDF terms by their learned contribution to each class."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a non-empty message to explain.")
    if limit < 1:
        raise ValueError("The explanation limit must be positive.")

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vector = vectorizer.transform([message]).tocsr()
    if vector.nnz == 0:
        return {
            "method": type(classifier).__name__,
            "terms": [],
            "spam_terms": [],
            "ham_terms": [],
        }

    classes = list(classifier.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError("Explanations require the saved Ham=0, Spam=1 labels.")

    if hasattr(classifier, "feature_log_prob_"):
        weights = (
            classifier.feature_log_prob_[classes.index(1)]
            - classifier.feature_log_prob_[classes.index(0)]
        )
        method = "Naive Bayes log-likelihood difference"
    elif hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
        if classes.index(1) == 0:
            weights = -weights
        method = f"{type(classifier).__name__} learned coefficients"
    else:
        raise ValueError(
            f"{type(classifier).__name__} does not expose supported "
            "feature-level explanation parameters."
        )

    feature_names = vectorizer.get_feature_names_out()
    contributions = vector.data * weights[vector.indices]
    terms = [
        {
            "term": str(feature_names[index]),
            "contribution": float(contribution),
        }
        for index, contribution in zip(vector.indices, contributions)
    ]
    terms.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    return {
        "method": method,
        "terms": terms,
        "spam_terms": [
            item for item in terms if item["contribution"] > 0
        ][:limit],
        "ham_terms": [
            item for item in terms if item["contribution"] < 0
        ][:limit],
    }
