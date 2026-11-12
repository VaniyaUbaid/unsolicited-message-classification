import re


PATTERNS = {
    "Prize or reward language": (
        re.compile(
            r"\b(?:winner|won|prize|reward|free gift|claim your|congratulations)\b",
            re.IGNORECASE,
        ),
        "Prize or reward wording can be used to attract a response.",
    ),
    "Urgency or pressure": (
        re.compile(
            r"\b(?:urgent|act now|immediately|limited time|expires today|"
            r"last chance|respond now)\b",
            re.IGNORECASE,
        ),
        "Pressure to act quickly can discourage careful verification.",
    ),
    "Financial reward claim": (
        re.compile(
            r"(?:[$£€]\s?\d|\b(?:cash|bonus|loan|payment|refund|"
            r"bank transfer|money)\b)",
            re.IGNORECASE,
        ),
        "A money-related claim may warrant checking through an official channel.",
    ),
    "Link or shortened URL": (
        re.compile(
            r"(?:https?://|www\.)\S+|\b(?:bit\.ly|tinyurl\.com|t\.co|"
            r"goo\.gl|is\.gd)/\S*",
            re.IGNORECASE,
        ),
        "Links, especially shortened links, conceal their final destination.",
    ),
    "Personal or account information request": (
        re.compile(
            r"\b(?:password|one[- ]time code|otp|pin|account number|"
            r"credit card|verify your account|personal details)\b",
            re.IGNORECASE,
        ),
        "Requests for credentials or sensitive details should be verified independently.",
    ),
    "Promotional emphasis": (
        re.compile(
            r"\b(?:free|buy now|subscribe|special offer|discount|"
            r"call now|txt stop)\b",
            re.IGNORECASE,
        ),
        "Promotional wording or calls to action can be unsolicited.",
    ),
}


def analyze_suspicious_patterns(message: str) -> list[dict[str, str]]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a non-empty message to analyze.")
    findings = []
    for indicator, (pattern, explanation) in PATTERNS.items():
        match = pattern.search(message)
        if match:
            findings.append(
                {
                    "indicator": indicator,
                    "matched_text": match.group(0),
                    "explanation": explanation,
                }
            )
    return findings
