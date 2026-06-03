"""
intent.py

This file detects user intent using a pretrained NLP transformer model.

Intent detection is done using zero-shot classification.

Supported intents:
- text
- image
- audio
- video

Why this approach:
- It does not depend on simple keyword matching.
- It uses a pretrained NLP model.
- It understands the meaning of the full query.
- No training dataset is required for the current project.

Model used:
- facebook/bart-large-mnli

How it works:
- The user query is compared with intent labels.
- The intent with the highest semantic score is selected.
"""

from transformers import pipeline

from app.logger import get_logger


# Create logger for intent detection.
log = get_logger("intent")


# Load zero-shot classification model.
# This may take time during first startup.
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# Candidate labels with clear intent meaning.
# These labels are semantic descriptions, not only single keywords.
CANDIDATE_LABELS = [
    "text explanation or description request",
    "image or photo request",
    "audio or sound playback request",
    "video or clip playback request",
]


# Mapping model labels to pipeline intent names.
LABEL_TO_INTENT = {
    "text explanation or description request": "text",
    "image or photo request": "image",
    "audio or sound playback request": "audio",
    "video or clip playback request": "video",
}


def classify_intent(query: str):
    """
    Classifies user query into one of:
    text, image, audio, video.

    Uses transformer-based zero-shot classification.
    """

    try:
        result = classifier(
            query,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template="The user wants {}."
        )

        best_label = result["labels"][0]
        best_score = float(result["scores"][0])

        intent = LABEL_TO_INTENT.get(best_label, "text")

        log.info(
            "NLP intent detected | query=%s | intent=%s | label=%s | score=%s",
            query,
            intent,
            best_label,
            round(best_score, 4),
        )

        return intent

    except Exception as e:
        log.warning(
            "NLP intent detection failed | query=%s | error=%s",
            query,
            str(e),
        )

        return "text"