"""
logger.py

This file configures logging for the backend pipeline.

Logs are written to:
- terminal/console
- logs/pipeline.log file

Logs help track:
- query received
- intent detected
- retrieval counts
- cold start decisions
- evaluation metrics
- response type
"""

import logging
import os

# Create logs folder if it does not exist
os.makedirs("logs", exist_ok=True)

# Standard log format used across project
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8")
    ]
)


def get_logger(name: str):
    # Returns logger object for a specific module
    return logging.getLogger(name)