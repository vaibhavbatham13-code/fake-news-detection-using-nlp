"""
Tests for the text cleaning, features, model pipeline and helpers.

Run with either:
    python -m pytest tests/
    python tests/test_utils.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sample_data import SAMPLE_DATA
from utilis import (
    build_pipeline,
    clean_text,
    fact_check_lookup,
    fake_probability,
    style_features,
    verdict,
)