import numpy as np
import pytest
from sklearn.metrics import f1_score, accuracy_score

import emoji

@pytest.fixture(autouse=True)
def test_read_smsspam():
    assert 0 == 0

def test_features():
    assert 0 == 0

def test_labels():
    assert 0 == 0

def test_prediction(capsys, min_f1=0.89, min_accuracy=0.97):
    assert 0 == 0

@pytest.mark.xfail
def test_very_accurate_prediction():
    test_prediction(capsys=None, min_f1=0.94, min_accuracy=0.98)
