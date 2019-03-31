import numpy as np
import pytest
from sklearn.metrics import f1_score, accuracy_score

import emoji

@pytest.fixture(autouse=True)
def test_read_smsspam():

def test_features():

def test_labels():

def test_prediction(capsys, min_f1=0.89, min_accuracy=0.97):

@pytest.mark.xfail
def test_very_accurate_prediction():
    test_prediction(capsys=None, min_f1=0.94, min_accuracy=0.98)