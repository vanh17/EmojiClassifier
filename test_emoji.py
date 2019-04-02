import numpy as np
import pytest
from sklearn.metrics import f1_score, accuracy_score

# import our classification class
import emoji

@pytest.fixture(autouse=True)
def test_read_tweet():
    assert 0 == 0

def test_features():
    assert 0 == 0

def test_labels():
    assert 0 == 0

def test_prediction(capsys, min_f1=0.5):
	# truth_file_lines = open("dataset/us_test.labels", encoding='utf8').readlines()
	# gold_keys = np.zeros(len(truth_file_lines), dtype=int)
 #    # populating the dictionary one entry for each emoji that appears in our gold_key
 #    for i in range(len(truth_file_lines)):
 #        emoji_code_gold = int(truth_file_lines[i].replace("\n",""))
 #        gold_keys[i] = emoji_code_gold
    # get texts and labels from the training data
    train_examples = emoji.read_tweet("dataset/us_train.text", "dataset/us_train.labels", solver='lbfgs')
    train_labels, train_texts = zip(*train_examples)

    # get texts and labels from the development data 
    # The organizer provide the test set for development process
    # to evaluate the whole system we need to submit the output_label_text file to their website
    devel_examples = emoji.read_tweet("dataset/us_test.text", "dataset/us_test.labels")
    devel_labels, devel_texts = zip(*devel_examples)

    # create the feature extractor and label encoder
    to_features = emoji.TextToFeatures(train_texts)
    to_labels = emoji.TextToLabels(train_labels)

    # train the classifier on the training data
    classifier = emoji.Classifier()
    classifier.train(to_features(train_texts), to_labels(train_labels))

    # make predictions on the development data
    predicted_indices = classifier.predict(to_features(devel_texts))

    # measure performance of predictions
    devel_indices = to_labels(devel_labels)
    f1 = f1_score(devel_indices, predicted_indices, average="macro")
    # accuracy = accuracy_score(devel_indices, predicted_indices)

    # print out performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} macro-F1 on dataset development data"
            print(msg.format(f1))

    # make sure that performance is adequate
    assert f1 > min_f1

@pytest.mark.xfail
def test_very_accurate_prediction():
    test_prediction(capsys=None, min_f1=0.7)
