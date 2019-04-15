import numpy as np
import pytest
from sklearn.metrics import f1_score, accuracy_score
import io

# import our classification class
import emoji
import rnn

def to_file(result: np.ndarray, full_path_to_output_file: str):
    with io.open(full_path_to_output_file, "w") as output:
        for r in result:
            output.write(str(r) + "\n")

# def test_prediction_ngram(capsys, min_f1=0.36):
# 	# truth_file_lines = open("dataset/us_test.labels", encoding='utf8').readlines()
# 	# gold_keys = np.zeros(len(truth_file_lines), dtype=int)
#  #    # populating the dictionary one entry for each emoji that appears in our gold_key
#  #    for i in range(len(truth_file_lines)):
#  #        emoji_code_gold = int(truth_file_lines[i].replace("\n",""))
#  #        gold_keys[i] = emoji_code_gold
#     # get texts and labels from the training data
#     train_examples = emoji.read_tweet("dataset/us_train.text", "dataset/us_train.labels")
#     train_labels, train_texts = zip(*train_examples)

#     # get texts and labels from the development data 
#     # The organizer provide the test set for development process called trial
#     # to evaluate the whole system we need to submit the output_label_text file to their website
#     devel_examples = emoji.read_tweet("dataset/us_test.text", "dataset/us_test.labels")
#     devel_labels, devel_texts = zip(*devel_examples)

#     # create the feature extractor and label encoder
#     to_features = emoji.TextToFeatures(train_texts)
#     to_labels = emoji.TextToLabels(train_labels)

#     # train the classifier on the training data
#     classifier = emoji.Classifier()
#     classifier.train(to_features(train_texts), to_labels(train_labels))

#     # make predictions on the development data
#     predicted_indices = classifier.predict(to_features(devel_texts))

#     # measure performance of predictions
#     devel_indices = to_labels(devel_labels)
#     f1 = f1_score(devel_indices, predicted_indices, average="macro")
#     accuracy = accuracy_score(devel_indices, predicted_indices)

#     print("N-gram classifier results:")
#     # print out performance
#     if capsys is not None:
#         with capsys.disabled():
#             msg = "\n{:.1%} macro-F1 on dataset development data"
#             print(msg.format(f1))
#             msg = "\n{:.1%} accuracy on dataset development data"
#             print(msg.format(accuracy))

#     # make sure that performance is adequate
#     assert f1 > min_f1
#     #making the output file so that we can use that for the final testing
#     # test_tweet = emoji.read_test_tweets("dataset/us_test.text")
#     # predicted_indices_gold = classifier.predict(to_features(test_tweet))
#     # to_file(predicted_indices_gold)

train_labels, train_texts = rnn.read_tweet("dataset/us_train.text", "dataset/us_train.labels")
classifier = rnn.RNN()
# classifier.train(train_texts, train_labels)

def test_prediction_lstm_dev(capsys, min_f1=0.30):
    # truth_file_lines = open("dataset/us_test.labels", encoding='utf8').readlines()
    # gold_keys = np.zeros(len(truth_file_lines), dtype=int)
    # get texts and labels from the training data
    # train_labels, train_texts = rnn.read_tweet("dataset/us_train.text", "dataset/us_train.labels")

    # get texts and labels from the development data 
    # The organizer provide the test set for development process called trial
    # to evaluate the whole system we need to submit the output_label_text file to their website
    devel_labels, devel_texts = rnn.read_tweet("dataset/us_trial.text", "dataset/us_trial.labels")

    # train the classifier on the training data
    # classifier = rnn.RNN()
    # classifier.train(train_texts, train_labels)

    # make predictions on the development data
    predicted_indices = classifier.predict(devel_texts)

    # measure performance of predictions
    devel_indices = classifier.to_labels(devel_labels)
    f1 = f1_score(devel_indices, predicted_indices, average="macro")
    accuracy = accuracy_score(devel_indices, predicted_indices)

    print("LSTM classifier dev results:")
    # print out performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} macro-F1 on dataset development data"
            print(msg.format(f1))
            msg = "\n{:.1%} accuracy on dataset development data"
            print(msg.format(accuracy))

    # make sure that performance is adequate
    assert f1 > min_f1
    #making the output file so that we can use that for the final testing
    # test_tweet = emoji.read_test_tweets("dataset/us_test.text")
    # predicted_indices_gold = classifier.predict(to_features(test_tweet))
    # to_file(predicted_indices_gold)

def test_prediction_lstm_test(capsys, min_f1=0.30):
    # truth_file_lines = open("dataset/us_test.labels", encoding='utf8').readlines()
    # gold_keys = np.zeros(len(truth_file_lines), dtype=int)
    # get texts and labels from the training data

    # get texts and labels from the development data 
    # The organizer provide the test set for development process called trial
    # to evaluate the whole system we need to submit the output_label_text file to their website
    devel_labels, devel_texts = rnn.read_tweet("dataset/us_test.text", "dataset/us_test.labels")

    # train the classifier on the training data
    # classifier = rnn.RNN()
    # classifier.train(train_texts, train_labels)

    # make predictions on the development data
    predicted_indices = classifier.predict(devel_texts)

    # measure performance of predictions
    devel_indices = classifier.to_labels(devel_labels)
    f1 = f1_score(devel_indices, predicted_indices, average="macro")
    accuracy = accuracy_score(devel_indices, predicted_indices)

    print("LSTM classifier test results:")
    # print out performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} macro-F1 on dataset development data"
            print(msg.format(f1))
            msg = "\n{:.1%} accuracy on dataset development data"
            print(msg.format(accuracy))

    # make sure that performance is adequate
    assert f1 > min_f1
    #making the output file so that we can use that for the final testing
    # test_tweet = emoji.read_test_tweets("dataset/us_test.text")
    # predicted_indices_gold = classifier.predict(to_features(test_tweet))
    # to_file(predicted_indices_gold)