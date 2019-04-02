from typing import Iterator, Iterable, Tuple, Text, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import io
import numpy as np
from scipy.sparse import spmatrix

NDArray = Union[np.ndarray, spmatrix]

def read_tweet(tweet_path: str, emoji_path: str) -> Iterator[Tuple[Text, Text]]:
	"""Generates (emoji, tweet) tuples from the lines in emoji and tweet files.
	The two files have the same number of lines
    Emoji file contains one emoji per line and tweet file contains one tweet per line. 
    Each line in emoji file will have one emoji out of top 20 most common emoji for tweets in English,
    Here are some examples of emoji in emoji file, each number is related to its rank of being common:
      2
      10
      6
      1
      16
      17
    Here are some examples of tweet in tweet file:
      en Pelham Parkway
      The calm before...... | w/ sofarsounds @user | : B. Hall.......#sofarsounds…
      Just witnessed the great solar eclipse @ Tampa, Florida
      This little lady is 26 weeks pregnant today! Excited for baby Cam to come! @ Springfield,…
      Great road trip views! @ Shartlesville, Pennsylvania
      CHRISTMAS DEALS BUY ANY 3 SMALL POMADES 1.5 OR 1.7 OZ RECEIVE THE F&amp;S COLLECTOR TIN &amp; COMB…
    :param tweet_path: The path of an tweet file, formatted as above.
    :param emoji_path: The path of an emoji file, formatted as above.
    :return: An iterator over (emoji, tweet) tuples.
    """
    tweets = open("dataset/us_test.text", encoding='utf8').readlines()
    emojis = open("dataset/us_test.labels", encoding='utf8').readlines()
	for i in range(len(tweets)):
		yield (emojis[i], tweets[i])

class TextToFeatures:
    def __init__(self, texts: Iterable[Text], tf_idf = 1):
        """Initializes an object for converting texts to features.
        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.
        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").
        :param texts: The training texts.
        """
        # stop_words = ['the', 'be', 'of', 'and', 'a', 'and', 'in', 'that', 'have', 'for', 'not', 'on', 'this', 'with', 'but', 'by', 'from']
        # tuning the parameter, ignore the term that has document frequency more than .17, and ignore term that appears lower than 0.018
        self.vectorizer = CountVectorizer(analyzer = 'char', ngram_range=(1,6), binary = True, max_df = 0.18, min_df = 0.016)
        # fit the traing text to create the feature matrix
        self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names()

    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.
        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        return self.feature_names.index(feature)

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.
        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.
        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.
        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        # apply the Countvector in the initialization for the new sample to find
        # its feature matrix
        return self.vectorizer.transform(texts)

class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.
        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.
        :param labels: The training labels.
        """
        # initialize the lable encoder to start the label list
        self.lbEncoder = preprocessing.LabelEncoder()
        self.lbEncoder.fit(labels)
        # assing the lbEncnder classes to the feature name
        # so that we can access it in the index
        self.feature_names = list(self.lbEncoder.classes_)

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.
        :param label: A label
        :return: The unique integer index associated with the label.
        """
        # the feature_name will be the ndarray, we need to turn it back
        # to normal list so that we can return the index number
        return (self.feature_names).index(label)

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.
        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.
        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        return self.lbEncoder.transform(labels)

class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        self.clfLR = LogisticRegression(random_state=0, multiclass='multinomial')

    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.
        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        # fit (train) the clfLR based on the feature matrix and the labels for each docs
        # in such training dataset
        self.clfLR.fit(features, labels) 


    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.
        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        return self.clfLR.predict(features)
