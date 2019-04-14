#Importing libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from typing import Iterator, Iterable, Tuple, Text, Union

#Since fit_to_texts only able to receive list of texts.
#have to create new read_tweet function
def read_tweet(tweet_path: str, emoji_path: str):
    tweets = open(tweet_path, encoding='utf8').readlines()
    emojis = open(emoji_path, encoding='utf8').readlines()
    tweetList = []
    emojiList = []
    for i in range(len(tweets)):
        tweetList.append(tweets[i])
        emojiList.append(emojis[i])
    return (emojiList, tweetList)
#same for reading test tweets
def read_test_tweets(tweet_path: str):
    tweets = open(tweet_path, encoding='utf8').readlines()
    return tweets
    
class RNN:
    #Converting into pandas dataframe and filtering only text and ratings given by the users
    #Will need to handle reading data here somehow
    def __init__(self, train_texts: Iterator[Text], Iterator[Text]):
        """Initalizes a logistic regression classifier.
        """
        #tokenizer to maximum word is 2500, cannot have more than this
        self.tokenizer = Tokenizer(nb_words = 2500, split=' ')
        #this will help us keep track of the words that is frequent
        self.tokenizer.fit_on_texts(data['text'].values)
        #print(tokenizer.word_index)  # To see the dicstionary
        #this will give us the sequence of interger represent for those index create
        #with the fit_on_texts
        X = tokenizer.texts_to_sequences(data['text'].values)
        #pad_sentence will simply make sure that all the representation has the same length
        #of the longest sentence because not all the sentence have the same length
        #without this this can mess up our embedding
        X = pad_sequences(X)
        

    def train(self, features: NDArray, labels: NDArray) -> None:


    def predict(self, features: NDArray) -> NDArray:

embed_dim = 128
lstm_out = 300
batch_size= 32

##Buidling the LSTM network

model = Sequential()
model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout=0.1))
model.add(LSTM(lstm_out, dropout_U=0.1, dropout_W=0.1))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

Y = pd.get_dummies(data['sentiment']).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)

#Here we train the Network.

model.fit(X_train, Y_train, batch_size =batch_size, nb_epoch = 1,  verbose = 5)

# Measuring score and accuracy on validation set

score,acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))