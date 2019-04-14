#Importing libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from typing import Iterator, Tuple, Text, Sequence

#Since fit_to_texts only able to receive list of texts.
#have to create new read_tweet function
def read_tweet(tweet_path: str, emoji_path: str) -> Tuple[Sequence[Text], Sequence[Text]]:
    tweets = open(tweet_path, encoding='utf8').readlines()
    emojis = open(emoji_path, encoding='utf8').readlines()
    tweetList = []
    emojiList = []
    for i in range(len(tweets)):
        tweetList.append(tweets[i].replace("\n", ""))
        emojiList.append(emojis[i].replace("\n", ""))
    return (emojiList, tweetList)

class RNN:
    #Converting into pandas dataframe and filtering only text and ratings given by the users
    #Will need to handle reading data here somehow
    def __init__(self):
        self.embed_dim = 128
        self.lstm_out = 300
        self.batch_size= 64
        #tokenizer to maximum word is 2500, cannot have more than this
        self.tokenizer = Tokenizer(nb_words = 3500, split=' ')
        #initial the model with Sequenctial class from Keras
        self.model = Sequential()

    def train(self, train_texts: Sequence[Text], train_labels: Sequence[Text]):
        #this will help us keep track of the words that is frequent
        self.tokenizer.fit_on_texts(train_texts)

        #print(tokenizer.word_index)  # To see the dicstionary
        #this will give us the sequence of interger represent for those index create
        #with the fit_on_texts
        doc_feat_matrix = self.tokenizer.texts_to_sequences(train_texts)

        #pad_sentence will simply make sure that all the representation has the same length
        #of the longest sentence because not all the sentence have the same length
        #without this this can mess up our embedding
        doc_feat_matrix = pad_sequences(doc_feat_matrix)

        ##Buidling the LSTM network
        self.model.add(Embedding(2500, self.embed_dim,input_length = doc_feat_matrix.shape[1], dropout=0.1))
        self.model.add(LSTM(self.lstm_out, dropout_U=0.1, dropout_W=0.1))
        self.model.add(Dense(20,activation='softmax'))
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        self.model.fit(np.array(doc_feat_matrix), np.array(train_labels), batch_size = self.batch_size, nb_epoch = 10,  verbose = 2)

    def predict(self, test_texts: Sequence[Text]):
        test_feat_matrix = pad_sequences(self.tokenizer.texts_to_sequences(test_texts))
        return self.model.predict(np.array(test_feat_matrix), batch_size=64, verbose=1)