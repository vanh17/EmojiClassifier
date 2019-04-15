#Importing libraries
import numpy as np
import h5py
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils import to_categorical
from typing import Iterator, Tuple, Text, Sequence
from sklearn import preprocessing
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.layers import SpatialDropout1D

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
        self.tokenizer = Tokenizer(nb_words=2500, split=' ')
        #initial the model with Sequenctial class from Keras
        self.model = Sequential()
        #initialize label encoder
        self.lbEncoder = preprocessing.LabelEncoder()

    def to_labels(self, labels: Sequence[Text]):
        return self.lbEncoder.transform(labels)

    def to_text(self, i):
        return self.feature_names[i] 

    def train(self, train_texts: Sequence[Text], train_labels: Sequence[Text]):
        #this will help us keep track of the words that is frequent
        self.tokenizer.fit_on_texts(train_texts)
        # assing the lbEncnder classes to the feature name
        # so that we can access it in the index
        self.lbEncoder.fit(train_labels)
        self.feature_names = list(self.lbEncoder.classes_)

        #print(tokenizer.word_index)  # To see the dicstionary
        #this will give us the sequence of interger represent for those index create
        #with the fit_on_texts
        doc_feat_matrix = self.tokenizer.texts_to_sequences(train_texts)

        #pad_sentence will simply make sure that all the representation has the same length
        #of the longest sentence because not all the sentence have the same length
        #without this this can mess up our embedding
        doc_feat_matrix = pad_sequences(doc_feat_matrix)
        self.maxlen = doc_feat_matrix.shape[1]

        ##Buidling the LSTM network
        # Keras 2.0 does not support dropout anymore
        # Add spatial dropout instead
        self.model.add((Embedding(2500, self.embed_dim,input_length = doc_feat_matrix.shape[1], dropout=0.1))
        self.model.add(LSTM(self.lstm_out, dropout_U=0.1, dropout_W=0.1))
        self.model.add(Dense(20,activation='softmax'))
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

        # do early stopping
        es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.5)

        #save the best model
        filepath="models/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint, es]

        #start the training here
        self.model.fit(doc_feat_matrix, to_categorical(self.lbEncoder.transform(train_labels)), batch_size = self.batch_size, epochs = 10,  callbacks = callbacks_list, verbose = 0)

    def save_model(self, path_to_folder: Text, fname: Text):
        model_json = self.model.to_json()
        with open(path_to_folder + "/" + fname + ".json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model from disk")

    def load_model(self, path_to_folder: Text, fname: Text):
        json_file = open(path_to_folder + "/" + fname + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(path_to_folder + "/" + "weights.best.hdf5.h5")
        print("Loaded model from disk")

    def predict(self, test_texts: Sequence[Text]):
        test_feat_matrix = pad_sequences(self.tokenizer.texts_to_sequences(test_texts), maxlen=self.maxlen)
        return np.argmax(self.model.predict(test_feat_matrix, batch_size=64, verbose=1), axis=1)