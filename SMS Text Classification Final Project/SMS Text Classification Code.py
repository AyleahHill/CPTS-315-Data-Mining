# Author: Ayleah Hill
# Professor Ananth Jillepalli
# CPTS 315 Data Mining
# December 11, 2020

# CPTS 315 Final Project: Analysis of SMS Text Messages for Text Classification
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.layers import LSTM, Dense, Embedding, Input, Activation, Dropout
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop, Adam
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import numpy
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Sources used to assist with coding of the project:
# For coding the LSTM:
# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
# https://www.kaggle.com/kredy10/simple-lstm-for-text-classification
# For coding the word cloud:
# https://www.datacamp.com/community/tutorials/wordcloud-python

# TO RUN THE CODE:
# Run the program and it will run through the LSTM
# At the end of the LSTM it will make both Word Clouds


# pre process the data to clean it up and make it consistent
def preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.split(' ')
    text = [w for w in text if not w in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# LSTM setup of model to use for data analysis


def LSTMModel():
    vals = Input(shape=[max_length])
    embedding_layer = Embedding(max_words, 50, input_length=max_length)(vals)
    embedding_layer = LSTM(64)(embedding_layer)
    embedding_layer = Dense(256, activation='relu')(embedding_layer)
    embedding_layer = Dropout(0.5)(embedding_layer)
    embedding_layer = Dense(1, activation='sigmoid')(embedding_layer)
    model = Model(inputs=vals, outputs=embedding_layer)
    return model


max_words = 1000
max_length = 150

# read in the file
df = pd.read_csv('spam.csv', encoding='latin')

# drop the weird columns and rename
df = df.drop(
    labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
# df.columns = ['label', 'text']

# text preprocessing
df.v2 = df.v2.apply(lambda x: preprocessing(x))
label = df.v1
label = LabelEncoder().fit_transform(label)
label = label.reshape(-1, 1)
text = df.v2

# separate training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    text, label, test_size=0.25)

# tokenize the training data
token = Tokenizer(num_words=max_words)
token.fit_on_texts(x_train)

# add padding to the training data
seq = token.texts_to_sequences(x_train)
seq_padded = pad_sequences(seq, maxlen=max_length)

# create an LSTM model
model = LSTMModel()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
model.summary()

# run the training data
model.fit(seq_padded, y_train, epochs=10,
          validation_split=0.25, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

# run the testing data
testing = token.texts_to_sequences(x_test)
test_seq_padded = sequence.pad_sequences(testing, maxlen=max_length)

# show the final results on testing data
final = model.evaluate(test_seq_padded, y_test)
print('Results for the test data:\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(
    final[0], final[1]))


# creating the spam word cloud
spam = df[df['v1'] == 'spam']
spamwords = " ".join(msg for msg in spam.v2)
cloudsw = set(STOPWORDS)
spamcloud = WordCloud(
    stopwords=cloudsw, background_color="white").generate(spamwords)
plt.imshow(spamcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# creating the ham word cloud
ham = df[df['v1'] == 'ham']
hamwords = " ".join(msg for msg in ham.v2)
hamcloud = WordCloud(
    stopwords=cloudsw, background_color="white").generate(hamwords)
plt.imshow(hamcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
