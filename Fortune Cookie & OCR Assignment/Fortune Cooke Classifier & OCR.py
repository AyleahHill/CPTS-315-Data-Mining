# Ayleah K. Hill
# CPTS 315
# Programming Assignment 3

# This code is based off the code provided in class presentations
# I am having troubles with printing stuff to the file and I do not want to turn in the project later than now
# I understand the concept the project is trying to convey.
# Turned in late because I have been sick and feeling under the weather.

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')

##################################################################
# PART 1: FORTUNE COOKIES
# stop words
stop = stopwords.words('english')

# read in the data from the training, testing and stoplist files
trainData = pd.read_csv("./traindata.txt", header=None)
trainLabels = pd.read_csv("./trainlabels.txt", header=None)
stoplist = pd.read_csv("./stoplist.txt", header=None)
preddata = pd.read_csv("./testdata.txt", header=None)
predlabels = pd.read_csv("./testlabels.txt", header=None)

# set the size of the training data
trainDataRange = trainData.size
# set the size of the testing data
preddataRange = preddata.size

# rename columns
trainData.columns = ['Data']
trainLabels.columns = ['Label']
stoplist.columns = ['Stop']
preddata.columns = ['Data']
predlabels.columns = ['Label']

# combining trainint and testing data
comboData = pd.concat([trainData, preddata])

# creating a list of stop words
stopwords = stoplist['Stop'].tolist()

# creating tokenized data
comboData['Filt_Data'] = comboData['Data'].apply(lambda w: ' '.join(
    [word for word in w.split() if word not in (stopwords)]))
del comboData['Data']
comboData['Tokenized_Data'] = comboData.apply(
    lambda row: nltk.word_tokenize(row['Filt_Data']), axis=1)

# turning training labels into a list of training labels
trainLabels = trainLabels['Label'].tolist()
# turning testing labels into a list of testing labels
predlabels = predlabels['Label'].tolist()

with open("output.txt", 'w') as dataout:
    # run training data through TFIDF
    v = TfidfVectorizer()
    Tfidf = v.fit_transform(comboData['Filt_Data'])
    df1 = pd.DataFrame(Tfidf.toarray(), columns=v.get_feature_names())
    dataout.write(df1.to_string() + '\n')

    # separate the training data and the testing data again
    trainData = df1[0:trainDataRange]
    preddata = df1[trainDataRange: trainDataRange + preddataRange]

    # run an implementation of the perceptron algorithm
    ppn = Perceptron(max_iter=20, eta0=1, random_state=0, verbose=1)
    # fit the training data and testing data
    ppn.fit(trainData, trainLabels)
    traindatapredict = ppn.predict(preddata)

    # print the accuracy of the predictions
    dataout.write('Accuracy: %.2f' %
                  accuracy_score(predlabels, traindatapredict))
    dataout.write('\n')

##################################################################
# PART 2: OCR
# Creating a list of words from a string of words


def split(word):
    return list(word)
# Cleaning up data files (pre-processing step)


def pre_process(file):
    # with the file open, read all the lines and process information
    with open(file, 'r') as f:
        lines = f.readlines()
    i = 0
    k = []
    label = []
    # For every line in the file
    for line in lines:
        l = re.split(r'\t+', line)
        # filter out empty lines
        # separate index, data, and label
        # process the data
        # create labels for words
        if len(l) > 2:
            print(l)
            print(l[0])
            l1 = split(l[1][2: len(l[1])])
            dsl = len(l1)
            print('data string length: ', len(l1))
            if i == 0:
                k1 = np.array(l1)
                k = k1
                label1 = np.array(l[2])
                label = np.append(label, label1)
            else:
                k1 = np.array(l1)
                k = np.append(k, k1)
                label1 = np.array(l[2])
                label = np.append(label, label1)
            # TODO: FIX THIS!! WHERE DOES Y COME FROM?
            dataout.write(k + '\n')
            i = i + 1
        dataout.write(i.to_string() + '\n')

    # reshape
    k2 = k.reshape(i, dsl)

    # create a dataframe from the array
    dataframe = pd.DataFrame.from_records(k2)
    return(dataframe, label)

# perceptron implenentation


def implement_perceptron(train_data, train_labels, test_data, test_labels):
    with open("output.txt", 'w') as dataout:
        # run perceptron function
        ppn = Perceptron(max_iter=50, eta=0.1, random_state=0, verbose=1)
        # fir dataframe label
        ppn.fit(train_data, train_labels)
        # calculate the results for the test data
        testpredict = ppn.predict(test_data)
        dataout.write(testpredict.to_string() + '\n')


# read in the data for the training and testing
train = r"./ocr_train.txt"
test = r"./ocr_test.txt"
# pre-process the data for both the training and testing data
traindata2, trainlabels2 = pre_process(train)
testdata2, testlabels2 = pre_process(test)
# run the implementation of the perceptron function
implement_perceptron(traindata2, trainlabels2, testdata2, testlabels2)
