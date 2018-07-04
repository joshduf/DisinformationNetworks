# These functions take tweets in a csv format and organize them
#   for processing by a deep learning framework

import csv
import numpy as np
import random
import re

# Non-ASCII values are ignored to ensure the model doesn't
#   use artifacts from the encoding process to classify tweets
#   as I have no control over the methods NBCNews used to create
#   it's csv of positive examples.
# Only time and tweet text are used and are tagged
def getFile(file, MAX):
    STWEET = "<SW>"
    ETWEET = "<EW>"
    SHOUR = "<SH>"
    EHOUR = "<EH>"
    STEXT = "<ST>"
    ETEXT = "<ET>"

    lines = []
    with open(file, "r", encoding="utf-8") as f:
        data = csv.reader(f)

        for _, row in zip(range(MAX), data):
            time = row[1].encode('ascii',errors='ignore').decode()
            text = row[2].encode('ascii',errors='ignore').decode()
            lines.append(STWEET + SHOUR + time + EHOUR + STEXT + text + ETEXT + ETWEET)
        
        f.close()
        
    return lines

# To be fed into a CNN, all examples are padded to the length of the
#   longest example
def pad(data):
    longest = len(max(data, key=len))
    padded = [row.ljust(longest, ">") for row in data]
    
    return padded

# Tweets are grouped together. This assumes they were organized by
#   user first, then time. The dataset used had on average hundreds
#   of tweets per user and the group size was <= 20 so the possibility
#   of a group consisting of tweets from 2 different users don't
#   significantly effect the outcome of classification and are ignored.
def combine(data, GROUPSIZE):
    START = "<S>"
    END = "<E>"
    newData = []
    
    for newRow in range(len(data)//GROUPSIZE):
        combined = ""
        for oldRow in range(newRow*GROUPSIZE, (newRow + 1)*GROUPSIZE):
            combined += data[oldRow] + " "
        combined = START + " " + combined + END
        newData.append(combined)
    
    return newData

# Takes positive and negative examples and creates input and output vectors
def readData(POSFILE, NEGFILE, NEGFILE2, MAXPOS, MAXNEG, MAXNEG2, GROUP):    
    pos = getFile(file=POSFILE, MAX=MAXPOS)
    neg = getFile(file=NEGFILE, MAX=MAXNEG) + getFile(file=NEGFILE2, MAX=MAXNEG2)

    pos = combine(pos, GROUP)
    neg = combine(neg, GROUP)
    
    y = [1 for line in pos] + [0 for line in neg]
    x = pad(pos + neg)

    return (x, y)

# Creates a dictionary mapping characters to an associated index
def getIndexes(x):
    letters = set()

    for line in x:
        letters.update(line)

    indexes = dict((letter, index) for index, letter in enumerate(letters))
    
    return indexes

# Turns character vector into vector of character indexes
def vectorize(x, y, indexes):
    Y = np.array(y)
    X = [[indexes.get(letter, -1) for letter in line] for line in x]
    
    return (X, Y)

# Splits data into test and train sections
def splitData(X, Y, SPLITSIZE):
    x_train = np.array(X[:SPLITSIZE])
    x_test = np.array(X[SPLITSIZE:])

    y_train = np.array(Y[:SPLITSIZE])
    y_test = np.array(Y[SPLITSIZE:])
    
    return (x_train, y_train), (x_test, y_test)
