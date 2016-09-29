# This file processes the data into a format amenable for classification

import csv
import sys
import numpy as np

def readFile(fname):
    ''' returns an array containing the data from the file '''
    data = []
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # skip the header
        reader.next()
        for row in reader:
            # row[0] contains the id, row [1] contains the data
            data.append(row[1])
    return data

def readData():
    ''' reads all train and test data and returns as three arrays '''
    abstractsTrain = readFile('../datasets/train_in.csv')
    y = readFile('../datasets/train_out.csv')
    abstractsTest = readFile('../datasets/test_in.csv')
    return abstractsTrain, y, abstractsTest

def processData():
    # preprocessing to go here
    return readData()
