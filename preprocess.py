# This file processes the data into a format amenable for classification
from __future__ import print_function

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

def getClassCounts(y):
    numMath = 0
    numStats = 0
    numCs = 0
    numPhysics = 0
    for v in y:
        if v == "math":
            numMath += 1
        elif v == "stat":
            numStats += 1
        elif v == "cs":
            numCs += 1
        elif v == "physics":
            numPhysics += 1
    print("Math: %d abstracts, %f percent" % (numMath, (float(numMath) / len(y)) * 100.0))
    print("Stats: %d abstracts, %f percent" % (numStats, (float(numStats) / len(y)) * 100.0))
    print("CS: %d abstracts, %f percent" % (numCs, (float(numCs) / len(y)) * 100.0))
    print("Physics: %d abstracts, %f percent" % (numPhysics, (float(numPhysics) / len(y)) * 100.0))
    return numMath, numStats, numCs, numPhysics
