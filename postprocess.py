# postprocessing

import csv
import sys

def writeToCsv(filename, data, fieldnames):
    ''' writes and array of data to csv with field names fieldNames'''
    with open(filename + '.csv', 'w') as mycsvfile:
        writer = csv.writer(mycsvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

def writeClassifier(filename, pred):
    ''' takes an array of predictions and converts it into an array of id, prediction entries '''
    data = [[i,v] for i,v in pred]
    fieldnames = ['id', 'category']
    writeToCsv(filename, data, fieldnames)

def postProcess(results):
    for i, v in results:
        filename = "%s_%d" % (v[0], i)
        writeClassifier(filename, v[1])
