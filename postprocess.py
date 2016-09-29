# postprocessing

import csv
import sys

def writeToCsv(filename, data, fieldnames):
    ''' writes and array of data to csv with field names fieldNames'''
    print "Writing predictions for %s classifier to %s.csv" % (filename, filename)
    with open(filename + '.csv', 'w') as mycsvfile:
        writer = csv.writer(mycsvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

def writeResults(filename, pred):
    ''' takes an array of predictions and converts it into an array of id, prediction entries '''
    data = [[i,v] for i,v in enumerate(pred)]
    fieldnames = ['id', 'category']
    writeToCsv(filename, data, fieldnames)

def postProcess(results):
    for i, v in results:
        filename = "%s_%d" % (v[0], i)
        writeResults(filename, v[1])
