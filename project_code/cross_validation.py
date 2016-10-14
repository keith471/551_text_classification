'''Cross validation'''
from __future__ import print_function

from time import time

from data_partitioner import DataPartitioner

class CrossValidate:

    def __init__(self, X, y, clf, accuracyFunc, cv=3):
        self.X = X
        self.y = y
        self.cv = cv
        self.clf = clf
        self.accuracyFunc = accuracyFunc
        self.partitioner = DataPartitioner(cv, X, y)

    def crossValidate(self):
        '''Trains and tests the given classifier on cv folds, and returns the average accuracy'''
        sumAccuracy = 0.0
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.partitioner.getPartitions()):
            print("Training on training set %d" % i)
            t0 = time()
            self.clf.fit(X_train, y_train)
            dur = time() - t0
            print("completed training in %fs" % dur)
            print("Predicting on test set %d" % i)
            t0 = time()
            pred = self.clf.predict(X_test)
            dur = time() - t0
            print("completed predictions in %fs" % dur)
            accuracy = self.accuracyFunc(y_test, pred)
            sumAccuracy += accuracy
            print("Accuracy of %dth partition:" % i)
            print(accuracy)
            print()
        print("Average accuracy:")
        avgAcc = sumAccuracy / self.cv
        print(avgAcc)
        print()
        return avgAcc
