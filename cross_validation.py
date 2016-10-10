'''Cross validation'''

from data_partitioner import DataPartitioner

class CrossValidate:

    def __init__(self, X, y, clf, accuracyFunc, cv=5):
        self.X = X
        self.y = y
        self.cv = cv
        self.clf = clf
        self.accuracyFunc = accuracyFunc
        self.partitioner = DataParitioner(cv, X, y)

    def crossValidate():
        '''Trains and tests the given classifier on cv folds, and returns the average accuracy'''
        sumAccuracy = 0.0
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.partitioner.getPartitions()):
            print("Training on training set %d" % i)
            t0 = time()
            clf.train(X_train, y_train)
            dur = time() - t0
            print("completed training in %fs" % dur)
            print()
            print("Predicting on test set %d" % i)
            t0 = time()
            pred = clf.predict(X_test)
            dur = time() - t0
            print("completed predictions in %fs" % dur)
            print()
            accuracy = self.accuracyFunc(y_dev, pred)
            sumAccuracy += accuracy
            print("Accuracy of %dth partition:" % i)
            print(accuracy)
            print()
        print("Average accuracy:")
        avgAcc = sumAccuracy / self.cv
        print(avgAcc)
        return avgAcc
