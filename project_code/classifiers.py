'''Custom Naive Bayes estimator (NOT compatible with sklearn)'''
from __future__ import print_function
import numpy as np
import math

class NaiveBayes:
    ''' Usage:
    call train() to save the probabilities on the class
    call predict(dataSet) to get the predictions for the given vectors
    call getAccuracy(dataSet, predictions) to get the percentage accuracy of the predictions
      dataSet should be the same set passed to predict
      predictions should be the return from predict
    '''

    def separateByClass(self, X, y):
        '''
        returns a map of class to all points in dataset that fall into that class
        e.g. {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
        '''
        separated = {}
        for i, v in enumerate(X):
            if (y[i] not in separated):
                separated[y[i]] = []
            separated[y[i]].append(v)
        return separated

    def getClassProbabilities(self, X, y):
        '''
        returns a map of class to its probability, as computed from dataset
        '''
        probabilities = {}
        dataSetSize = len(X)
        separated = self.separateByClass(X, y)
        for classValue, instances in separated.iteritems():
            probabilities[classValue] = float(len(instances))/dataSetSize
        return probabilities

    def getDiscreteProbabilities(self, numbers):
        '''
        here, numbers is a list of all the values a given feature of a given class has
        Thus len(numbers) = number of instances of the given class in the dataset
        this function returns a map of each discrete value to its probability within the list
        e.g. if numbers = [1,1,3,2] then this function returns:
        {1: 0.5, 2: 0.25, 3: 0.25}
        e.g. P(word "calculus" occurs 2 times in document | document class is "math") = 0.25
        '''
        setSize = len(numbers)
        counts = {}
        for value in numbers:
            if value not in counts:
                counts[value] = 0
            counts[value] += 1
        probabilities = {}
        for value, count in counts.iteritems():
            probabilities[value] = float(count) / setSize
        return probabilities

    def summarize(self, instances):
        '''
        Calculates and returns the prior probabilities of each feature for a class
        instances is an array of feature vectors of a single class
        returns an array of:
          - {featureValue: probability, anotherFeatureValue: prob} for each feature in the instances of a given class
        e.g. if instances = [[1,10, 20], [3, 10, 21]] then this function returns
        [{1: 0.5, 3: 0.5}, {10: 1.0}, {20: 0.5, 21: 0.5}]
        '''
        summaries = []
        clusters = zip(*instances)
        for index, cluster in enumerate(clusters):
            # cluster is an array of all the values of a given feature in instances
            summaries.append(self.getDiscreteProbabilities(cluster))
        return summaries

    def summarizeByClass(self, X, y):
        '''
        returns a map of class value to:
        {
            priorProbs: [dict of prior probs for each feature for the class],
            classProb: class probability
        }
        '''
        classProbabilities = self.getClassProbabilities(X, y)
        # get datapoints clustered by class
        separated = self.separateByClass(X, y)
        summaries = {}
        for classValue, instances in separated.iteritems():
            # e.g. classValue = "math", instances = array of all feature vectors for instances of that class
            summaries[classValue] = {'priorProbs': self.summarize(instances), 'classProb': classProbabilities[classValue]}
        return summaries

    def calculateProbabilitiesByClass(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummary in summaries.iteritems():
            probabilities[classValue] = classSummary['classProb']
            for i in range(len(classSummary['priorProbs'])):
                currProb = 0.0
                currFeatureValue = inputVector[i]
                if currFeatureValue in classSummary['priorProbs'][i]:
                    currProb = classSummary['priorProbs'][i][currFeatureValue]
                probabilities[classValue] *= currProb
        return probabilities

    ##
    # Now, in order to classify an instance, we merely need to iterate through
    # the probabilities dictionary and select the class with greatest probability
    ##

    def predictSingleInstance(self, summaries, inputVector):
        '''
        given the class summaries and an input vector, predicts the class of the input
        '''
        probabilities = self.calculateProbabilitiesByClass(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if probability > bestProb:
                bestLabel = classValue
                bestProb = probability
        return bestLabel

    def getPredictions(self, summaries, X):
        '''
        returns an array of classification predictions for the instances in a test set
        '''
        predictions = []
        for i in range(len(X)):
            result = self.predictSingleInstance(summaries, X[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, y, pred):
        '''
        returns the fraction of the test set that were classified correctly
        '''
        total = 0
        for i in range(len(y)):
            if y[i] == pred[i]:
                total += 1
        return (total/float(len(y)))

    # API -- this is all you need to use
    def fit(self, X, y):
        # X could be a sparse matrix of sparse matrices
        # If it is, convert it to an ndarray
        op = getattr(X, "toarray", None)
        if op:
            X = X.toarray()
        self.X = X
        self.y = y
        self.summaries = self.summarizeByClass(X, y)
        return self.summaries

    def predict(self, X):
        op = getattr(X, "toarray", None)
        if op:
            X = X.toarray()
        self.pred = self.getPredictions(self.summaries, X)
        return self.pred


class GaussianNaiveBayes:
    ''' Usage:
    call train() to save the probabilities on the class
    call predict(dataSet) to get the predictions for the given vectors
    call getAccuracy(dataSet, predictions) to get the percentage accuracy of the predictions
      dataSet should be the same set passed to predict
      predictions should be the return from predict
    '''

    def separateByClass(self, X, y):
        '''
        returns a map of class to all points in dataset that fall into that class
        e.g. {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
        '''
        separated = {}
        for i, v in enumerate(X):
            if (y[i] not in separated):
                separated[y[i]] = []
            separated[y[i]].append(v)
        return separated

    def mean(self, numbers):
    	return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
    	avg = self.mean(numbers)
    	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    	return math.sqrt(variance)

    def getClassProbabilities(self, X, y):
        '''
        returns a map of class to its probability, as computed from dataset
        '''
        probabilities = {}
        dataSetSize = len(X)
        separated = self.separateByClass(X, y)
        for classValue, instances in separated.iteritems():
            probabilities[classValue] = float(len(instances))/dataSetSize
        return probabilities

    def getDiscreteProbabilities(self, numbers):
        '''
        here, numbers is a list of all the values a given feature of a given class has
        Thus len(numbers) = number of instances of the given class in the dataset
        this function returns a map of each discrete value to its probability within the list
        e.g. if numbers = [1,1,3,2] then this function returns:
        {1: 0.5, 2: 0.25, 3: 0.25}
        e.g. P(word "calculus" occurs 2 times in document | document class is "math") = 0.25
        '''
        setSize = len(numbers)
        counts = {}
        for value in numbers:
            if value not in counts:
                counts[value] = 0
            counts[value] += 1
        probabilities = {}
        for value, count in counts.iteritems():
            probabilities[value] = float(count) / setSize
        return probabilities

    def summarize(self, instances):
        '''
        Calculates and returns the prior probabilities of each feature for a class
        instances is an array of feature vectors of a single class
        returns an array of:
        (mean, stdev) tuples for each feature in the dataset
        '''
        summaries = []
        clusters = zip(*instances)
        for index, cluster in enumerate(clusters):
            # cluster is an array of all the values of a given feature in instances
            summaries.append((self.mean(cluster), self.stdev(cluster)))
        return summaries

    def summarizeByClass(self, X, y):
        '''
        returns a map of class value to:
        {
            featStats: [tuple of mean and stdev for corresponding to index i],
            classProb: class probability
        }
        '''
        classProbabilities = self.getClassProbabilities(X, y)
        # get datapoints clustered by class
        separated = self.separateByClass(X, y)
        summaries = {}
        for classValue, instances in separated.iteritems():
            # e.g. classValue = "math", instances = array of all feature vectors for instances of that class
            summaries[classValue] = {'featStats': self.summarize(instances), 'classProb': classProbabilities[classValue]}
        return summaries

    def calculateProbability(self, x, mean, stdev):
        if stdev == 0.0:
            if x == mean:
                return 1.0
            else:
                return 0.0
        exponent = math.exp(-(math.pow(x - mean, 2)/(2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateProbabilitiesByClass(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummary in summaries.iteritems():
            probabilities[classValue] = classSummary['classProb']
            for i in range(len(classSummary['featStats'])):
                currProb = 0.0
                currFeatureValue = inputVector[i]
                mean, stdev = classSummary['featStats'][i]
                currProb = self.calculateProbability(currFeatureValue, mean, stdev)
                probabilities[classValue] *= currProb
        return probabilities

    ##
    # Now, in order to classify an instance, we merely need to iterate through
    # the probabilities dictionary and select the class with greatest probability
    ##

    def predictSingleInstance(self, summaries, inputVector):
        '''
        given the class summaries and an input vector, predicts the class of the input
        '''
        probabilities = self.calculateProbabilitiesByClass(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if probability > bestProb:
                bestLabel = classValue
                bestProb = probability
        return bestLabel

    def getPredictions(self, summaries, X):
        '''
        returns an array of classification predictions for the instances in a test set
        '''
        predictions = []
        for i in range(X.shape[0]):
            result = self.predictSingleInstance(summaries, X[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, y, pred):
        '''
        returns the fraction of the test set that were classified correctly
        '''
        total = 0
        for i in range(len(y)):
            if y[i] == pred[i]:
                total += 1
        return (total/float(len(y)))

    # API -- this is all you need to use
    def fit(self, X, y):
        # X is a sparse matrix of sparse matrices
        # convert it to an ndarray
        X = X.toarray()
        self.X = X
        self.y = y
        self.summaries = self.summarizeByClass(X, y)
        return self.summaries

    def predict(self, X):
        X = X.toarray()
        self.pred = self.getPredictions(self.summaries, X)
        return self.pred
