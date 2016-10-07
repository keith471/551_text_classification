'''Custom Naive Bayes estimator (NOT compatible with sklearn)'''

class NaiveBayes:
    ''' Usage:
    call train() to save the probabilities on the class
    call predict(dataSet) to get the predictions for the given vectors
    call getAccuracy(dataSet, predictions) to get the percentage accuracy of the predictions
      dataSet should be the same set passed to predict
      predictions should be the return from predict
    '''

    def compose(self, X, y):
        composed = []
        for i, v in enumerate(X):
            entry = np.append(v, y[i])
            composed.append(entry)
        return composed

    def separateByClass(self, dataset):
        '''
        returns a map of class to all points in dataset that fall into that class
        e.g. {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
        '''
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    def getClassProbabilities(self, dataset):
        '''
        returns a map of class to its probability, as computed from dataset
        '''
        probabilities = {}
        dataSetSize = len(dataset)
        separated = self.separateByClass(dataset)
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
        del clusters[-1] # delete the cluster of y values as we need not compute anything for these
        for index, cluster in enumerate(clusters):
            # cluster is an array of all the values of a given feature in instances
            summaries.append(self.getDiscreteProbabilities(cluster))
        return summaries

    def summarizeByClass(self, dataset):
        '''
        returns a map of class value to:
        {
            priorProbs: [dict of prior probs for each feature for the class],
            classProb: class probability
        }
        '''
        classProbabilities = self.getClassProbabilities(dataset)
        # get datapoints clustered by class
        separated = self.separateByClass(dataset)
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
                else:
                    print("NOTE: Bayesian probability that input vector {0} has class " +
                        "{1} is zero as the training data contained no instances with " +
                        "feature {2} given class {1}").format(inputVector, classValue, i)
                    print()
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

    def getPredictions(self, summaries, testSet):
        '''
        returns an array of classification predictions for the instances in a test set
        '''
        predictions = []
        for i in range(len(testSet)):
            result = self.predictSingleInstance(summaries, testSet[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, y, pred):
        '''
        returns the fraction of the test set that were classified incorrectly
        '''
        total_incorrect = 0
        for i in range(len(y)):
            if y[i] != predictions[i]:
                total_incorrect += 1
        return (total_incorrect/float(len(y)))

    # API -- this is all you need to use
    def train(self, X, y):
        self.trainingSet = self.compose(X, y)
        self.summaries = self.summarizeByClass(self.featureTypes, self.trainingSet)
        return self.summaries

    def predict(self, X, y=None):
        if y == None:
            y = ['?' for i in range(len(X))]
        dataSet = self.compose(X, y)
        self.pred = self.getPredictions(self.summaries, dataSet)
        return self.pred

    def cost(self, y, predictions):
        return self.getAccuracy(testSet, predictions)


if __name__ == "__main__":
