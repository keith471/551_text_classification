'''Text classification using Naive Bayes'''
from __future__ import with_statement
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys, os, codecs
from time import time

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from classifiers import NaiveBayes
from classifiers import GaussianNaiveBayes
from preprocess import readData
from postprocess import writeResults
from cross_validation import CrossValidate

################################################################################
# logging and options
################################################################################

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--max_n_gram",
              action="store", type="int", dest="max_n_gram_length",
              help="The maximum n-gram size to be used.")
op.add_option("--use_tf_idf",
              action="store_true",
              help="If set, tf-idf term weighting will be used.")
op.add_option("--lowercase",
              action="store_true",
              help="If set, the documents will be converted to lowercase.")
op.add_option("--lemmatize",
              action="store_true",
              help="If set, all words will be lemmatized.")
op.add_option("--remove_stop_words",
              action="store_true",
              help="If set, sklearn's list of English stop words will be removed.")
op.add_option("--test",
              action="store", type="float", dest="test_fraction",
              help="Run on a fraction of the entire training corpus")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--gaussian",
              action="store_true",
              help="If set, features will be treated as continuous random variables with Gaussian distributions")
op.add_option("--use_scikit",
              action="store_true",
              help="If set, use scikit's Gaussin naive bayes implementation")
op.add_option("--predict",
              action="store_true",
              help="If set, predictions will be made for the unknown test data")
op.add_option("--cv_range",
              action="store", type=int, nargs=3, dest="cv_range",
              help="Three positive integers separated by spaces where the first and second are equal to the start and end of the range, inclusive, and the middle is equal to the step size")
op.add_option("--devset",
              action="store_true",
              help="If set, accuracy will be measured against a 30 percent dev set. Cannot be used in tandem with --cv_range.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

if opts.max_n_gram_length:
    if opts.max_n_gram_length < 1:
        op.error("Max n-gram length must be positive")
        sys.exit()

if opts.test_fraction:
    if opts.test_fraction > 1.0 or opts.test_fraction < 0.0:
        op.error("The test fraction must be between 0.0 and 1.0")
        sys.exit(1)

if opts.cv_range:
    start, end, step = opts.cv_range
    if start < 0 or start > end or step < 1:
        op.error("Invalid range")
        sys.exit(1)

if opts.cv_range and opts.devset:
    op.error("Can only use one of cross validation or a develpoment set")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(word) for word in word_tokenize(doc)]

################################################################################
# Helpers
################################################################################

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

def selectChi2(X_train, y_train, X_test, k, feature_names=None):
    print("Extracting %d best features by a chi-squared test" % k)
    t0 = time()
    # the SelectKBest object is essentially a vectorizer that will select only the most influential k features of your input vectors
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test) # revectorize X_test
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    return X_train, X_test, feature_names

def selectChi2Cv(X_train, y_train, k):
    print("Extracting %d best features by a chi-squared test" % k)
    t0 = time()
    # the SelectKBest object is essentially a vectorizer that will select only the most influential k features of your input vectors
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    return X_train

def crossValidate(X_train, y_train, clf, rng):
    '''Return a an array of tuples: (# features used, avg prediction accuracy)'''
    arr = []
    if type(clf) == "GuassianNB":
        accuracyFunc = metrics.accuracy_score
    else:
        accuracyFunc = clf.getAccuracy
    for numFeats in rng:
        X_t = selectChi2Cv(X_train, y_train, numFeats)
        crossValidator = CrossValidate(X_t, y_train, clf, accuracyFunc)
        acc = crossValidator.crossValidate()
        arr.append((numFeats, acc))
    return arr

def makePredictions(X_train, y_train, X_test, clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    print()
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    print()
    return pred

def getXTrain(docs_train):
    maxNGramLength = 1
    if opts.max_n_gram_length:
        maxNGramLength = opts.max_n_gram_length
    print("Using n-grams of up to %d words in length" % maxNGramLength)

    if opts.lowercase:
        lowercase = True
        print("Converting all text to lowercase")
    else:
        lowercase = False

    if opts.lemmatize:
        tokenizer = LemmaTokenizer()
        print("Lemmatizing all words")
    else:
        tokenizer = None

    if opts.remove_stop_words:
        stop_words = 'english'
        print("Using stop words")
    else:
        stop_words = None

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        vectorizer = HashingVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words,
                                        ngram_range=(1,maxNGramLength), non_negative=True, n_features=opts.n_features)
        X_train = vectorizer.transform(docs_train)
    elif opts.use_tf_idf:
        # a way to re-weight the count features such that extremely common words such
        # as "the" and "a" are less important than the less common ones
        print("Extracting features from the test data using a tfidf vectorizer")
        vectorizer = TfidfVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words,
                                        ngram_range=(1,maxNGramLength))
        X_train = vectorizer.fit_transform(docs_train)
    else:
        print("Extracting features from the test data using a count vectorizer")
        vectorizer = CountVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words,
                                        ngram_range=(1,maxNGramLength))
        X_train = vectorizer.fit_transform(docs_train)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    if opts.use_scikit:
        X_train = X_train.toarray()
    return X_train, vectorizer

def getXTest(docs_test, vectorizer):
        print("Extracting features from the development data using the same vectorizer")
        t0 = time()
        X_test = vectorizer.transform(docs_test)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
        print("n_samples: %d, n_features: %d" % X_test.shape)
        print()
        if opts.use_scikit:
            X_test = X_test.toarray() # required for GuassianNB implementation
        return X_test

def benchmark(X_train, y_train, X_dev, y_dev):
    if opts.use_scikit:
        X_train = X_train.toarray()
        X_dev = X_dev.toarray()
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        print()

        t0 = time()
        pred = clf.predict(X_dev)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
        print()

        # get the accuracy of the predictions against the train data
        score = metrics.accuracy_score(y_dev, pred)
        print("accuracy:   %0.3f" % score)
        print()
    else:
        print("Training on training set")
        t0 = time()
        clf.fit(X_train, y_train)
        dur = time() - t0
        print("completed training in %fs" % dur)
        print()
        print("Predicting on development set")
        t0 = time()
        pred = clf.predict(X_dev)
        dur = time() - t0
        print("completed predictions in %fs" % dur)
        print()
        accuracy = clf.getAccuracy(y_dev, pred)
        print("Accuracy:")
        print(accuracy)
        print()

def printAccuracies(accs):
    print("# features\taccuracy")
    for (numFeats, acc) in accs:
        print("%d\t%f" % (numFeats, acc))

if __name__ == "__main__":

    ################################################################################
    # data loading
    ################################################################################

    all_docs_train, all_y_train, docs_test = readData()
    print("data loaded")

    # If we train on only a fraction of the data, we will need to know this
    training_on_data_fraction = False

    ################################################################################
    # optionally extract only a portion of data for training
    ################################################################################

    if opts.test_fraction:
        training_on_data_fraction = True
        percent = (opts.test_fraction * 100.0)
        print("Using only %.f percent of the training data" % percent)
        threshold = int(opts.test_fraction * len(all_docs_train))
        if threshold == 0:
            print("Fraction too small, please choose a larger fraction")
            print()
            sys.exit(1)
        docs_train = all_docs_train[:threshold]
        y_train = all_y_train[:threshold]
    else:
        docs_train = all_docs_train
        y_train = all_y_train
    print("Train set size: %d documents" % len(docs_train))
    print("Test set size: %d documents" % len(docs_test))
    print("done")
    print()

    data_train_size_mb = size_mb(docs_train)
    data_test_size_mb = size_mb(docs_test)
    print("%d abstracts - %0.3fMB (training set)" % (
        len(docs_train), data_train_size_mb))
    print("%d abtracts - %0.3fMB (test set)" % (
        len(docs_test), data_test_size_mb))
    print()

    # define the categories
    categories = [
        'stats',
        'math',
        'physics',
        'cs'
    ]

    ################################################################################
    # extract a development set
    ################################################################################
    if opts.devset:
        #  if we are not cross validating, we may still select a development set to evaluate performance (albeit more crudely)
        training_on_data_fraction = True
        print("Extracting development set from training set")
        docs_train, docs_dev, y_train, y_dev = train_test_split(docs_train, y_train, test_size=0.3, random_state=0)
        print("Using %d training examples and %d testing examples" % (len(docs_train), len(docs_dev)))
        print("done")
        print()

        data_dev_size_mb = size_mb(docs_dev)
        print("%d abstracts - %0.3fMB (development set)" % (
            len(docs_dev), data_dev_size_mb))
        print()

    ################################################################################
    # vectorize the training data
    ################################################################################

    X_train, vectorizer = getXTrain(docs_train)

    feature_names = vectorizer.get_feature_names()

    if opts.devset:
        X_dev = getXTest(docs_dev, vectorizer)

    ################################################################################
    # classification and prediction
    ################################################################################

    if opts.gaussian:
        clf = GaussianNaiveBayes()
    elif opts.use_scikit:
        clf = GaussianNB()
    else:
        clf = NaiveBayes()

    # either cross validate over a range of numbers of features, or determine the
    # performance on the development set for all or just some features (no cross validation)
    if opts.cv_range:
        print("Cross validating to find the best number of features in the provided range")
        # Cross validate over the range of numbers of features
        start, end, step = opts.cv_range
        rng = range(start, end+1, step)
        accuracies = crossValidate(X_train, y_train, clf, rng)

        # print out the accuracies
        print("Summary of accuracies:")
        printAccuracies(accuracies)
        print()

        bestAcc = 0
        bestNumFeats = 0
        for (numFeats, acc) in accuracies:
            if acc > bestAcc:
                bestAcc = acc
                bestNumFeats = numFeats
        print("Best number of features: %d" % bestNumFeats)
        print()
        numFeatsToPredictOn = bestNumFeats
    elif opts.devset:
        print("Gauging model performance against development set")
        if opts.select_chi2:
            # we didn't specify a range. But we still might want to select only
            # the top k features to make predictions against the development set
            X_train, X_dev, feature_names = selectChi2(X_train, y_train, X_dev, opts.select_chi2, feature_names)
            # covert feature_names to an ndarray
            feature_names = np.asarray(feature_names)
            benchmark(X_train, y_train, X_dev, y_dev)
            numFeatsToPredictOn = opts.select_chi2
        else:
            print("Using all features")
            # use all the features to make predictions against the development set
            benchmark(X_train, y_train, X_dev, y_dev)
            numFeatsToPredictOn = -1
    else:
        # just in case we don't train or validate but still want to predict
        if opts.select_chi2:
            numFeatsToPredictOn = opts.select_chi2
        else:
            numFeatsToPredictOn = -1

    if opts.predict:
        print("Making predictions!!!")
        # make predictions for docs_test
        if training_on_data_fraction:
            # revectorize on ALL the data
            X_train, vectorizer = getXTrain(all_docs_train)

        X_test = getXTest(docs_test, vectorizer)

        if numFeatsToPredictOn != -1:
            print("Selecting the best %d features" % numFeatsToPredictOn)
            X_train, X_test, feature_names = selectChi2(X_train, y_train, X_test, numFeatsToPredictOn, feature_names)

        print("Making predictions")
        pred = makePredictions(X_train, y_train, X_test, clf)
        writeResults(pred, "naive_bayes")

'''
    if opts.use_scikit:
        X_train = X_train.toarray()
        X_dev = X_dev.toarray()
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        print()

        t0 = time()
        pred = clf.predict(X_dev)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
        print()

        # get the accuracy of the predictions against the train data
        score = metrics.accuracy_score(y_dev, pred)
        print("accuracy:   %0.3f" % score)
        print()
    else:
        print("Training on training set")
        t0 = time()
        clf.train(X_train, y_train)
        dur = time() - t0
        print("completed training in %fs" % dur)
        print()
        print("Predicting on development set")
        t0 = time()
        pred = clf.predict(X_dev)
        dur = time() - t0
        print("completed predictions in %fs" % dur)
        print()
        accuracy = clf.getAccuracy(y_dev, pred)
        print("Accuracy:")
        print(accuracy)
        print()
'''
