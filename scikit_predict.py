from __future__ import with_statement
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys, os, codecs
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.extmath import density
from sklearn import metrics

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from preprocess import processData
from postprocess import postProcess

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
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
op.add_option("--tune",
              action="store_true",
              help="If set, models that can be tuned will be tuned using cross "
              "validation and grid search. This will slow down the program substantially.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

if opts.max_n_gram_length:
    if opts.max_n_gram_length < 1:
        op.error("Max n-gram length must be positive")
        sys.exit()

print(__doc__)
op.print_help()
print()

#################################################################################
## read data
#################################################################################

def getData():
    return processData()

################################################################################
# custom tokenizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(word) for word in word_tokenize(doc)]

################################################################################
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

################################################################################
# helper for benchmarking
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

# Benchmark classifiers
# clf is the classifier
def benchmark(clf, X_train, y_train, X_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    print()

    '''
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()
    '''

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    # need the classes of the test set for this
    '''
    # get the accuracy of the predictions
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    '''

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        # Print the 10 features of highest weight!
        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()


    # need the classes of the test set for this
    '''
    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    '''

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, pred, train_time, test_time

def printSummary(results):
    print('=' * 80)
    print("Results summary (accuracies):")
    print('_' * 80)
    for i in range(0,2):
        for result in results:
            print(result[i], end="\t")
        print()
    print()

################################################################################
## main
################################################################################

if __name__ == "__main__":
    X_train, y_train, X_test = getData()
    print("data loaded")

    data_train_size_mb = size_mb(X_train)
    data_test_size_mb = size_mb(X_test)

    print("%d abstracts - %0.3fMB (training set)" % (
        len(X_train), data_train_size_mb))
    print("%d abtracts - %0.3fMB (test set)" % (
        len(X_test), data_test_size_mb))
    print()

    # define the categories
    categories = [
        'stats',
        'math',
        'physics',
        'cs'
    ]

    # get all the input opt parameters relevant to creating a vectorizer

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

    if opts.use_tf_idf:
        # a way to re-weight the count features such that extremely common words such
        # as "the" and "a" are less important than the less common ones
        print("Extracting features from the test data using a tfidf vectorizer")
        vectorizer = TfidfVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words, ngram_range=(1,maxNGramLength), sublinear_tf=True)
    else:
        print("Extracting features from the test data using a count vectorizer")
        vectorizer = CountVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words, ngram_range=(1,maxNGramLength))

    t0 = time()
    X_train = vectorizer.fit_transform(X_train)
    duration = time() - t0
    print("done vectorizing in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    feature_names = vectorizer.get_feature_names()

    # optionally select only the top k features
    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2)
        t0 = time()
        # the SelectKBest object is essentially a vectorizer that will select only the most k features of your input vectors
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test) # revectorize X_test
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X_test.shape)
        print()

    # covert feature_names to an ndarray
    feature_names = np.asarray(feature_names)

    if opts.tune:
        # parameters for tuning the models using grid search
        log_reg_tuned_parameters = [{'C': [1, 5, 10, 25, 100, 1000], 'tol': [1e-3, 1e-4, 1e-5], 'penalty': ['l1', 'l2']}]
        svc_tuned_parameters = [{'C': [1, 5, 10, 25, 100, 1000], 'tol': [1e-3, 1e-4, 1e-5]}]
        print("Tuning logistic regression and linear SVC models using 5-fold cross validation and over the following parameters:")
        print("Logistic regression:", log_reg_tuned_parameters)
        print("Linear SVM:", svc_tuned_parameters)
        print()
        logRegClf = GridSearchCV(LogisticRegression(), log_reg_tuned_parameters, cv=5)
        linearSvcClf = GridSearchCV(LinearSVC(), svc_tuned_parameters, cv=5)
    else:
        logRegClf = LogisticRegression()
        linearSvcClf = LinearSVC()

    # results has form [(clf_descriptor, predictions, training_time, testing_time),...]
    results = []
    for clf, name in (
            (logRegClf, "Logistic Regression"),
            (linearSvcClf, "Linear SVC"),
            (GaussianNB(), "Naive Bayes")):
        print('=' * 80)
        print(name)
        Xtr = X_train
        Xte = X_test
        ytr = y_train
        if type(clf) is GaussianNB:
            Xtr = Xtr.toarray()
            Xte = Xte.toarray()
        results.append(benchmark(clf, Xtr, ytr, Xte))

    print("Writing predictions to csv")
    postProcess(results)
    print()
