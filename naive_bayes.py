'''Text classification using Naive Bayes'''
from __future__ import with_statement
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys, os, codecs
from time import time

from sklearn.feature_selection import SelectKBest, chi2

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from classifiers import NaiveBayes
from preprocess import readData
from postprocess import writeResults

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
op.add_option("--test",
              action="store", type="float", dest="test_fraction",
              help="Run on a fraction of the entire training corpus")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

if opts.max_n_gram_length:
    if opts.max_n_gram_length < 1:
        op.error("Max n-gram length must be positive")
        sys.exit()

if opts.test_fraction > 1.0 or opts.test_fraction < 0.0:
    op.error("The test fraction must be between 0.0 and 1.0")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

def selectChi2(X_train, y_train, X_test, feature_names=None):
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    # the SelectKBest object is essentially a vectorizer that will select only the most influential k features of your input vectors
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
    return X_train, X_test, feature_names

if __name__ == "__main__":

    ################################################################################
    # data loading
    ################################################################################

    docs_train, y_train, docs_test = getData()
    print("data loaded")

    data_train_size_mb = size_mb(docs_train)
    data_test_size_mb = size_mb(docs_test)

    print("%d abstracts - %0.3fMB (training set)" % (
        len(docs_train), data_train_size_mb))
    print("%d abtracts - %0.3fMB (test set)" % (
        len(docs_test), data_test_size_mb))
    print()

    ################################################################################
    # optionally extract only a portion of data for training
    ################################################################################

    if opts.test_fraction:
        percent = (opts.test_fraction * 100.0)
        print("Using only %.f percent of the training data" % percent)
        threshold = int(opts.test_fraction * len(docs_train))
        if threshold == 0:
            print("Fraction too small, please choose a larger fraction")
            print()
            sys.exit(1)
        docs_train = docs_train[:threshold]
        y_train = y_train[:threshold]
    print("Train set size: %d documents" % len(docs_train))
    print("Test set size: %d documents" % len(abstractsTest))
    print("done")
    print()

    ################################################################################
    # extract a development set
    ################################################################################

    print("Extracting development set from training set")
    docs_train, docs_dev, y_train, y_dev = train_test_split(docs_train, y_train, test_size=0.3, random_state=0)
    print("Using %d training examples and %d testing examples" % (len(docs_train), len(docs_dev)))
    print("done")
    print()

    # define the categories
    categories = [
        'stats',
        'math',
        'physics',
        'cs'
    ]

    ################################################################################
    # vectorize the training and development data
    ################################################################################

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

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        vectorizer = HashingVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words,
                                        ngram_range=(1,maxNGramLength), non_negative=True, n_features=opts.n_features)
        X_train = vectorizer.transform(docs_train)
    else if opt.use_tf_idf:
        # a way to re-weight the count features such that extremely common words such
        # as "the" and "a" are less important than the less common ones
        print("Extracting features from the test data using a tfidf vectorizer")
        vectorizer = TfidfVectorizer(lowercase=lowercase, tokenizer=tokenizer, stop_words=stop_words,
                                        ngram_range=(1,maxNGramLength), sublinear_tf=True, max_df=0.5)
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

    print("Extracting features from the development data using the same vectorizer")
    t0 = time()
    X_dev = vectorizer.transform(docs_dev)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_dev.shape)
    print()

    feature_names = vectorizer.get_feature_names()

    # optionally select only the top k features
    if opts.select_chi2:
        X_train, X_dev, feature_names = selectChi2(X_train, X_dev, feature_names)

    # covert feature_names to an ndarray
    feature_names = np.asarray(feature_names)

    ################################################################################
    # classification and prediction
    ################################################################################

    clf = NaiveBayes()

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
    cost = clf.cost(y_dev)
    print("Cost:")
    print(cost)
    print()
