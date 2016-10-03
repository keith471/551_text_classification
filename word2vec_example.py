'''Classifies documents by learning a model using word2vec features'''
from __future__ import print_function
from __future__ import with_statement

import sys
import logging
from time import time
import string

from optparse import OptionParser

from gensim.models.word2vec import Word2Vec

import numpy as np

import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.svm import LinearSVC

# my modules
from preprocess import processData
from postprocess import writeResults

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lowercase",
              action="store_true",
              help="If set, the documents will be converted to lowercase.")
op.add_option("--lemmatize",
              action="store_true",
              help="If set, all words will be lemmatized.")
op.add_option("--remove_stop_words",
              action="store_true",
              help="If set, sklearn's list of English stop words will be removed.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

def loadVectors():
    print("loading word2vec vectors...")
    t0 = time()
    model = Word2Vec.load_word2vec_format('/Volumes/Seagate Backup Plus Drive/GoogleNews-vectors-negative300.bin', binary = True)
    loadTime = time() - t0
    print("word2vec vectors loaded in %0.3f seconds" % loadTime)
    print()

    # done "training" the model; we can do the following to trim uneeded memory
    t0 = time()
    print("trimming model memory...")
    model.init_sims(replace=True)
    trimTime = time() - t0
    print("trimmed memory in %0.3f seconds" % trimTime)
    print()

    return model

def processDocument(doc, lowercase, lemmatize):
    ''' takes all text of a document and returns just the words (punctuation removed, other than apostrophes) '''
    # must remove punctuation as we have no word2vec vectors for them
    nopunc = doc.translate(None, string.punctuation)
    tokens =  word_tokenize(nopunc)
    if lowercase:
        tokens = [w.lower() for w in tokens]
    if lemmatize:
        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(w) for w in tokens]
    return tokens

def benchmark(clf, X_train, y_train, X_test):
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
    clf_descr = str(clf).split('(')[0]
    return clf_descr, pred

if __name__ == "__main__":

    model = loadVectors()

    print("Processing data...")
    abstractsTrain, y, abstractsTest = processData()
    print("done")
    print()

    # X is our feature vectors
    print("Constructing feature vectors")
    X = []
    maxDocLength = 0
    for doc in abstractsTrain:
        features = []
        tokens = processDocument(doc, opts.lowercase, opts.lemmatize)
        wordCount = 0
        for token in tokens:
            if token in model:
                features.append(model[token])
                wordCount += 1
        X.append(features)
        if len(features) > maxDocLength:
            maxDocLength = len(features)
    print("done")
    print()

    # exend any feature vectors shorter than the longest document by zero vectors
    print("Extending feature vectors")
    zeros = np.zeros(model.vector_size)
    for vector in X:
        while len(vector) < maxDocLength:
            vector.append(zeros)
    print("done")
    print()

    clf = LinearSVC()
    clfDesc, pred = benchmark(clf, X, y, abstractsTest)

    writeResults(clfDesc, pred)