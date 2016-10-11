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
from postprocess import writeResults

from math import log

from collections import defaultdict

def getData():
    return processData()

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(word) for word in word_tokenize(doc)]

def selectChi2(X_train, y_train, X_test, feature_names=None):
    # the SelectKBest object is essentially a vectorizer that will select only the most influential k features of your input vectors
    ch2 = SelectKBest(chi2, k=300)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test) # revectorize X_test
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    return X_train, X_test, feature_names

#object representation of decision tree node
class dtreenode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col # column index of criteria being tested
        self.value=value # vlaue necessary to get a true result
        self.results=results # dict of results for a branch, None for everything except endpoints
        self.tb=tb # true decision nodes 
        self.fb=fb # false decision nodes

#usage: rows = X_train_arr
def splitset(rows,column,value=1):
	# Make a function that tells us if a row is in the first group 
	# (true) or the second group (false)
	split_function=lambda row:row[column]>=value
	# Divide the rows into two sets and return them
	set1=[row for row in rows if split_function(row)] 
	set2=[row for row in rows if not split_function(row)]
	indices = [i for i in range(0,len(rows)) if split_function(rows[i])] #get the indices which return true
	return (set1,set2, indices)

#usage: rows = y_train, categories = categories
def outputcounts(rows, categories):
    results = defaultdict(lambda: 0)
    for row in rows:
    	r = row
        results[r]+=1
    dd = dict(results)
    #filter out the anomalies (there's always one or two)
    for k in dd.keys():
    	if k not in categories:
    		dd.pop(k, None)
    return dd

#usage: rows = y_train, categories = categories
def entropy(rows, categories):
    log2=lambda x:log(x)/log(2) 
    results=outputcounts(rows,categories)
    # Now calculate the entropy
    ent=0.0
    for r in results.keys():
        p=float(results[r])/len(rows) 
        ent=ent-p*log2(p)
    return ent

#usage: rows = X_train_arr, output = y_train, categories = categories
def buildtree(rows, output, categories, entropy=entropy):
    if len(rows) == 0: return dtreenode()
    current_score = entropy(output, categories)
    best_gain = 0.0
    best_col = None
    best_sets = None
    column_count = len(rows[0])
    for col in range(0, column_count):     
        #OPTIONAL: TRY TO DIVIDE ON EVERY VALUE
        # column_values = set([row[col] for row in rows]) <- possible values
        # for value in column_values:
        #     set1, set2 = splitset(rows, col, value)
        #OTHERWISE: DIVIDE ON VALUE >=1 
        set1, set2, indices = splitset(rows, col)
        # Information gain
        p = float(len(set1)) / len(rows) 
        set1_ys = [output[i] for i in indices]
        set2_ys = [output[i] for i in list(set(range(0,len(output)))-set(indices))]
        gain = current_score - p*entropy(set1_ys, categories) - (1-p)*entropy(set2_ys, categories)
        if gain > best_gain and len(set1) > 0 and len(set2) > 0:
            best_gain = gain
            best_col = col #if dividing on every value: best_col = (col, value)
            best_sets = (set1, set2)
            best_sets_ys = (set1_ys,set2_ys)
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0],best_sets_ys[0],categories)
        falseBranch = buildtree(best_sets[1], best_sets_ys[1], categories)
        return dtreenode(col=best_col, value=1,
                tb=trueBranch, fb=falseBranch)
    else:
        return dtreenode(results=outputcounts(output,categories))

def printtree(tree, feature_names, indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print(str(tree.results))
    else:
        # Print the criteria
        print('Column ' + feature_names[tree.col] + ' >= '+ str(tree.value)+'? ')
        # Print the branches
        print(indent+'True->',)
        printtree(tree.tb,feature_names,indent+'  ')
        print(indent+'False->',)
        printtree(tree.fb,feature_names,indent+'  ')

X_train, y_train, X_test = getData()
print("data loaded")

cats = [
    'stat',
    'math',
    'physics',
    'cs'
]

maxNGramLength = 1
lowercase = True
# tokenizer = LemmaTokenizer()
stop_words = 'english'
print("Using n-grams of up to %d words in length" % maxNGramLength)
print("Extracting features from the test data using a count vectorizer")
vectorizer = CountVectorizer(lowercase=lowercase, stop_words=stop_words, ngram_range=(1,maxNGramLength))


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names()



X_train, X_test, feature_names = selectChi2(X_train, y_train, X_test, feature_names)
X_train_arr = X_train.toarray()
X_test_arr = X_test.toarray()
#converting sparse matrices to full matrices
#a lot more inneficient, but working with sparse matrices is a headache :(

#for testing:
# X_train_arr = X_train_arr[0:100]
# y_train = y_train[0:100]

print("Building the decision tree")
dtree = buildtree(rows=X_train_arr, output=y_train, categories=cats) #<-magic happens here
printtree(dtree, feature_names)