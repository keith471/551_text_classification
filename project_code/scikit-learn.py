from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import processData
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from postprocess import writeResults
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# accuracy = 0.926406438026
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), max_features=100000)),
                     ('tfidf', TfidfTransformer(sublinear_tf=True)),
                     ('clf', SGDClassifier(loss='modified_huber', penalty='l2', n_iter=6))
                     #('classifier', MultinomialNB())
                    ]);
                    
features_train_complete, labels_train_complete, features_test_competition = processData()

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
               }

sizeOfTraining = int(round(len(features_train_complete) * 0.7))

# test accuracy
features_train = features_train_complete[:sizeOfTraining]
labels_train = labels_train_complete[:sizeOfTraining]

features_test = features_train_complete[sizeOfTraining:]
labels_test = labels_train_complete[sizeOfTraining:]

# do actual predictions
#features_train = features_train_complete
#labels_train = labels_train_complete

#features_test = features_test_competition

print "Training"
text_clf.fit(features_train, labels_train)

print "Predicting"
predicted = text_clf.predict(features_test)

print accuracy_score(labels_test, predicted)

#writeResults(predicted, 'predicted_theo.csv')
