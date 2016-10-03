from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import processData
from sklearn.metrics import accuracy_score
from postprocess import writeResults

# accuracy =0.89331377858
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42))
                    ]);
                    
features_train_complete, labels_train_complete, features_test_competition = processData()

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

#writeResults('predicted_theo', predicted)