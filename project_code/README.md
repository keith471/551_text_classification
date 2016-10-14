# README

## General notes
To run any of the scripts, you should have the datasets up one directory and within another directory called `datasets`, i.e. ../datasets/, or modify the method `readData` in `preprocess.py`

## Naive Bayes
- For a list of options, run `python naive_bayes.py --h`
- As an example, you might run `python naive_bayes.py --lowercase --remove_stop_words --chi2_select=300 --devset` to preprocess by converting to lowercase and removing English stop words, then training a classifier on the top 300 features and measuring its performance against 30% of the training data

## Decision tree
Run with `python dtrees.py`. That's it!

## Scikit learn

