# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 02:36:41 2020

@author: Vijay Narsing Chakole

"""
# spam message filter

# spam : irrelevant or unsolicited messages sent over the Internet,
# typically to a large number of users, 
# for the purposes of advertising, phishing, spreading malware, etc.

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import os 

os.chdir("C:\\Users\hp\\Desktop\\practicals_ml\\Main Projects\\spam message finder\\")
df = pd.read_table("SMSSpamCollection", header = None)
df.info()
df.columns = ['label', 'text']
df.columns

# separating target variable
y = df['label']
y.value_counts()

# feature engineering
# as target variable is binary ham or spam we have to convert it into 1 or 1
# to ensure compatibility with some sklearn models
# ham (not a spam) = 0 and spam  = 1

from sklearn.preprocessing import LabelEncoder
labelencode = LabelEncoder()
y = labelencode.fit_transform(y)
y

# separating message data
raw_text = df['text']

type(raw_text)
# Text preprocessing
# Normalization
# in spam mesaage
# Replace email addresses with 'emailaddr'
# Replace URLs with 'httpaddr'
# Replace money symbols with 'moneysymb'
# Replace phone numbers with 'phonenumbr'
# Replace numbers with 'numbr'

processed = raw_text.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 
                                 'emailaddr')

processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                  'httpaddr')

processed = processed.str.replace(r'£|\$', 'moneysymb') 

processed = processed.str.replace(
    r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    'phonenumbr')    

processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')  


# we'll remove all punctuation since "today" and "today?"
# refer to the same word. In addition, 
# let's collapse all whitespace (spaces, line breaks, tabs) into 
# a single space. Furthermore,
# we'll eliminate any leading or trailing whitespace.


processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')

# lowercase the entire corpus
processed = processed.str.lower()

# Removing the stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# This list of stop words is literally stored in a Python list.
# If instead we convert it to a Python set, iterating over the stop words will go much faster,
# and shave time off this preprocessing step.

processed  = processed.apply(lambda x : ' '.join(
    term for term in x.split() if term not in set(stop_words) ))

# Stemming
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
porter.stem("called")

"""
import nltk
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('done', wordnet.VERB))
porter.stem("done")

"""

processed = processed.apply(lambda x : ' '.join(
    porter.stem(term) for term in x.split()))

#**************************************************************
corpus = []
import re
#newdata = ["Congratulationals !!!! you won price of $100000000000, please collect your Price as soon as possible"]

type(raw_text)
for i in range(len(raw_text)):
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 
                                 'emailaddr', raw_text[i]) 
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     cleaned)
    cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
    cleaned = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.lower()
    cleaned = cleaned.split()
    ps = PorterStemmer()
    cleaned = [ps.stem(word) for word in cleaned if not word in set(stopwords.words('english'))]
    cleaned = ' '.join(cleaned)
    corpus.append(cleaned)
    
  
corpus
# Feature engineering 
# creating Bag of word Model
# A matrix that contains a lot of zeros called sparse matrix    
# tokennization means taking all the different word of review and making column for each these word
  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
type(X)

# spliting data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 42)

# Fitting the naive bayes on training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm
score = accuracy_score(y_test, y_pred)
score

#*******************************************************************
# Fitting Logistic Regression Model = 98%

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm
score = accuracy_score(y_test, y_pred)
score

#*******************************************************************
"""
ngram_rangetuple (min_n, max_n), default=(1, 1)
The lower and upper boundary of the range of n-values for 
different n-grams to be extracted.
 All values of n such that min_n <= n <= max_n will be used.
 For example an ngram_range of (1, 1) means only unigrams,
 (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.

"""
# Implementing tf-idf statistic
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range = (1, 2))

n_grams = vectorizer.fit_transform(corpus)
n_grams.shape

# Training and evaluating a model
X_train, X_test, y_train, y_test = train_test_split(n_grams, y, test_size = 0.20, stratify = y, random_state = 42)

from sklearn.svm import LinearSVC

classifier = LinearSVC(loss= 'hinge')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


"""
The F score, also called the F1 score or F measure, is a measure of a test's accuracy. ... The F score reaches the best value, meaning perfect precision and recall, at a value of 1. The worst F score, which means lowest precision and lowest recall, would be a value of 0.


F1 score - F1 Score is the weighted average of Precision and Recall.
 Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution
"""

from sklearn.metrics import f1_score
F1_score = f1_score(y_test, y_pred)
F1_score

cm = confusion_matrix(y_test, y_pred)
cm
score = accuracy_score(y_test, y_pred)
score


cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=[['actual', 'actual'], ['ham', 'spam']],
    columns=[['predicted', 'predicted'], ['ham', 'spam']]
)
cm


#*******************************************************************

# Using nested cross-validation to minimize information leakage
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


param_grid = [{'C' : np.logspace(-4, 4, 20)}]

grid_search = GridSearchCV(estimator = LinearSVC(loss= 'hinge'), param_grid = param_grid,
                           cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 42),
                           scoring = 'f1',
                           n_jobs = -1
                           )



scores = cross_val_score(estimator = grid_search, X = n_grams, y = y, 
                         cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 42),
                           scoring = 'f1',
                           n_jobs = -1 )

scores
scores.mean()

#******************************************************************

# what terms are the top predictors of spam ?
grid_search.fit(n_grams, y)
final_classifier = LinearSVC(loss = 'hinge', C = grid_search.best_params_['C'])
final_classifier.fit(n_grams, y)

predictor_words = pd.Series(final_classifier.coef_.T.ravel(),
                            index = vectorizer.get_feature_names()
                            ).sort_values(ascending = False)[:20]


predictor_words

#**********************************************************************

import re

type(cleaned)
def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     cleaned)
    cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
    cleaned = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
    return ' '.join(
        porter.stem(term) 
        for term in cleaned.split()
        if term not in set(stop_words)
    )


#***************************************************************************

#*******************************************************************
# To finish up, let's write up a function that'll decide whether a string is spam or not, and apply it on the hypothetical message from earlier.

def spam_filter(message):
    if final_classifier.predict(vectorizer.transform([preprocess_text(message)])):
        return 'spam'
    else:
        return 'Not spam'




example = "Congratulationals !!!! phonenumbr numbrp moneysymbnumbr you won price of $100000000000, please collect your Price as soon as possible"
type(example)

example = ' 8308534844 vijaychakole23@gmail.com , Hi Vijay, you won $10000000000'
example = 'hello world good morning'

spam_filter(example)


