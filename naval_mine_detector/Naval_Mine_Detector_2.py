# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:50:16 2020

@author: Vijay Narsing Chakole

"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv("C:/Users/hp/PycharmProjects/Project/Sonar.csv")
df
df.shape

df.head(3)

type(df.Class[1])
df.Class = df.Class.astype('category')

#Exploring data Analysis

from sklearn.manifold import MDS

mds = MDS(n_components = 2)

mds_data = mds.fit_transform(df.ix[:, :-1]) 


import matplotlib.pyplot as plt

plt.scatter(mds_data[:, 0], mds_data[:, 1], c = df.Class.cat.codes, s = 50)


#heatmap
heatmap = plt.pcolor(df.corr(), cmap='jet')
plt.colorbar(heatmap)




#Boxplot
df.plot.box(figsize=(12,4), xticks = [])

#density plot
df.plot.density(figsize=(6,60), subplots=True, yticks=[])

#Box plot suggest we should Standardize the data

from sklearn.preprocessing import  StandardScaler, RobustScaler

data, Class = df.ix[:, :-1], df.ix[:, -1]

data.head(3)

data_scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
#if there are gross outliers, we can use a robust routine

data_robust = pd.DataFrame(RobustScaler().fit_transform(data), columns= data.columns)

data_robust.head(3)


#Dimension reduction

from sklearn.decomposition import PCA

data.shape

pca = PCA()

pca

data_scaled_pca = pd.DataFrame(pca.fit_transform(data_scaled), columns=data.columns)

v= pca.explained_variance_ratio_
v
vc = v.cumsum()
vc

pd.DataFrame(list(zip(v,vc)),columns=['explained','cumsum']).head(10)


#let us  just use the principal components that explain at least  95% of total variance
import numpy as np

n_comps = 1 + np.argmax(vc> 0.95)
n_comps

data_scaled_pca = data_scaled_pca.ix[:, :n_comps]

data_scaled_pca

data_scaled_pca.shape

df_pca = pd.concat([data_scaled_pca, Class], axis=1)

df_pca
df_pca.shape

#Classification

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(data_scaled_pca, Class, test_size = 0.30, random_state = 42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)

#Using Support Vector Classifier and Grid Search

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

parameters =[{'kernel':['rbf'], 'gamma':[1e-3, 1e-4],
              'C':[1,10,100,1000]}, {'kernel':['linear'], 'C':[1, 10, 100, 1000]}]

#Do grid  Search with parallel jobs
clf = GridSearchCV(SVC(C=1), parameters, cv = 5, scoring='accuracy', n_jobs = -1)
clf.fit(X_train,y_train)

clf.best_params_
clf.best_score_
clf.best_estimator_
clf.best_index_
clf.classes_
clf.cv_results_



clf.best_params_
clf.best_score_

clf.score(X_test,y_test)

#Classification report

from sklearn.metrics import classification_report

actual_y, pred_y = y_test, clf.predict(X_test)

print(classification_report(actual_y, pred_y))


#Using Random Forests Classifier

from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test = train_test_split(data_scaled, Class, test_size = 0.30, random_state = 42)

parameters = [{'n_estimators': list(range(25, 201, 25)), 
               'max_features':list(range(2,15,2))}]


clf = GridSearchCV(RandomForestClassifier(), parameters, cv = 5, scoring = 'accuracy', n_jobs = -1)

clf.fit(X_train,y_train)

clf.best_params_

clf.score(X_test,y_test)




#Which Features are important ?

imp = clf.best_estimator_.feature_importances_

idx = np.argsort(imp)

plt.figure(figsize=(6,18))
plt.barh(range(len(imp)), imp[idx])
plt.yticks(np.arange(len(imp))+0.5, idx)



#Using pipeline
#For cross-validation (e.g. grid search for best parameters), 
#we often need to chain a series of steps and treat it as a single model. 
#This chaining can be done wiht a Pipeline object.

from sklearn.pipeline import Pipeline

X_train,X_test,y_train,y_test = train_test_split(data_scaled, Class, test_size = 0.30, random_state = 42)


scaler = StandardScaler()
pca = PCA()
clf = LogisticRegression()


pipe = Pipeline(steps=[('scaler',scaler),('pca',pca),('clf',clf)])
n_components = [20,30,40,50,60]
Cs = np.logspace(-4,4,1)

estimator1 = GridSearchCV(pipe, dict(pca_n_components = n_components,
                                    clf_C=Cs), n_jobs = -1)

estimator1.fit(X_train, y_train)


estimator1.best_estimator_.named_steps['pca'].n_components

estimator1.score(X_test, y_test)

y_true, y_pred = y_test, estimator1.predict(X_test)
print(classification_report(y_true, y_pred))

# %load_ext version_information
# %version_information numpy, pandas, sklearn
