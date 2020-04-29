#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival dataset 

# In[ ]:





# In[ ]:





# In[176]:



# import libraries
# For data loading and manipulation
import pandas as pd
# For maths calculations
import numpy as np

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For splitting data into train set and test set
from sklearn.model_selection import train_test_split

# Classification Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Hyperparameter Tuning
# Hyperparameter Tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
# Hyperparameter Tuning using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

# For evaluating various metrics 
from sklearn.model_selection import cross_val_score

# For ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve


# Exporting and importing a trained model
from joblib import dump, load
import pickle


# In[177]:


dataset = pd.read_csv('TitanicDataset.csv')


# In[178]:


dataset.head()
len(dataset)
type(dataset)
dataset.shape
dataset.index
dataset.head()
dataset.columns
dataset.dtypes
dataset.info()
dataset.describe()
dataset.tail()


# In[179]:


dataset.head()


# In[180]:


dataset.columns


# In[181]:


len(dataset)


# In[182]:


type(dataset)


# In[183]:


dataset.index


# In[184]:


dataset.dtypes


# In[185]:


dataset.info()


# In[186]:


dataset.shape


# In[187]:


# count the number of NaN values in each column
dataset.isnull().sum()


# In[188]:


#dataset.head(20)
# drop na values (delete rows)
dataset.dropna(inplace = True, axis = 0) 


# In[189]:



dataset.isnull().sum() 


# In[ ]:





# In[190]:


# creating X and y
X = dataset.drop('Survived', axis = 1)
y = dataset.Survived


# In[191]:


type(X)


# In[192]:


type(y)


# In[193]:



y.value_counts()


# In[194]:



# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
scatter = ax.scatter(dataset["Age"], 
                     dataset["Fare"], 
                     c=dataset["Survived"])

# Customize the plot
ax.set(title="Survival Rate",
       xlabel="Age",
       ylabel="Fare");
ax.legend(*scatter.legend_elements(), title="Target")

# Add a meanline
ax.axhline(dataset["Fare"].mean(),
           linestyle="--");


# In[195]:


# spliting dataset into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[196]:


# dropping na values
X_train.isnull().sum()


# In[197]:


# View different shape of training and testing set
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[198]:


# Building Model
from sklearn.ensemble import RandomForestClassifier


# In[199]:


# Instantiate an instance of RandomForestClassifier as classifier
classifier_rf = RandomForestClassifier()


# In[200]:


# We'll leave the hyperparameters as default to begin with...
classifier_rf.get_params()


# In[201]:


# fitting the model
classifier_rf.fit(X_train, y_train)


# In[202]:


X_train.shape


# In[203]:


# Use the fitted model to make predictions on the test data and
# save the predictions to a variable called y_pred

y_pred = classifier_rf.predict(X_test)


# In[204]:


# Evaluate the fitted model on the training set using score() function
classifier_rf.score(X_train, y_train)


# In[205]:


# Evaluate the fitted model on the testing set using score()
classifier_rf.score(X_test,y_test)


# In[206]:


# hyperparameter tuning for RandomForestClassifier
# n_estimations : means number of decision tree

for i in range(10,100,10):
    print(f"trying model with {i} estimators...")
    model = RandomForestClassifier(n_estimators = i, random_state = 42).fit(X_train, y_train)
    print(f"model accuracy on test set : {model.score(X_test, y_test):.2f}")
   # print(f"cross-validation score : {np.mean(cross_val_score(model, X, y, cv = 5)) * 100:.2f}")
    print("")
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# # Experimenting with differnt classification Algorithms

# In[207]:


from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


models = {"LinearSVC" : LinearSVC(),
         "KNN" : KNeighborsClassifier(),
         "LogisticRegression" : LogisticRegression(),
         "RandomForestClassifier" : RandomForestClassifier()}


# In[208]:


# created empty dictionary results to store the results
results = {}


# In[209]:


# Loop through the models dictionary items, 
## fitting the model on the training data
# and appending the model name and model score on the test data 
# to the results dictionary

# for getting same results across the multiple times execution code
np.random.seed(42)

for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = ((model.score(X_test, y_test)) * 100)
    


# In[210]:


results


# In[ ]:





# In[211]:


# make our results a little more Visual

# Create a pandas dataframe with the data as the values of the results dictionary,
# the index as the keys of the results dictionary and a single column called accuracy.
# Be sure to save the dataframe to a variable.

results_df = pd.DataFrame(results.values(),
                         results.keys(),
                         columns = ['Accuracy'])

results_df


# In[212]:


results_df.shape


# In[217]:


# create bar plot of the results dataframe using plot.bar

results_df.plot.bar()


# results in the RandomForestClassifier model perfoming the best 
# 
# RandomForestClassifier has highest accuracy : 87 %
# 
# 
# let's first build model using RandomForestClassifier by tuning its parameters
# 
# and then go for second highest accuracy : Logistic Regression 
# 
# let's try second highest accuracy model LogisticRegression  
# 
# LogisticRegression  has second highest accuracy : 85 %
# 
# results in the LogisticRegression model perfoming the better 
# 
# Let's tune its hyperparameters and see if we can improve it.
#  
# use one of them while parameter tuning
#  
#     1) Exhaustively with GridSearchCV or
# 
#     2) Randomly with RandomizedSearchCV.

# In[218]:


# Hyperparameter Tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV


# In[219]:


# let's take a look on parameters of RandomForestClassifier
RandomForestClassifier().get_params()


#  Define parameters to search over
#  
#  only this thing (param_grid) will be change according to algorithms
#  
#  we have to define hyper parameter as per provided by specific algorithms
# 
# 

# In[228]:


# Define parameters to search over
param_grid = {"n_estimators" : [i for i in range(10,100,10)]}

# setup the gridsearch

grid = GridSearchCV(estimator = RandomForestClassifier(random_state = 42),
                   param_grid = param_grid,
                   cv = 5)


# In[229]:


# fit grid to data
grid.fit(X, y)


# In[230]:


# find the best parameters
grid.best_params_


# we get the best result when we create 10 decision trees in RandomForestClassifier Algorithm.

# In[233]:


# set  the model to the best estimator
classifier = grid.best_estimator_


# In[234]:


classifier


# In[235]:


# fit the best model
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)


# In[236]:


# Find best model score using test set
classifier.score(X_test, y_test)


# we are successfully improved 1% accuracy by tuning hyper-parameters 
# 
# RandomForestClassifier Accuracy before tuning : 87 %
# 
# RandomForestClassifier Accuracy After tuning : 88 %
#     

# # hyperparameter tuning using RandomizedSearchCV

# In[237]:


# different LogisticRegression hyperprarameters
log_reg_grid = {"C" : np.logspace(-4, 4, 20),
               "solver" : ["liblinear"]}



from sklearn.model_selection import RandomizedSearchCV
# Setup an instance of RandomizedSearchCV 
# with a LogisticRegression() estimator,
# our log_reg_grid as the param_distributions, a cv of 5 
# and n_iter of 5.
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(estimator = LogisticRegression(),
                                param_distributions = log_reg_grid,
                                cv = 5,
                                n_iter = 5,
                                verbose = 1)


# In[245]:


# Fit the instance of RandomizedSearchCV
rs_log_reg.fit(X_train, y_train)


# In[246]:


# finding the best parameters of RandomizedSeachCV using instance the best_params_

rs_log_reg.best_params_


# In[247]:


# Score the instance of RandomizedSearchCV using the test data
rs_log_reg.score(X_test, y_test)


# In[248]:


# Instantiate a LogisticRegression classifier using  
# the best hyperparameters from RandomizedSearchCV

classifier_log_reg = LogisticRegression(solver = "liblinear", C = 206.913808111479)


# In[249]:


# Fit the new instance of LogisticRegression with 
# the best hyperparameters on the training data 
classifier_log_reg.fit(X_train, y_train)


# # Model Evaluation

# In[250]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve


# In[251]:


# make predictions on test data and save them
y_pred = classifier_log_reg.predict(X_test)


# In[252]:


# return probrabilities rather than labels
classifier_log_reg.predict_proba(X_test)


# In[253]:


# compare predict() and predict_proba
# taking only first 5 entries

classifier_log_reg.predict(X_test[:5])


# In[254]:


classifier_log_reg.predict_proba(X_test[:5])


# In[255]:


# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[256]:


y_test.value_counts()


# In[257]:


# import seaborn for improving visualisation of confusion_matrix
import seaborn as sns


# In[258]:


# make confusion matrix more visual
def plot_conf_mat(y_test, y_pred):
    """plots confusion matrix using seaborn's heatmap"""
    
    fig, ax = plt.subplots(figsize = (3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot = True, # Anotate the boxes
                     fmt = "d", # digit
                    cbar = False)
    
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    
    # Fix the broken annotations (this happened in Matplotlib 3.1.1)
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)


# In[259]:


# call plot_conf_mat function
plot_conf_mat(y_test, y_pred)


# In[260]:


# classification report
classification_report(y_test, y_pred)
# make classification report more visual
class_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict = True))
class_report


# In[261]:


# precision score
precision_score(y_test, y_pred)


# In[262]:


# recall score
recall_score(y_test, y_pred)


# In[263]:


# F1 score
f1_score(y_test, y_pred)


# # plot ROC Curve

# In[264]:


plot_roc_curve(estimator = classifier_rf, X= X_test, y = y_test)


# In[265]:


# another method to plot a ROC curve


# In[266]:


# Make predictions with probabilities
y_probs = classifier_rf.predict_proba(X_test)

# Keep the probabilites of the positive class only
y_probs = y_probs[:, 1]

# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Check the false positive rate
fpr

def plot_roc_curve_fun(fpr, tpr):
    """
    Plots a ROC curve given the false positve rate (fpr) and 
    true postive rate (tpr) of a classifier.
    """
    # plot ROC curve
    plt.plot(fpr, tpr, color = 'orange', label = 'ROC')
    # Plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color = 'darkblue', linestyle = '--', label = 'Guessing')
    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciever Operating Characteristics (ROC) Curve')
    plt.legend()
    plt.show()
    
plot_roc_curve_fun(fpr, tpr)


# The maximum ROC AUC score you can achieve is 1.0 and generally,
# the closer to 1.0, the better the model.

# In[267]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_probs)


# The most ideal position for a ROC curve to run along the top left corner of the plot. 
# This would mean the model predicts only true positives and no false positives. And would result in a ROC AUC score of 1.0.
# You can see this by creating a ROC curve using only the y_test labels.

# In[268]:


# plot perfect ROC curve using y_test same time
fpr, tpr, thresholds = roc_curve(y_test, y_test)
plot_roc_curve_fun(fpr, tpr)
# perfect ROC AUC score
roc_auc_score(y_test, y_test) # 1.0 : In reality, a perfect ROC curve is unlikely.


# ***************************************************************************

# # cross-validation
# We can calculate various evaluation metrics using cross-validation
# 
# using Scikit-Learn's cross_val_score() function along with the scoring parameter.

# In[269]:


# import cross_val_score
from sklearn.model_selection import cross_val_score

# By default cross_val_score returns 5 values (cv = 5)

cross_val_acc = cross_val_score(classifier_rf,
                               X,
                               y,
                               scoring = "accuracy",
                               cv = 5)


# In[270]:


cross_val_acc


# In[271]:


# find overall accuracy using mean()
cross_val_acc = cross_val_acc.mean()
cross_val_acc


# In[272]:


# Find the cross-validated precision
cross_val_precision = np.mean(cross_val_score(classifier,
                                              X,
                                              y,
                                              scoring="precision",
                                              cv=5))

cross_val_precision


# In[273]:


# Find the cross-validated recall
cross_val_recall = np.mean(cross_val_score(classifier,
                                           X,
                                           y,
                                           scoring="recall",
                                           cv=5))

cross_val_recall


# In[274]:


# Find the cross-validated F1 score
cross_val_f1 = np.mean(cross_val_score(classifier,
                                       X,
                                       y,
                                       scoring="f1",
                                       cv=5))

cross_val_f1


# # Drawing special graph for Algorithm comparison

# In[275]:


# it specially for drawing special graph for comparing accuracy,presion
# recall, f1_score of specific model together to other model
from sklearn.metrics import accuracy_score


# In[276]:


def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2), 
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict


# In[277]:


# creating baseline matrics
# Make predictions
# random forest with trees 10
# classifier from GridSearchCV
y_pred = classifier.predict(X_test)

baseline_metrics = evaluate_preds(y_test, y_pred)
baseline_metrics
#*************************************************************************


# In[278]:


classifier


# In[279]:


# Create a second classifier
# RandomForest with trees 100
clf_2 = RandomForestClassifier(n_estimators=100)
clf_2.fit(X_train, y_train)
# Make predictions
y_preds_2 = clf_2.predict(X_test)
# Evaluate the 2nd classifier
clf_2_metrics = evaluate_preds(y_test, y_preds_2)
clf_2_metrics


# In[280]:


clf_2


# In[282]:


# create third classifier with Logistic regression

classifier_log_reg.fit(X_train, y_train)

y_pred_log_reg =  classifier_log_reg.predict(X_test)

logistic_metrics = evaluate_preds(y_test, y_pred_log_reg )
logistic_metrics


# In[283]:


# create fourth classifier with KNN

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_metrics = evaluate_preds(y_test, y_pred_knn)
knn_metrics


# In[284]:


# create fifth classifier with SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
svc_metrics = evaluate_preds(y_test, y_pred_svc)
svc_metrics


# In[ ]:





# In[ ]:





# In[ ]:





# In[285]:


compare_metrics = pd.DataFrame({"baseline" : baseline_metrics,
                                "rf_t100" : clf_2_metrics,
                                "Logistic_reg" : logistic_metrics,
                                "KNN" : knn_metrics,
                                "SVC" : svc_metrics
                                })

compare_metrics.plot.bar(figsize = (10, 5))


# RandomForestClassifier with 10 (baseline) trees and RandomForestClassifier with 100 trees gives best accuracy.
# 
# RandomForestClassifier with 10 tress (baseline) gives best precision score.
# 
# so we go through with baseline model that is RandomForestClassifier with 20 tress.
# 
# 

# # Exporting and importing a trained model

# Once you've trained a model, you may want to export it and save it to file so you can share it or use it elsewhere.
# 
# One method of exporting and importing models is using the joblib library.
# 
# In Scikit-Learn, exporting and importing a trained model is known as model persistence.

# In[172]:


# Import the dump and load functions from the joblib library
from joblib import dump, load


# In[286]:


# use the dump function to export the trained model to file
dump(classifier, 'trained-classifier.joblib')


# In[287]:


# Use the load function to import the trained model you just exported
# Save it to a different variable name to the origial trained model
loaded_clf = load("trained-classifier.joblib")


# In[288]:


# Evaluate the loaded trained model on the test data
loaded_clf.score(X_test, y_test)


# In[ ]:




