import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xlsxwriter


def Logistic_Regression():
    print("inside data preparation")

    train_df =pd.read_csv("train.csv")
   # print(train_df.head())

    cols = train_df.columns.values
    print(cols)

    newcols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
 'Ticket', 'Fare', 'Cabin', 'Embarked','Survived']

    # reindexing : keep the target column at last
    train_df = train_df.reindex(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
 'Ticket', 'Fare', 'Cabin', 'Embarked','Survived'], axis=1)
    print(train_df.head())

    test_df = pd.read_csv("test.csv")
   # print(test_df.head())

    gender_submission = pd.read_csv("gender_submission.csv")
   # print(gender_submission.head())

    #merge test_df and gender_submission on keyPassengerId

    test_df = test_df.merge(gender_submission, on="PassengerId")
    print(test_df.head())

    frames = [train_df,test_df]
    titanic_df = pd.concat(frames)
    print(titanic_df.head())
    print(train_df.shape)
    print(test_df.shape)
    print(titanic_df.shape)
    titanic_df.to_excel

    #removing unneccesary columns from our data set

    titanic_df = titanic_df.drop(['PassengerId', 'Name', 'SibSp', 'Parch','Ticket',
   'Cabin'], axis = 1)

    print(titanic_df.head())
    titanic_df = titanic_df.dropna()
    print(titanic_df.head())
    print(titanic_df.count())

    #creating dummies variable of our data set columns
    dummies_df = titanic_df
    # converting numbers into dummies variable
    # first we have to find its range means its minimum and maximum value
    # and then make group of it within some range

    print(dummies_df["Age"].min())
    Age_min = dummies_df["Age"].min()
    print(dummies_df["Age"].max())
    Age_max = dummies_df["Age"].max()

    bins = np.linspace(Age_min, Age_max, 8)
    print(bins)
    which_bins = np.digitize(dummies_df["Age"], bins=bins)
    print(which_bins)
    dummies_df["Age"] = which_bins
    print(dummies_df)

    print(dummies_df["Fare"].min())
    print(dummies_df["Fare"].max())

    Fare_min = dummies_df["Fare"].min()
    Fare_max = dummies_df["Fare"].max()

    bins = np.linspace(Fare_min, Fare_max, 12)
    print(bins)
    which_bins = np.digitize(dummies_df["Fare"], bins=bins)
    dummies_df["Fare"] = which_bins
    print(dummies_df)
################################################################################

    #converting numbers range into string
    dummies_df["Age"] = dummies_df["Age"].astype(str)
    dummies_df["Fare"] = dummies_df["Fare"].astype(str)
    dummies_df["Pclass"] = dummies_df["Pclass"].astype(str)


#*********************************************************************************
    titanic_dummies_df = pd.get_dummies(dummies_df,drop_first=True)
    print(titanic_dummies_df)

    # its for Logit()
   # titanic_dummies_df = titanic_dummies_df.drop(['Age_8','Fare_12'],axis=1)
   # print(titanic_dummies_df.columns)

    titanic_dummies_df.to_excel("result.xlsx")

    X = titanic_dummies_df.iloc[:,1:19]
    y = titanic_dummies_df.iloc[:,0]
    print(titanic_dummies_df.columns)
    print(X)
    print(y)

    #step 3 spliting data and training model

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    logreg = LogisticRegression()

    logreg.fit(X_train,y_train)
    pred_y = logreg.predict(X_test)

   # import statsmodels.api as sm
   # logit_model = sm.Logit(y,X)
   # result = logit_model.fit()
   # print(result.summary2())
    print(f"Confusion Matrix : '\n' {confusion_matrix(y_test,pred_y)}")
    print(f"classification report : '\n' {classification_report(y_test,pred_y)}")
    print(f"Accuracy : '\n' {accuracy_score(y_test,pred_y)}")




def main():
    print("inside main")
    Logistic_Regression()

if __name__ == "__main__":
    main()