import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
import seaborn as sns
from seaborn import countplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


def Titanic_Logistic():
    print("Titanic Survival Prediction using Logistic Regression ")

    # step 1 : Load data
    titanic_data = pd.read_csv("TitanicDataset.csv")

    #first 5 entries from loaded dataset
#    print(titanic_data.head(5))

    print("Number of Passengers are "+str(len(titanic_data)))


    # step 2 : analyse data

    print("Visualization of Survived and non-Survived Passengers")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target).set_title("Survived and Non-Survived Passenger")
#    show()

    print("Visualization of Survived and Non-survived based on Gender")
    figure()
    target = "Survived"
    countplot(data=titanic_data, x = target, hue = titanic_data["Sex"]).set_title("Survived and Non-Survived based on Gender")
#    show()

    print("Visualization of Survived and Non-Survived based on passenger Class")
    figure()
    target = "Survived"
    countplot(data = titanic_data, x = target, hue=titanic_data["Pclass"]).set_title("Survived and Non-Survived based on Passenger Class")
#    show()

    print("Visualization of Survived and Non-Survived based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and Non-Survived Based on Age")
#    show()

    print("Visualization of Survived and Non-Survived based on Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and Non-Survived Based on Fare")
#    show()

    #step 3 : data cleaning
    titanic_data.drop("zero", axis= 1, inplace= True)
    print("First 5 entries from loaded data")
    print(titanic_data.head(5))


    print("Values of Sex Column")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
    # values of Sex after removing one field
    print(Sex.head())

    print("Values of Pclass")
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    #values of Pclass after removing one field
    print(Pclass.head())

    print("Values of dataSet after Concating new columns")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1) # column = 1
    print(titanic_data.head())

    print("Values of dataset after removing irrelevant columns")
    titanic_data.drop(["Sex","sibsp","Parch","Pclass","Embarked"],axis = 1, inplace = True)
    print(titanic_data.head())


    X = titanic_data.drop("Survived",axis = 1)
    y = titanic_data["Survived"]
    print(X.head())
    print(y.head())

    #step 4 : data training
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)

    #step 4 : data testing
    pred_y = logreg.predict(X_test)

    #calculating Accuracy
    print(f"confusion matrix : '\n' {confusion_matrix(y_test,pred_y)}")
    print(f"Accuracy : '\n' {accuracy_score(y_test,pred_y)}")
    print(f"classification report : '\n' {classification_report(y_test,pred_y)}")


def main():
    print("inside main")
    print("Logistic Regression on Titanic data set")
    Titanic_Logistic()

if __name__ == "__main__":
    main()

