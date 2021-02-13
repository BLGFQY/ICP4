from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd


def two():
    X = pd.read_csv("glass.csv")
    Y = pd.read_csv("glass.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(y_train)

    print("Number of mislabeled points out of a total %d points : %d"
               % (X_test.shape[0], (y_test != y_pred).sum()))
