import pandas as pd
from sklearn.svm import SVC


def one():

    train_df = pd.read_csv('./train_preprocessed.csv')
    test_df = pd.read_csv('./test_preprocessed.csv')

    train_df.info()
    #test_df.info()
    print("\n\n\n")

    print(train_df.corr())
    #print(test_df.corr())

    # The correlation between sex and survival was 0.543351 according to the "./train_preprocessed.csv"
    # Because this number is greater that one half ( >0.5 ) that means that it is considered a significant correlation.
    # Based on the correlations we are given, sex had the greatest correlation to survival rate among all features.
