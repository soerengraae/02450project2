'''
A regularization parameter is used to avoid overfitting.
'''

from data_load import getTargets
from data_standardization import X as X_num, missing_values as missing_values_num
from data_encoding import X as X_cat, missing_values as missing_values_cat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

missing_values = list(set(missing_values_num).union(set(missing_values_cat)))

X = pd.concat([X_num, X_cat], axis=1)
y = getTargets()

X = X.drop(missing_values)
y = y.drop(missing_values)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split the dataset into training (66%) and test datasets
kf = KFold(n_splits=10)

error_rates = []

def predict(X_train, X_test, y_train):
    model = LogisticRegression(multi_class='multinomial', random_state=0, max_iter=1000)

    # Train the multimonial regression model
    model.fit(X_train, y_train)

    # Predict the class for all the samples in the test dataset
    yhat = model.predict(X_test)

    return yhat