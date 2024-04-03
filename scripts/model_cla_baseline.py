'''
The baseline clasification model computes the largest class in the dataset,
and predicts that class for all the samples in the test dataset.

10-folds cross-validation is used.
'''

from data_load import getTargets
from data_standardization import X as X_num, missing_values as missing_values_num
from data_encoding import X as X_cat, missing_values as missing_values_cat
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

missing_values = list(set(missing_values_num).union(set(missing_values_cat)))

X = pd.concat([X_num, X_cat], axis=1)
y = getTargets()

X = X.drop(missing_values)
y = y.drop(missing_values)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Initialize an empty array to store the predicted labels
yhat = np.zeros_like(y)

# Perform 10-fold cross-validation
kf = KFold(n_splits=10)
error_rates = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Compute the largest class in the training dataset
    largest_class = y_train.value_counts().idxmax()

    # Predict the largest class for all the samples in the test dataset
    yhat[test_index] = [largest_class] * len(test_index)

    # Calculate the error rate for this fold
    error_rate = np.mean(yhat[test_index] != y_test)
    error_rates.append(error_rate)

# Print the error rates for each fold
for i, error_rate in enumerate(error_rates):
    print(f"Error rate for fold {i+1}: {error_rate}")