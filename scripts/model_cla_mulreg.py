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

model = LogisticRegression(multi_class='multinomial', random_state=0, max_iter=1000)

# Split the dataset into training (66%) and test datasets
kf = KFold(n_splits=10)

error_rates = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Done to avoid DataConversionWarning
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Train the multimonial regression model
    model.fit(X_train, y_train)

    # Predict the class for all the samples in the test dataset
    yhat = model.predict(X_test)

    errors = np.zeros_like(y_test, dtype=bool)
    for i in range(len(y_test)):
        errors[i] = y_test[i] != yhat[i]
    error_rate = errors.mean()

    error_rates.append(error_rate)

# Print the error rates for each fold
for i, error_rate in enumerate(error_rates):
    print(f"Error rate for fold {i+1}: {error_rate}")