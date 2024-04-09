from sklearn.model_selection import KFold

#Implement two-level cross-validation We will use 2-level cross-validation to compare the models with K1 = K2 = 10 folds As a baseline model, we will apply a linear regression model with nofeatures, i.e. it computes the mean of y on the training data, and use this value to predict y on the test data. Make sure you can fit an ANN model to the data. As complexity-controlling parameter for the ANN, we will use the number of hidden units5 h. Based on a few test-runs, select a reasonable range of values for h (which should include h = 1), and describe the range of values you will use for h and λ.

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from data_fetch import automobile_id, getTargets, getFeatures, categorical_features
import data_encoding
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

# Load the data
X_cat = data_encoding.encode(getFeatures(automobile_id)[categorical_features])
y = getTargets(automobile_id)

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_cat):
    # Split the data into training and test sets
    X_train, X_test = X_cat.iloc[train_index], X_cat.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Impute missing values in the training data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)