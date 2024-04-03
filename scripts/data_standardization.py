'''
The purpose of this script is to standardize the numerical features in the dataset.
Import X from this script to get the standardized numerical features of the dataset with missing values dropped.
'''

from data_load import getFeatures, missingValues, numerical_features
import numpy as np

X = getFeatures()[numerical_features]
missing_values = missingValues(X)

# Logistic transformation of the data
X = np.log(X)

# Standardize the data
X = (X - X.mean()) / X.std()