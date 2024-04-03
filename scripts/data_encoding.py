'''
The purpose of this script is to encode the categorical features in the dataset.
Import X from this script to get the encoded categorical features of the dataset, with missing values dropped.
'''

from data_load import getFeatures, missingValues, categorical_features
import pandas as pd

X = getFeatures()[categorical_features]
missing_values = missingValues(X)

X = pd.get_dummies(X)