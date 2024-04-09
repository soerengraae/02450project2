'''
The purpose of this script is to encode the categorical features in a dataset.
'''

import pandas as pd

def encode(X):
    X = pd.get_dummies(X)
    return X