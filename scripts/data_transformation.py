'''
The purpose of this script is to standardize the numerical features in the dataset,
as well as performing a logistic scaling on the data.
'''

import numpy as np

def transform(X):
    '''
    Returns: The standardized and logistically scaled data.
    '''
    # Logistic transformation of the data
    X = np.log(X)

    # Standardize the data
    X = (X - X.mean()) / X.std()

    return X