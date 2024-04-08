'''
The baseline clasification model computes the largest class in the train dataset,
and predicts that class for all the samples in the test dataset.
'''

import numpy as np
def predict(y_test, y_train):
    # Compute the largest class in the training dataset (a numpy array)
    largest_class = np.argmax(np.bincount(y_train))

    # Predict the largest class for all the samples in the test dataset
    yhat = np.full(y_test.shape, largest_class)

    return yhat