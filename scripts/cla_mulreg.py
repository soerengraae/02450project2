'''
A regularization parameter is used to avoid overfitting.
'''

import numpy as np
from sklearn.linear_model import LogisticRegression

def estimate_weights(X_train, y_train, opt_lambda, M):
    '''
    Functions that estimates the weights for the optimal value of lambda.
    '''

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    return np.linalg.solve(XtX + lambdaI, Xty).squeeze()

def fit(X_train, y_train, max_iter=1000, regularization=1):
    
    model = LogisticRegression(multi_class='multinomial', random_state=0, max_iter=max_iter, C=regularization)

    # Train the multimonial regression model
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    # Predict the class for all the samples in the test dataset
    return model.predict(X_test)