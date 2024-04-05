'''
A regularization parameter is used to avoid overfitting.
'''

from sklearn.linear_model import LogisticRegression

def fit(X_train, y_train, max_iter=1000, regularization=1):
    model = LogisticRegression(multi_class='multinomial', random_state=0, max_iter=max_iter, C=regularization)

    # Train the multimonial regression model
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    # Predict the class for all the samples in the test dataset
    return model.predict(X_test)