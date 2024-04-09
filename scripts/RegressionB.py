from sklearn.model_selection import KFold

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

# Create an imputer object that replaces missing values with the mean value of each column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_cat):
    # Split the data into training and test sets
    X_train, X_test = X_cat.iloc[train_index], X_cat.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Impute missing values in the training data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    # Fit the linear regression model
    y_train_mean = np.mean(y_train)
    y_test_pred = np.full_like(y_test, y_train_mean)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_test_pred)

imputer.fit(X_train)

# Transform the training and test data
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)



# Set the range of values for h (number of hidden units)
h_values = [1, 5, 10, 20, 50]

# Set the range of values for λ (regularization parameter)
lambda_values = [0.001, 0.01, 0.1, 1, 10]


for h in h_values:
    for lambda_val in lambda_values:
        model = MLPRegressor(hidden_layer_sizes=(h,), alpha=lambda_val, max_iter=10000)
        model.fit(X_train_imputed, y_train.values.ravel())  # Convert y_train to a numpy array before calling ravel()
        
        y_test_pred = model.predict(X_test_imputed)
        mse = mean_squared_error(y_test, y_test_pred)
        print(f"Number of hidden units: {h}, Regularization parameter: {lambda_val}")
        print("Mean Squared Error:", mse)