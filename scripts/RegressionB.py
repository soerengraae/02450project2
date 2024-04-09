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




#Create a table that shows the optimal value for h and lambda for each fold as well as the estimated generalization errors also baseline test error
# Create a table to store the results
results = np.zeros((5, 5))

# Perform 2-level cross-validation
kf_outer = KFold(n_splits=5)
kf_inner = KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kf_outer.split(X_cat)):
    X_train, X_test = X_cat.iloc[train_index], X_cat.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for j, (train_index_inner, test_index_inner) in enumerate(kf_inner.split(X_train)):
        X_train_inner, X_test_inner = X_train.iloc[train_index_inner], X_train.iloc[test_index_inner]
        y_train_inner, y_test_inner = y.iloc[train_index_inner], y.iloc[test_index_inner]

        # Impute missing values in the training data
        imputer = SimpleImputer(strategy='mean')
        X_train_inner_imputed = imputer.fit_transform(X_train_inner)

        # Fit the linear regression model
        y_train_inner_mean = np.mean(y_train_inner)
        y_test_inner_pred = np.full_like(y_test_inner, y_train_inner_mean)

        # Calculate the mean squared error
        mse_inner = mean_squared_error(y_test_inner, y_test_inner_pred)

        imputer.fit(X_train_inner)

        # Transform the training and test data
        X_train_inner_imputed = imputer.transform(X_train_inner)
        X_test_inner_imputed = imputer.transform(X_test_inner)

        for h in h_values:
            for lambda_val in lambda_values:
                model = MLPRegressor(hidden_layer_sizes=(h,), alpha=lambda_val, max_iter=10000)
                model.fit(X_train_inner_imputed, y_train_inner.values.ravel())

                y_test_inner_pred = model.predict(X_test_inner_imputed)
                mse_inner = mean_squared_error(y_test_inner, y_test_inner_pred)

                results[i, j] = mse_inner

# Find the optimal value for h and λ for each fold
optimal_h_values = []
optimal_lambda_values = []
generalization_errors = []


for i in range(5):
    min_mse = np.min(results[i])
    min_mse_index = np.where(results[i] == min_mse)
    optimal_h_values.append(h_values[min_mse_index[0][0] // 5])
    optimal_lambda_values.append(lambda_values[min_mse_index[0][0] % 5])
    generalization_errors.append(min_mse)

# Calculate the baseline test error
baseline_test_error = y_test_pred

# Print the results
print("Fold\tOptimal h\tOptimal λ\tGeneralization Error")
for i in range(5):
    print(f"{i+1}\t{optimal_h_values[i]}\t{optimal_lambda_values[i]}\t{generalization_errors[i]}")
print(f"Baseline test error: {baseline_test_error}")





    