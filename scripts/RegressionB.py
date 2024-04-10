# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor
# from data_fetch import automobile_id, getTargets, getFeatures, categorical_features
# import data_encoding
# import matplotlib.pyplot as plt
# from sklearn.impute import SimpleImputer
# import numpy as np
# from sklearn.linear_model import LinearRegression


# # Load the data
# X_cat = data_encoding.encode(getFeatures(automobile_id)[categorical_features])
# y = getTargets(automobile_id)

# # Create an imputer object that replaces missing values with the mean value of each column
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# kf = KFold(n_splits=5)
# # Define kf object

# for train_index, test_index in kf.split(X_cat):
#     X_train, X_test = X_cat.iloc[train_index], X_cat.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     # Impute missing values in the training data
#     X_train_imputed = imputer.fit_transform(X_train)

#     # Fit the linear regression model
#     y_train_mean = np.mean(y_train)
#     y_test_pred = np.full_like(y_test, y_train_mean)

#     # Calculate the mean squared error
#     mse = mean_squared_error(y_test, y_test_pred)

# imputer.fit(X_train)
# imputer.fit(X_train)
# X_test_imputed = imputer.transform(X_test)
# # Impute missing values in the test data
# X_train_imputed = imputer.transform(X_train)



# # Set the range of values for h (number of hidden units)
# h_values = [1, 5, 10, 20, 50]

# # Set the range of values for λ (regularization parameter)
# lambda_values = [0.001, 0.01, 0.1, 1, 10]


# for h in h_values:
#     for lambda_val in lambda_values:
#         model = MLPRegressor(hidden_layer_sizes=(h,), alpha=lambda_val, max_iter=10000)
#         model.fit(X_train_imputed, y_train.values.ravel())  # Convert y_train to a numpy array before calling ravel()
#         y_test_pred = model.predict(X_test_imputed)
#         mse = mean_squared_error(y_test, y_test_pred)
#         print(f"Number of hidden units: {h}, Regularization parameter: {lambda_val}")
#         print("Mean Squared Error:", mse)





# # Create a table to store the results
# results = np.zeros((5, 7))

# # Perform 2-level cross-validation
# kf_outer = KFold(n_splits=5)
# kf_inner = KFold(n_splits=5)
# results = np.zeros((5, 7))
# for i, (train_index, test_index) in enumerate(kf_outer.split(X_cat)):
#     X_train, X_test = X_cat.iloc[train_index], X_cat.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     for j, (train_index_inner, test_index_inner) in enumerate(kf_inner.split(X_train)):
#         X_train_inner, X_test_inner = X_train.iloc[train_index_inner], X_train.iloc[test_index_inner]
#         y_train_inner, y_test_inner = y.iloc[train_index_inner], y.iloc[test_index_inner]

#         # Impute missing values in the training data
#         imputer = SimpleImputer(strategy='mean')

#         # Fit the linear regression model
#         y_train_inner_mean = np.mean(y_train_inner)
#         y_test_inner_pred = np.full_like(y_test_inner, y_train_inner_mean)

#         # Calculate the mean squared error for linear regression
#         mse_inner_linear = mean_squared_error(y_test_inner, y_test_inner_pred)

#         imputer.fit(X_train_inner)

#         imputer.fit(X_train_inner)
#         X_train_inner_imputed = imputer.transform(X_train_inner)
#         X_test_inner_imputed = imputer.transform(X_test_inner)

#         for h in h_values:
#             for lambda_val in lambda_values:
#                 model = MLPRegressor(hidden_layer_sizes=(h,), alpha=lambda_val, max_iter=10000)
#                 model.fit(X_train_inner_imputed, y_train_inner.values.ravel())

#                 y_test_inner_pred = model.predict(X_test_inner_imputed)
#                 mse_inner_mlp = mean_squared_error(y_test_inner, y_test_inner_pred)

#                 results[i, j] = mse_inner_mlp

#         # Calculate the mean squared error for baseline
#         y_test_inner_pred_baseline = np.full_like(y_test_inner, np.mean(y_train_inner))
#         mse_inner_baseline = mean_squared_error(y_test_inner, y_test_inner_pred_baseline)
#         results[i, 6] = mse_inner_baseline

#         # Calculate the mean squared error for linear regression
#         results[i, 6] = mse_inner_linear

#         results[i, 6] = mse_inner_linear
# optimal_h_values_mlp = []
# optimal_lambda_values_linear = []
# generalization_errors_mlp = []
# generalization_errors_baseline = []
# generalization_errors_linear = []

# for i in range(5):
#     min_mse_mlp = np.min(results[i, :5])
#     min_mse_index_mlp = np.argmin(results[i, :5])
#     optimal_h_values_mlp.append(h_values[min_mse_index_mlp % 5])

#     # For linear regression
#     linear_model = LinearRegression()
#     linear_model.fit(X_train_inner_imputed, y_train_inner)
#     y_test_inner_pred_linear = linear_model.predict(X_test_inner_imputed)
#     mse_inner_linear = mean_squared_error(y_test_inner, y_test_inner_pred_linear)
#     results[i, 6] = mse_inner_linear
#     optimal_lambda_values_linear.append(lambda_values[np.argmin(results[i, 6:])])

# # Print the results
# print("Fold\tOptimal h (MLP)\tOptimal λ (Linear Regression)\tMLP Generalization Error\tBaseline Generalization Error\tLinear Regression Generalization Error")
# for i in range(5):
#     print(f"{i+1}\t{optima'l_h_values_mlp[i]}\t{optimal_lambda_values_linear[i]}\t{generalization_errors_mlp[i]}\t{generalization_errors_baseline[i]}\t{generalization_errors_linear[i]}")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from data_fetch import automobile_id, getTargets, getFeatures, categorical_features
import data_encoding
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer

# Load and encode the data
X_cat = data_encoding.encode(getFeatures(automobile_id)[categorical_features])
y = getTargets(automobile_id)

# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_cat_imputed = imputer.fit_transform(X_cat)

# Feature Scaling
scaler = StandardScaler()
X_cat_scaled = scaler.fit_transform(X_cat_imputed)

# Define KFold for cross-validation
kf = KFold(n_splits=5)

# Initialize results storage
results = []

# Perform cross-validation
for train_index, test_index in kf.split(X_cat_scaled):
    X_train, X_test = X_cat_scaled[train_index], X_cat_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Baseline model: predict the mean of y_train
    y_train_mean = np.mean(y_train)
    y_pred_baseline = np.full(y_test.shape, y_train_mean)
    mse_baseline = mean_squared_error(y_test, y_pred_baseline)

    # Linear Regression Model
    linear_model = Ridge()  # Using Ridge as an example of a regularized linear model
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)

    # MLP Model evaluation
    h_values = [1, 5, 10, 20, 50]
    lambda_values = [0.001, 0.01, 0.1, 1, 10]
    mlp_mses = []
    for h in h_values:
        for lambda_val in lambda_values:
            model = MLPRegressor(hidden_layer_sizes=(h,), alpha=lambda_val, max_iter=10000)
            model.fit(X_train, y_train.values.ravel())
            y_pred_mlp = model.predict(X_test)
            mse_mlp = mean_squared_error(y_test, y_pred_mlp)
            mlp_mses.append(mse_mlp)

    # Store results
    best_mse_mlp = min(mlp_mses)
    results.append((best_mse_mlp, mse_baseline, mse_linear))

# Print the results
print("MLP MSE, Baseline MSE, Linear MSE")
for result in results:
    print(result)
