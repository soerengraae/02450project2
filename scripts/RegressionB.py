from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from data_fetch import automobile_id, getTargets, getFeatures, categorical_features
import data_encoding
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_rel

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
    linear_model = Ridge()
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

# Now, perform the paired t-test for MLP vs. Linear Regression
# Extract the MSEs for MLP and Linear Regression from your results
mlp_mse = [result[0] for result in results]  # MLP MSEs
linear_mse = [result[2] for result in results]  # Linear Regression MSEs

# Perform the paired t-test between MLP and Linear Regression
t_stat, p_value = ttest_rel(mlp_mse, linear_mse)

# Output the results
print(f"t-statistic: {t_stat}, p-value: {p_value}")

mlp_mse = [result[0] for result in results]  # MLP MSEs
baseline_mse = [result[1] for result in results]  # Baseline MSEs

t_stat, p_value = ttest_rel(mlp_mse, baseline_mse)

print(f"t-statistic: {t_stat}, p-value: {p_value}")

linear_mse = [result[2] for result in results]  # Linear Regression MSEs
baseline_mse = [result[1] for result in results]  # Baseline MSEs

t_stat, p_value = ttest_rel(linear_mse, baseline_mse)

print(f"t-statistic: {t_stat}, p-value: {p_value}")
