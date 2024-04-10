'''
This script compares the performance of the baseline model,
the multinomial-regression model, and the method-2 model (ANN).

The comparison is made using the same dataset and a two-level cross-validation strategy.
This ensures a fair comparison between the three models, when evaluated on the outer fold.
'''

import time
import cla_baseline
import cla_mulreg
from data_fetch import automobile_id, getFeatures, missingValues, categorical_features, numerical_features
import data_transformation
import data_encoding
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from dtuimldmtools import train_neural_net, mcnemar

# Start the timer
start_time = time.time()

# --------------------------------------------------------------

# Gather the data to be used
new_categorical_features = categorical_features.copy()
new_categorical_features.remove('make') # Remove the target column from the features to seperate y from X
X_cat = data_encoding.encode(getFeatures(automobile_id)[new_categorical_features])
X_num = data_transformation.transform(getFeatures(automobile_id)[numerical_features])

X = pd.concat([X_num, X_cat], axis=1)
y = getFeatures(automobile_id)['make']

missing_values = missingValues(X)

# Drop missing values
X = X.drop(missing_values)
attributeNames = X.columns
y = y.drop(missing_values)

# Initialize LabelEncoder to encode 'make'
le = LabelEncoder()

# Fit and transform the target column
y = le.fit_transform(y) # This converts y to an ArrayLike type
y = np.array(y) # Convert to a proper NDArray type

# Convert X to numpy array for use with torch
X = X.to_numpy()

# Convert booleans in X to floats, must be done to use with torch
X = X.astype(float)

# Split dataset
K = 5
outer_cv = KFold(n_splits=K, shuffle=True)

# Define the strength values to be tested
strengths = np.power(10.0, range(-18, 0))

N, M = X.shape
C = np.max(y) + 1 # Number of classes

n_hidden_units = np.arange(112, 117, 1)
loss_fn = torch.nn.CrossEntropyLoss()
max_iter = 10000
n_replicates = 2

baseline_error_rates = []

mulreg_error_rates = []
strengths_best = []

ann_error_rates = []
n_hidden_units_best = []

yhat = []
y_true = []
w_rlr = np.zeros((M, K))
for i, (outer_train_index, outer_test_index) in enumerate(outer_cv.split(X, y)):
    print(f'Outer Fold {i+1}/{K}')
    
    outer_X_train, outer_X_test = X[outer_train_index, :], X[outer_test_index, :]
    outer_y_train, outer_y_test = y[outer_train_index], y[outer_test_index]

    dy = []
    
    # Create inner folds
    inner_cv = KFold(n_splits=K, shuffle=True)

    print('Training ANN model...')
    inner_ann_error_rate_best = np.inf
    for n_hidden_unit in n_hidden_units:
        print(f'Hidden Units: {n_hidden_unit}')
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_unit),  # M features to n hidden units
            torch.nn.ReLU(),  # 1st transfer function,
            torch.nn.Linear(n_hidden_unit, C),  # n hidden units to C output neuron, C = categories
            torch.nn.Softmax(dim=1),  # final tranfer function
        )

        inner_ann_error_rate = []
        for k, (inner_train_index, inner_test_index) in enumerate(inner_cv.split(outer_X_train)):
            '''
            The inner loop is used for hyperparameter tuning the ANN model
            '''

            print(f'Inner Fold {k+1}/{K}')
            # Extract training and test set for current CV fold, convert to tensors
            X_train = outer_X_train[inner_train_index, :]
            X_test = outer_X_train[inner_test_index, :]

            y_train = outer_y_train[inner_train_index]

            y_test = outer_y_train[inner_test_index]

            # Train the net on training data
            net, _, _ = train_neural_net(
                model,
                loss_fn,
                X=torch.tensor(X_train, dtype=torch.float),
                y=torch.tensor(y_train, dtype=torch.long),
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            # Determine probability of each class using trained network
            softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()

            # Determine errors
            e = np.zeros_like(y_test)
            # Determine errors
            for i in range(len(y_test_est)):
                if y_test_est[i] != y_test[i]:
                    e[i] = 1

            # The error rate for the ANN model is stored.
            inner_ann_error_rate.append(np.mean(e).round(2))
        inner_ann_error_rate_average = np.mean(inner_ann_error_rate)
        if inner_ann_error_rate_average < inner_ann_error_rate_best:
            inner_ann_error_rate_best = inner_ann_error_rate_average
            n_hidden_unit_best = n_hidden_unit
    
    print('Training multinomial-regression model...')
    inner_mulreg_error_rate_best = np.inf
    strength_best = 0
    for strength in strengths:
        print(f'Strength: {strength}')
        inner_mulreg_error_rates = []
        for k, (inner_train_index, inner_test_index) in enumerate(inner_cv.split(outer_X_train)):
            '''
            The inner loop is used for hyperparameter tuning the multi-regression model
            '''

            # print(f'Inner Fold {k+1}/{K}')
            # Inner fold sets are derived from the outer fold
            inner_X_train, inner_X_test = outer_X_train[inner_train_index, :], outer_X_train[inner_test_index, :]
            inner_y_train, inner_y_test = outer_y_train[inner_train_index], outer_y_train[inner_test_index]
            
            # The cla_mulreg.fit() function creates our multinomial regression model,
            # fits it to the training data, and returns the model.
            model_mulreg = cla_mulreg.fit(inner_X_train, inner_y_train, regularization=strength, max_iter=10000)

            # The model is then used to predict the classes of the test data.
            yhat_mulreg = cla_mulreg.predict(model_mulreg, inner_X_test)
            yhat_mulreg = yhat_mulreg.reshape(-1, 1) # This ensures that the shape of yhat_mulreg is the same as inner_y_test (n, 1)

            # The error rate for the multi-regression model is calculated.
            inner_mulreg_error_rates.append(np.mean(yhat_mulreg != inner_y_test).round(2))

        inner_mulreg_error_rate_average = np.mean(inner_mulreg_error_rates)
        if inner_mulreg_error_rate_average < inner_mulreg_error_rate_best:
            inner_mulreg_error_rate_best = inner_mulreg_error_rate_average
            strength_best = strength

    print('Evaluating models...')
    # Convert y_test and y_train to 1d arrays
    outer_y_test = np.reshape(outer_y_test, outer_y_test.size)
    outer_y_train = np.reshape(outer_y_train, outer_y_train.size)

    # The baseline model is only tested on the outer test data
    # as the baseline model does not require hyperparameter tuning
    yhat_baseline = cla_baseline.predict(outer_y_test, outer_y_train)
    dy.append(yhat_baseline)
    baseline_error_rate = np.mean(yhat_baseline != outer_y_test).round(2)
    baseline_error_rates.append(baseline_error_rate)

    # The best strength is used to create a new model
    # that is trained on the outer training data, and tested on the outer test data
    model_mulreg = cla_mulreg.fit(outer_X_train, outer_y_train, regularization=strength_best, max_iter=10000)
    yhat_mulreg = cla_mulreg.predict(model_mulreg, outer_X_test)
    dy.append(yhat_mulreg)
    yhat_mulreg = yhat_mulreg.reshape(-1, 1) # This ensures that the shape of yhat_mulreg is the same as outer_y_test (n, 1)
    w_rlr[:, i] = cla_mulreg.estimate_weights(outer_X_train, outer_y_train, strength_best, M)
    
    # The error rate for the multi-regression model is calculated.
    mulreg_error_rate = np.mean(yhat_mulreg != outer_y_test).round(2)
    mulreg_error_rates.append(mulreg_error_rate)
    strengths_best.append(strength_best)
    
    # The best n_hidden_unit is used to create a new model
    # that is trained on the outer training data, and tested on the outer test data
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_unit_best),  # M features to n hidden units
        torch.nn.ReLU(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_unit_best, C),  # n hidden units to C output neuron, C = categories
        torch.nn.Softmax(dim=1),  # final tranfer function
    )

    # Train the net on training data
    net, _, _ = train_neural_net(
        model,
        loss_fn,
        X=torch.tensor(outer_X_train, dtype=torch.float),
        y=torch.tensor(outer_y_train, dtype=torch.long),
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(outer_X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    yhat_ann = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    dy.append(yhat_ann)

    # Determine errors
    e = np.zeros_like(outer_y_test)
    for i in range(len(yhat_ann)):
        if yhat_ann[i] != outer_y_test[i]:
            e[i] = 1
    ann_error_rate = np.mean(e).round(2)
    ann_error_rates.append(ann_error_rate)
    n_hidden_units_best.append(n_hidden_unit_best)

    print(f'yhat_baseline.shape: {yhat_baseline.shape}')
    print(f'yhat_mulreg.shape: {yhat_mulreg.shape}')
    print(f'yhat_ann.shape: {yhat_ann.shape}')
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(outer_y_test)

# Create table with outer fold error rates for each model
error_rates = pd.DataFrame({
    'Outer Fold': range(1, K+1),
    'Baseline E.R.': baseline_error_rates,
    'Multi. Reg. E.R.': mulreg_error_rates,
    'Multi. Reg. Strength': strengths_best,
    'ANN E.R.': ann_error_rates,
    'ANN Hidden Units': n_hidden_units_best
})

# Convert strengths_best to scientific notation
error_rates['Multi. Reg. Strength'] = error_rates['Multi. Reg. Strength'].apply(lambda x: "{:.2e}".format(x))
print(error_rates.to_string(index=False))
error_rates.to_csv('exports/cla_comparison.csv', index=False)

print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))

# Statistically evaluate the three models pairwise using McNemar's test
yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)

alpha = 0.05

[thetahat, CI, p] = mcnemar(y_true, yhat[:, 0], yhat[:, 1], alpha=alpha)
print("theta = theta_Base - theta_Mulreg point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat, CI, p] = mcnemar(y_true, yhat[:, 0], yhat[:, 2], alpha=alpha)
print("theta = theta_Base - theta_ANN point estimate", thetahat, " CI: ", CI, "p-value", p)

[thetahat, CI, p] = mcnemar(y_true, yhat[:, 1], yhat[:, 2], alpha=alpha)
print("theta = theta_Mulreg - theta_ANN point estimate", thetahat, " CI: ", CI, "p-value", p)

# --------------------------------------------------------------

# Stop the timer
end_time = time.time()
# Calculate the runtime
runtime = end_time - start_time
print(f"\nRuntime for the program was {runtime} seconds.")