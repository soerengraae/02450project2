'''
This script compares the performance of the baseline model,
the multi-regression model, and the method-2 model.

The comparison is made using the same dataset and a two-level cross-validation strategy.
This ensures a fair comparison between the two models.
'''

import cla_baseline
import cla_mulreg
from data_fetch import automobile_id, getTargets, getFeatures, missingValues
import data_transformation
import data_encoding
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

# Gather the data to be used
X_cat = data_encoding.encode(getFeatures(automobile_id)['make'])
missing_values_cat = missingValues(X_cat)

X_num = data_transformation.transform(getFeatures(automobile_id)[['engine-size', 'horsepower', 'price']])
missing_values_num = missingValues(X_num)

missing_values = list(set(missing_values_num).union(set(missing_values_cat)))

X = pd.concat([X_num, X_cat], axis=1)
y = getTargets(automobile_id)

# Drop missing values
X = X.drop(missing_values)
y = y.drop(missing_values)

# Reset the index so it goes from 0 to n-1
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split dataset
K = 10
outer_cv = KFold(n_splits=K)

# Define the strength values to be tested in range(0.02, 5) with a step of 0.22
strengths = np.arange(0.01, 0.5, 0.01)

baseline_error_rates = []
mulreg_error_rates = []
strengths_best = []

i = 0
for outer_train_index, outer_test_index in outer_cv.split(X):
    i += 1
    print(f'Outer Fold {i}/{K}')
    
    outer_X_train, outer_X_test = X.iloc[outer_train_index], X.iloc[outer_test_index]
    outer_y_train, outer_y_test = y.iloc[outer_train_index], y.iloc[outer_test_index]

    # Create inner folds
    inner_cv = KFold(n_splits=K)

    inner_mulreg_error_rate_best = np.inf
    strength_best = 0
    for strength in strengths:
        # print(f'Testing strength: {strength}')
        inner_mulreg_error_rates = []
        for inner_train_index, inner_test_index in inner_cv.split(outer_X_train):
            '''
            The inner loop is used for hyperparameter tuning
            for the multi-regression model and the method-2 model
            '''

            # Inner fold sets are derived from the outer fold
            inner_X_train, inner_X_test = outer_X_train.iloc[inner_train_index], outer_X_train.iloc[inner_test_index]
            inner_y_train, inner_y_test = outer_y_train.iloc[inner_train_index], outer_y_train.iloc[inner_test_index]

            # Done to avoid DataConversionWarning
            inner_y_train = np.ravel(inner_y_train)
            inner_y_test = np.ravel(inner_y_test)
            
            # The cla_mulreg.fit() function creates our multinomial regression model,
            # fits it to the training data, and returns the model.
            model_mulreg = cla_mulreg.fit(inner_X_train, inner_y_train, regularization=strength)

            # The model is then used to predict the classes of the test data.
            yhat_mulreg = cla_mulreg.predict(model_mulreg, inner_X_test)
            yhat_mulreg = yhat_mulreg.reshape(-1, 1) # This ensures that the shape of yhat_mulreg is the same as inner_y_test (n, 1)

            # The error rate for the multi-regression model is calculated.
            inner_mulreg_error_rate = np.mean(yhat_mulreg != inner_y_test).round(2)
            inner_mulreg_error_rates.append(inner_mulreg_error_rate)

        inner_mulreg_error_rate_average = np.mean(inner_mulreg_error_rates)
        if inner_mulreg_error_rate_average < inner_mulreg_error_rate_best:
            inner_mulreg_error_rate_best = inner_mulreg_error_rate_average
            strength_best = strength

    # The baseline model is only tested on the outer test data
    # as the baseline model does not require hyperparameter tuning
    yhat_baseline = cla_baseline.predict(outer_y_test, outer_y_train)
    baseline_error_rate = np.mean(yhat_baseline != outer_y_test).round(2)
    baseline_error_rates.append(baseline_error_rate)
    
    # Done to avoid DataConversionWarning
    outer_y_train = np.ravel(outer_y_train)
    outer_y_test = np.ravel(outer_y_test)

    # The best strength is used to create a new model
    # that is trained on the outer training data, and tested on the outer test data
    model_mulreg = cla_mulreg.fit(outer_X_train, outer_y_train, regularization=strength_best)
    yhat_mulreg = cla_mulreg.predict(model_mulreg, outer_X_test)
    yhat_mulreg = yhat_mulreg.reshape(-1, 1) # This ensures that the shape of yhat_mulreg is the same as outer_y_test (n, 1)
    
    # The error rate for the multi-regression model is calculated.
    mulreg_error_rate = np.mean(yhat_mulreg != outer_y_test).round(2)
    mulreg_error_rates.append(mulreg_error_rate)
    strengths_best.append(strength_best)

# Create table with outer fold error rates for each fold and model
error_rates = pd.DataFrame({
    'Outer Fold': range(1, K+1),
    'Baseline E.R.': baseline_error_rates,
    'Multi. Reg. E.R.': mulreg_error_rates,
    'Multi. Reg. Strength': strengths_best
})

print(error_rates.to_string(index=False))