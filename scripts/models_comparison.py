'''
This script compares the performance of the baseline model,
the multi-regression model, and the method-2 model.

The comparison is made using the same dataset and cross-validation strategy.
This ensures a fair comparison between the two models.
'''

import model_cla_baseline
import model_cla_mulreg
from data_load import getTargets
from data_standardization import X as X_num, missing_values as missing_values_num
from data_encoding import X as X_cat, missing_values as missing_values_cat
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

missing_values = list(set(missing_values_num).union(set(missing_values_cat)))

X = pd.concat([X_num, X_cat], axis=1)
y = getTargets()

X = X.drop(missing_values)
y = y.drop(missing_values)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split dataset
K = 10
CV = KFold(n_splits=K)

baseline_error_rates = []
mulreg_error_rates = []
for train_index, test_index in CV.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    yhat_baseline = model_cla_baseline.predict(y_test, y_train)

    # Done to avoid DataConversionWarning
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    yhat_mulreg = model_cla_mulreg.predict(X_train, X_test, y_train)

    baseline_error_rate = np.mean(yhat_baseline != y_test).round(2)
    mulreg_error_rate = np.mean(yhat_mulreg != y_test).round(2)

    baseline_error_rates.append(baseline_error_rate)
    mulreg_error_rates.append(mulreg_error_rate)

# Create table with error rates for each fold and model
error_rates = pd.DataFrame({
    'Outer Fold': range(1, K+1),
    'Baseline': baseline_error_rates,
    'Multi. Reg.': mulreg_error_rates
})

print(error_rates.to_string(index=False))