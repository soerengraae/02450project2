'''
The baseline clasification model computes the largest class in the dataset,
and predicts that class for all the samples in the test dataset.
'''

from data_load import getTargets
from data_standardization import X as X_num, missing_values as missing_values_num
from data_encoding import X as X_cat, missing_values as missing_values_cat
import pandas as pd
import numpy as np

missing_values = list(set(missing_values_num).union(set(missing_values_cat)))

X = pd.concat([X_num, X_cat], axis=1)
y = getTargets()

X = X.drop(missing_values)
y = y.drop(missing_values)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Compute the largest class in the dataset
largest_class = y.value_counts().idxmax()

# Predict the largest class for all the samples in the test dataset
yhat = [largest_class[0]] * len(y)

errors = np.zeros_like(y, dtype=bool)
for i in range(len(y)):
    errors[i] = y.iloc[i] != yhat[i]

# Calculate error rate
error_rate = errors.mean()
print(f'Error rate: {(error_rate*100).round(2)}%')