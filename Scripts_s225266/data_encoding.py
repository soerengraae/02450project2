#s225266

from data_load import X as org_X
from data_inspection import categorical_features
import pandas as pd

X = org_X[categorical_features]
X = X.dropna()

X = pd.get_dummies(X)