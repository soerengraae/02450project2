#s225266

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset from ucimlrepo
automobile = fetch_ucirepo(id=10)

# Store the features and target in separate variables
X = automobile.data.features
y = automobile.data.targets