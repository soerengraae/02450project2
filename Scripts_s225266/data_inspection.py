#s225266

from data_load import X as org_X
import matplotlib.pyplot as plt
from matplotlib import rc

# Make a list of all the numeric features in the dataset
numerical_features = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
categorical_features = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

# Create a new dataframe with only the numerical features
X = org_X[numerical_features]

# In case of missing values in any of the numerical features, we will drop them
X = X.dropna()

# Used to see whether the value is in the real or natural number set, for the report
for feature in numerical_features:
  print(f"Data type of {feature}: {X[feature].dtype}")

# Set the font to be 'Helvetica'
rc('font', family='Helvetica')

fig, axes = plt.subplots(len(numerical_features)//3, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, feature in zip(axes, numerical_features):
  ax.hist(X[feature], bins=40, color='black', rwidth=0.9)
  ax.set_title(f"Histogram for {feature}\n$\hat{{\mu}}$: {X[feature].mean():.10f}, "\
         f"$\hat{{\sigma}}$: {X[feature].std():.10f}")

plt.tight_layout()
plt.savefig('/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/histograms.png', dpi=1000)
plt.close()
