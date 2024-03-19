#s225266

from data_load import X as org_X
import matplotlib.pyplot as plt

# Find the features with missing values
missing = org_X.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(ascending=False)

# Plot the missing values
fig, ax = plt.subplots(figsize=(10, 10))
ax.bar(missing.index, missing.values, color='black', width=0.1)
ax.set_title('Missing Values')
ax.set_ylabel('Number of missing values')
ax.set_xlabel('Features')
ax.set_xticklabels(missing.index, rotation=45)
plt.tight_layout()
plt.savefig(f'/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/missing_values.png', dpi=700)