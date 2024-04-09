#s225266

from data_inspection import X as org_X
from data_inspection import numerical_features
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
X = org_X[numerical_features]

# Logistic transformation of the data
X = np.log(X)

# Standardize the data
X = (X - X.mean()) / X.std()

# Set the font to be 'Helvetica'
rc('font', family='Helvetica')

# Plot the histograms with lines and annotations for the standard deviations
fig, axs = plt.subplots(len(numerical_features)//3, 3, figsize=(15, 8))
stds = [1, 2, 3]
colors = ['r', 'g', 'b']  # Colors for the lines for the figure below

for i, feature in enumerate(numerical_features):
    # Calculate the percentages
    percentages = [np.mean((X[feature] < std) & (X[feature] > -std)) for std in stds]
    
    # Plot the histogram
    ax = axs[i // 3, i % 3]
    ax.hist(X[feature], bins=40, color='black', rwidth=0.9)
    
    # Add lines and annotations for the standard deviations
    for std, percentage, color in zip(stds, percentages, colors):
        ax.axvline(X[feature].mean() + std * X[feature].std(), color=color, linestyle='dashed', linewidth=1)
        ax.axvline(X[feature].mean() - std * X[feature].std(), color=color, linestyle='dashed', linewidth=1)
        ax.text(X[feature].mean() + std * X[feature].std(), ax.get_ylim()[1] * 0.9, f'{percentage * 100:.2f}%', color=color)
    
    ax.set_title(f"Histogram for {feature}\n$\hat{{\mu}}$: {X[feature].mean():.10f}, "\
                                         f"$\hat{{\sigma}}$: {X[feature].std():.10f}")

plt.tight_layout()
plt.savefig(f'/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/histograms_std_wlines.png', dpi=1000)
plt.close()