#s225266

from data_standardization import X as X_numerical
from data_encoding import X as X_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
from sklearn.decomposition import PCA

X = pd.concat([X_numerical, X_categorical], axis=1)
X = X.dropna()

pca = PCA()
pca.fit(X) # X has already been standardized and encoded
X_pca = pca.transform(X)

# Set the font to be 'Helvetica'
rc('font', family='Helvetica')

# Calculate the explained variance
explained_variance = np.round(pca.explained_variance_ratio_*100, decimals=1)

# Create a scree plot
plt.figure(figsize=(12, 6))

# Plot individual explained variance
plt.bar(range(len(explained_variance)), explained_variance, color='black', width=0.9, align='center', label='Individual explained variance')

# Plot cumulative explained variance
plt.plot(np.cumsum(explained_variance), label='Cumulative explained variance', color='black')

# Find the index where cumulative explained variance reaches 80%
index = np.argmax(np.cumsum(explained_variance) >= 80)

# Add dashed vertical line at x=index
plt.axhline(y=80, color='green', linestyle='--')

# Annotate the PC where cumulative explained variance reaches 80%
plt.annotate(f'PC {index + 1}', xy=(index, np.cumsum(explained_variance)[index]), xytext=(index-2, np.cumsum(explained_variance)[index]+5),
  arrowprops=dict(arrowstyle='->', color='green'), color='green')

# Find the index where explained variance drops below 3
index = np.argmax(explained_variance < 3)

# Annotate the PC where explained variance drops below 3

plt.annotate(f'PC {index + 1}', xy=(index, explained_variance[index]), xytext=(index, explained_variance[index]+7),
  arrowprops=dict(arrowstyle='->', color='red'), color='red')

# Add dashed horizontal line at y=4
plt.axhline(y=3, color='red', linestyle='--')

plt.ylabel('Explained variance ratio [%]')
plt.xlabel('Principal components')
plt.title(f'Explained variance ratio for each principal component\n{len(explained_variance)} principal components')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/pca_visual.png', dpi=1000)

# Plot the first three components in a 2D scatter plot
plt.figure(figsize=(10, 8))

# Plot the data projected onto the first three components
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='black')

# Set labels and title
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('First two principal components')

plt.tight_layout()
plt.savefig(f'/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/pca_2d.png', dpi=700)
plt.close()

# Define a colormap that goes from yellow to green
cmap_yellow_to_green = LinearSegmentedColormap.from_list("yellow_to_green", ["yellow", "green"])

# Define a colormap that goes from yellow to red
cmap_yellow_to_red = LinearSegmentedColormap.from_list("yellow_to_red", ["yellow", "red"])

fig, axs = plt.subplots(4, 2, figsize=(8, 12))

for k in range(0, 7):
  row = k // 2
  col = k % 2

  for i in range(len(pca.components_[k])):
    gradient = np.linspace(0, 1, 256)  # Gradient from 0 to 1
    gradient = np.vstack((gradient, gradient)).T

    if pca.components_[k][i] < 0:
      cmap = cmap_yellow_to_red
    else:
      cmap = cmap_yellow_to_green

    # Create a color array using the colormap
    colors = cmap(pca.components_[k][i])

    axs[row, col].imshow(gradient, aspect='auto', cmap=cmap,
              origin='lower', extent=[i-0.4, i+0.4, 0, pca.components_[k][i]])

  # Create the bar plot with transparent bars to create a mask
  axs[row, col].bar(range(len(pca.components_[k])), pca.components_[k], color='none', edgecolor='black', linewidth=0.1)
  axs[row, col].set_xlabel('Feature Index')
  axs[row, col].set_ylabel('Magnitude')
  axs[row, col].set_title(f'{k+1}. Principal Component')

# Remove the empty plot at the 8th position
axs[3, 1].remove()

plt.tight_layout()
plt.savefig(f'/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/pc_plots.png', dpi=700)
plt.close()
