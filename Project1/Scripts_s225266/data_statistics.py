#s225266

from data_load import X
from data_inspection import categorical_features, numerical_features
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# Set the font to be 'Helvetica'
rc('font', family='Helvetica')

# Create table of summary statistics and save as figure
def table(summary, title, title_position, file_name, height, width):
  fig, ax = plt.subplots(figsize=(height, width))
  ax.axis('tight')
  ax.axis('off')

  table = ax.table(cellText=summary.values.round(2), 
                   colLabels=summary.columns, 
                   rowLabels=summary.index, 
                   cellLoc='center', 
                   loc='center')
  
  fig.suptitle(title, fontsize=12, y=title_position)
  table.auto_set_font_size(False)
  table.set_fontsize(8)
  table.scale(1, 1.5)
  plt.tight_layout()
  plt.savefig(f'/Users/sorengraae/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Machine Learning/Project 1/Exports/{file_name}', dpi=500)
  plt.close()

# get summary statistics of first half of numerical_features
summary = X[numerical_features[:len(numerical_features)//2]].describe()
 
table(summary, 'Summary statistics for numerics features\n(1/2)', 0.95, 'summary_statistics_12.png', 10, 3)

# get summary statistics of second half of numerical_features
summary = X[numerical_features[len(numerical_features)//2:]].describe()

table(summary, 'Summary statistics for numerics features\n(2/2)', 0.95, 'summary_statistics_22.png', 10, 3)

# Get summary statistics of categorical_features
for feature in categorical_features:
  count = X[feature].value_counts()
  proportion = (count / count.sum())
  mode = X[feature].mode()[0] # Get the most occurring category
  
  # Create a table for the summary statistics
  summary = pd.DataFrame({'Count': count, 'Proportion': proportion})
  summary = summary.transpose()
  summary.columns.name = feature
  
  # Add a "Total" column
  summary['Total'] = summary.sum(axis=1)
  
  table(summary, f'Summary statistics for {feature}\nMost occurring: {mode}', 0.8, f'summary_statistics_{feature}.png', 14, 2)