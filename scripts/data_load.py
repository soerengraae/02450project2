'''
The purpose of this script is to load the dataset.
Import X and y from this script to get the features and target of the dataset.
Lists of the categorical and numerical features are also provided.
'''

categorical_features = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
numerical_features = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']




def missingValues(df):
    '''
    Returns: A list of the indices of rows with missing values.
    '''
    
    missing_values = df.index[df.isnull().any(axis=1)]
    return missing_values

from ucimlrepo import fetch_ucirepo
def getDataset():
    return fetch_ucirepo(id=10).data

# Store the features and target in separate variables
def getTargets():
    return getDataset().targets

def getFeatures():
    return getDataset().features