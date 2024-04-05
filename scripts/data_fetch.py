'''
The purpose of this script is to load the dataset.
Import X and y from this script to get the features and target of the dataset.
Lists of the categorical and numerical features are also provided.
'''
categorical_features = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
numerical_features = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

automobile_id = 10

def missingValues(df):
    '''
    Returns: A list of the indices of rows with missing values.
    '''
    
    missing_values = df.index[df.isnull().any(axis=1)]
    return missing_values

from ucimlrepo import fetch_ucirepo
def getDataset(id: int):
    '''
    Returns: The given dataset.
    '''
    return fetch_ucirepo(id=id).data

def getTargets(id: int):
    '''
    Returns: The targets of the given dataset.
    '''
    return getDataset(id=id).targets

def getFeatures(id: int):
    '''
    Returns: The features of the given dataset.
    '''
    return getDataset(id=id).features