# import numpy as np

# bs = np.array([[-1.4, 2.6], [-0.6, -1.6], [2.1, 5.0], [0.7, 3.8]])
# ws = np.array([[1.2, -2.1, 3.2], [1.2, -1.7, 2.9], [1.8, -1.1, 2.2]])

# K = 3
# for b in bs:
#     for w in ws:
#         yhat = np.array([1, b[0], b[1]]).dot(w)

import numpy as np

# Weights for each class
w1 = np.array([1.2, -2.1, 3.2])
w2 = np.array([1.2, -1.7, 2.9])
w3 = np.array([1.3, -1.1, 2.2])

# Observations
observations = {
    'A': np.array([-1.4, 2.6]),
    'B': np.array([-0.6, -1.6]),
    'C': np.array([2.1, 5.0]),
    'D': np.array([0.7, 3.8])
}

# Function to compute the scores for each class
def compute_scores(observation, weights):
    yhat = np.array([1, observation[0], observation[1]])
    return yhat.dot(weights)

# Function to calculate the softmax probabilities for the first 3 classes and class 4
def softmax_probabilities(scores):
    exp_scores = np.exp(scores)
    sum_exp_scores = 1 + np.sum(exp_scores)
    probs = exp_scores / sum_exp_scores  # Probabilities for the first 3 classes
    probs = np.append(probs, 1 / sum_exp_scores)  # Probability for the 4th class
    return probs

# Calculate the scores and probabilities for each observation
results = {}
for label, observation in observations.items():
    scores = np.array([
        compute_scores(observation, w1),
        compute_scores(observation, w2),
        compute_scores(observation, w3)
    ])
    probabilities = softmax_probabilities(scores)
    results[label] = probabilities

# Print the results
for label, probabilities in results.items():
    print(f"Observation {label}: {probabilities.round(3)}")

# Observation B has the highest probability for class 4 (0.73).