# Logistic Regression Implementation

# GOAL: predict whether it will rain tomorrow (1) or stay dry (0)

# IMPLEMENTATION:
#   1. FIT:     learn one weight per feature and one bias from the training days
#               (start at 0, then repeat: guess P(rain), measure the error, nudge the weights)
#   2. SCORE:   for a new day, z = (weights x features) + bias
#   3. PREDICT: sigmoid(z) gives P(rain)
#               If P(rain) >= threshold, predict rain.

# Input format (from data/preprocess.py -> preprocess_location):
#     X = list of lists, one row per example, already standardized
#     y = list of 0/1 labels (1 = it rained the next day)

# Output format (what metrics.py expects):
#     predict_proba -> list of P(rain), one number per example
#     predict       -> list of 0/1 predictions

# References: [1] IBM, "What is logistic regression?"
#             [2] GeeksforGeeks, "Logistic regression in machine learning"


import numpy as np  # arrays and basic math


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=500):
        # how big each nudge is, and how many times we repeat the nudging
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # to be filled in by fit()
        self.weights_ = None  # one weight per feature (n_features numbers)
        self.bias_ = None     # one extra number added to every score

    def _sigmoid(self, z):
        # squeeze any number into the range 0 to 1 (S-shaped curve)
        z = np.clip(z, -500, 500)         # keeps exp() from overflowing
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        # Learn weights and bias (using the TRAINING data)
        # convert the Python lists from preprocess.py into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # safety check: every example needs a label
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n_days, n_features = X.shape      # rows = examples, columns = features

        # start with no opinion: all weights 0, so P(rain) = 0.5 for every day
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            # current guess of P(rain) for ALL training days at once
            p = self._sigmoid(X @ self.weights_ + self.bias_)

            # error: how far each guess is from the real label
            error = y - p

            # gradient of the log-likelihood: average of (error x feature value)
            # nudge the weights and bias in the direction that makes the labels more likely
            self.weights_ += self.learning_rate * (X.T @ error) / n_days
            self.bias_ += self.learning_rate * error.sum() / n_days

        return self                       # write LogisticRegression().fit(X, y)

    def predict_proba(self, X):
        # Return a list with P(rain) for each example (for metrics.py)
        X = np.asarray(X, dtype=float)
        p_rain = self._sigmoid(X @ self.weights_ + self.bias_)
        return p_rain.tolist()                    # plain python list

    def predict(self, X, threshold=0.5):
        # Return a list of 0/1: 1 (rain) if P(rain) >= threshold, else 0 (dry)
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]