# Gaussian Naive Bayes Implementation

# GOAL: predict whether it will rain tomorrow (1) or stay dry (0)
# Full pseudocode: docs/pseudocode_gnb.md

# IMPLEMENTATION:
#   1. FIT:     learn from the training days (a) how common rain and dry are (the "priors"), 
#               and (b) the mean and variance of every feature for each class (the two "bell curves" per feature)
#   2. SCORE:   for a new day, measure how well it fits the Rain curves and the Dry curves (add logs instead of multiplying)
#   3. PREDICT: turn the two scores into P(rain)
#               If P(rain) >= threshold, predict rain.

# Input format (from data/preprocess.py -> preprocess_location):
#     X = list of lists, one row per example, already standardized
#     y = list of 0/1 labels (1 = it rained the next day)

# Output format (what metrics.py expects):
#     predict_proba -> list of P(rain), one number per example
#     predict       -> list of 0/1 predictions


import numpy as np  # arrays and basic math


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        # tiny number (scaled by the largest variance) added to every variance so we never divide by zero later
        self.var_smoothing = var_smoothing

        # to be filled in by fit(). row 0 = Dry class, row 1 = Rain class.
        self.priors_ = None  # how common each class is (2 numbers)
        self.means_ = None   # mean of each feature, per class (2 x n_features)
        self.vars_ = None    # variance of each feature, per class (2 x n_features)

    def fit(self, X, y):
        # Learn priors, means, and variances (using the TRAINING data)
        # convert the Python lists from preprocess.py into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # safety check: every example needs a label
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n_days, n_features = X.shape      # rows = examples, columns = features

        # empty arrays that we fill in below (index 0 = Dry, index 1 = Rain)
        self.priors_ = np.zeros(2)
        self.means_ = np.zeros((2, n_features))
        self.vars_ = np.zeros((2, n_features))

        for c in (0, 1):                  # loop once for Dry, once for Rain
            X_c = X[y == c]                 # keep only the rows that belong to class c
            n_c = X_c.shape[0]              # how many examples are in this class

            if n_c == 0:                    # a class with no examples cannot be learned
                raise ValueError("Training data has no examples of class %d." % c)

            # PRIOR: the share of training days in this class (lab: P(Spam), P(Ham))
            self.priors_[c] = n_c / n_days

            # MEAN of every feature: add up the values, divide by the count
            self.means_[c] = X_c.sum(axis=0) / n_c

            # VARIANCE of every feature: average squared distance from the mean
            self.vars_[c] = ((X_c - self.means_[c]) ** 2).sum(axis=0) / n_c

        # SMOOTHING: add a tiny number so no variance is exactly 0
        largest_var = self.vars_.max()
        if largest_var == 0:              # edge case: every feature was constant
            largest_var = 1.0
        self.vars_ = self.vars_ + self.var_smoothing * largest_var

        return self                       # write GaussianNaiveBayes().fit(X, y)

    def _log_scores(self, X):
        # score each example for each class (higher score -> better fit)
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], 2))        # one row per example, columns = [Dry, Rain]

        for c in (0, 1):
            # start with how common the class is (log of the prior)
            log_prior = np.log(self.priors_[c])

            # bell-curve formula in log form, for ALL features at once:
            #   close to the class mean -> small penalty (score stays high)
            #   far from the class mean -> big penalty (score drops)
            per_feature = (
                -0.5 * np.log(2 * np.pi * self.vars_[c])
                - ((X - self.means_[c]) ** 2) / (2 * self.vars_[c])
            )

            # "naive" step: treat features as independent and add them up (sum across columns)
            scores[:, c] = log_prior + per_feature.sum(axis=1)

        return scores

    def predict_proba(self, X):
        # Return a list with P(rain) for each example (for metrics.py)
        scores = self._log_scores(X)

        # subtract the larger score in each row so exp() cannot underflow to 0
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)

        # Rain's share of the total = probability of rain
        p_rain = exp_scores[:, 1] / exp_scores.sum(axis=1)
        return p_rain.tolist()                    # plain Python list

    def predict(self, X, threshold=0.5):
        # Return a list of 0/1: 1 (rain) if P(rain) >= threshold, else 0 (dry)
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]