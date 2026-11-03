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