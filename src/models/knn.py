#DECLARATION ON GENERATIVE AI
#During the preparation of this work, the author(s) used claude Sonnet 5.5
#to help optimize the knn to load faster.  
# After using this tool, the author(s) reviewed and edited the content as 
# needed and take(s) full responsibility for the content.

from itertools import islice
import sys

import numpy as np


_DISTANCE_BATCH_SIZE = 64


def _nearest_indices(distances, labels, k):
    #If every neighbor is being asked for, just sort by distance.
    if k == len(distances):
        return np.lexsort((labels, distances))

    #Pick the k smallest distances WITHOUT fully sorting
    selected = np.argpartition(distances, k - 1)[:k]
    cutoff = distances[selected[-1]]    #    # The last slot holds the k-th smallest, i.e. the farthest of the chosen rows.
    tied = np.flatnonzero(distances == cutoff)

    #Decides what to pick if there's a tie
    if len(tied) > 1:
        closer = selected[distances[selected] < cutoff]
        slots = k - len(closer)
        tied_by_label = tied[np.argsort(labels[tied], kind="stable")]
        selected = np.concatenate((closer, tied_by_label[:slots]))

    # Order the selected neighbors for voting, with the label as the tie-breaker.
    order = np.lexsort((labels[selected], distances[selected]))
    return selected[order]


def _neighbors_by_row(train_features, train_labels, X_eval, k, p_value):
    #Will hold all the neighbors of X_eval
    all_neighbors = []

    #Minkowski formula
    if p_value != 2:
        #Per iteration, handles ALL training rows at once.
        for values in X_eval:
            differences = np.abs(train_features - values)
            distances = np.sum(differences ** p_value, axis=1) ** (1 / p_value)
            all_neighbors.append(_nearest_indices(distances, train_labels, k))
        return all_neighbors


    #Note: This is done so it will caluclate by changing (xi -vi)^2 into xi^2 -2xivi + vi^2

    train_norms = np.sum(train_features * train_features, axis=1)   #a list containing the sum of the xi^2 values for each training values(Σxi^2)
    max_train_norm = train_norms.max()

    # Calculates a tolerance factor to account for floating-point rounding errors
    roundoff_scale = 8 * train_features.shape[1] * np.finfo(float).eps

    # Turn the test rows into an iterator so islice can take them in chunks.
    queries = iter(X_eval)


    
    while True:
    # Get up to 64 evaluation observations.
        rows = list(islice(queries, _DISTANCE_BATCH_SIZE))

        # Stop if there are no more observations.
        if not rows:
            break

        # Convert the retrieved observations into a NumPy array.
        batch = np.asarray(rows, dtype=float)

        # Calculate the sum of squared features for each evaluation observation.
        query_norms = np.sum(batch * batch, axis=1) #a list containing the sum of the vi^2 values for each test values(Σvi^2)

        # Calculate the squared Euclidean distances using the optimized formula.
        distances = np.matmul(batch, train_features.T) #This gets the dot product between every evaluation observation, represents Σ (xi * vi)
        distances *= -2 #Then this makes it into -2·Σ(xi·vi)

        #Now do Σxi^2  -  2·Σ(xi·vi)  +  Σvi^2
        distances += train_norms  
        distances += query_norms[:, None] 

        # Find the Kth smallest squared distance for each evaluation observation.
        cutoffs = np.partition(distances, k - 1, axis=1)[:, k - 1]

        # Calculate the tolerance margin for floating-point rounding errors.
        margins = roundoff_scale * (max_train_norm + query_norms)

        # Process each evaluation observation in the current batch.
        for values, row_distances, cutoff, margin in zip(batch, distances, cutoffs, margins):

            # Add the tolerance margin to the Kth smallest distance.
            limit = cutoff + margin

            # Get the training observations within the calculated limit.
            if np.isfinite(limit):
                candidates = np.flatnonzero(row_distances <= limit)
            else:
                candidates = np.arange(len(train_features))

            # Re-measure the shortlist exactly: subtract, square, and add up the features.
            differences = train_features[candidates] - values
            precise_distances = np.sum(differences * differences, axis=1)

            # Select the K nearest neighbors and return their original training indices.
            all_neighbors.append(candidates[_nearest_indices(precise_distances, train_labels[candidates], k)])
    return all_neighbors


def predict_knn(X_train, y_train, X_eval, k_value, p_value):
    """Return one rain score and one voted label per evaluation row."""
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")

    train_size = len(X_train)

    # Empty training set or non-positive / out-of-range k: default to 0
    if train_size == 0 or k_value <= 0 or k_value > train_size:
        rain_scores = [0.0 for _ in X_eval]
        return rain_scores, [0] * len(rain_scores)

    # Convert to numpy arrays for fast math.
    train_features = np.asarray(X_train, dtype=float)
    train_labels = np.asarray(y_train)
    rain_scores = []
    predictions = []

    #Main KNN algorithm
    for neighbors in _neighbors_by_row(train_features, train_labels, X_eval, k_value, p_value):

        #Find which label appeared the most
        label_count = {}
        for index in neighbors:
            label = y_train[index]

            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1

        #1 means rain, so here we're getting the value of the key
        if 1 in label_count:
            rain_count = label_count[1]
        else:
            rain_count = 0

        rain_score = rain_count / len(neighbors)
        rain_scores.append(rain_score)

        predicted_label = 0
        highest_count = 0
        for label, count in label_count.items():
            if count > highest_count:
                highest_count = count
                predicted_label = label
        predictions.append(predicted_label)

    return rain_scores, predictions


class KNearestNeighbors:
    """Object-oriented interface for K-Nearest Neighbors matching GNB, LR, and CART."""

    def __init__(self, k=15, p=2):
        self.k = k
        self.p = p
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict_proba(self, X):
        scores, _ = predict_knn(self.X_train_, self.y_train_, X, self.k, self.p)
        return scores

    def predict(self, X, threshold=0.5):
        if threshold == 0.5:
            _, preds = predict_knn(self.X_train_, self.y_train_, X, self.k, self.p)
            return preds
        scores = self.predict_proba(X)
        return [1 if s >= threshold else 0 for s in scores]


# Aliases
KNN = KNearestNeighbors
