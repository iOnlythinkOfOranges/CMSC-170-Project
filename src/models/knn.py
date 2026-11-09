from itertools import islice

import numpy as np


_DISTANCE_BATCH_SIZE = 64


def _nearest_indices(distances, labels, k):
    if k == len(distances):
        return np.lexsort((labels, distances))

    selected = np.argpartition(distances, k - 1)[:k]
    cutoff = distances[selected[-1]]
    tied = np.flatnonzero(distances == cutoff)
    if len(tied) > 1:
        closer = selected[distances[selected] < cutoff]
        slots = k - len(closer)
        tied_by_label = tied[np.argsort(labels[tied], kind="stable")]
        selected = np.concatenate((closer, tied_by_label[:slots]))

    # Order the selected neighbors for voting, with the label as the tie-breaker.
    order = np.lexsort((labels[selected], distances[selected]))
    return selected[order]


def _neighbors_by_row(train_features, train_labels, X_eval, k, p_value):
    if p_value != 2:
        for values in X_eval:
            differences = np.abs(train_features - values)
            distances = np.sum(differences ** p_value, axis=1) ** (1 / p_value)
            yield _nearest_indices(distances, train_labels, k)
        return

    train_norms = np.sum(train_features * train_features, axis=1)
    max_train_norm = train_norms.max()
    roundoff_scale = 8 * train_features.shape[1] * np.finfo(float).eps
    queries = iter(X_eval)
    while rows := list(islice(queries, _DISTANCE_BATCH_SIZE)):
        batch = np.asarray(rows, dtype=float)
        query_norms = np.sum(batch * batch, axis=1)

        # Squared Euclidean distance = ||query||² + ||train||² - 2 * query · train.
        distances = batch @ train_features.T
        distances *= -2
        distances += train_norms
        distances += query_norms[:, None]
        cutoffs = np.partition(distances, k - 1, axis=1)[:, k - 1]
        margins = roundoff_scale * (max_train_norm + query_norms)

        for values, row_distances, cutoff, margin in zip(batch, distances, cutoffs, margins):
            # Recompute the shortlist directly so dot-product roundoff cannot
            # exclude a near tie at the cutoff or decide the order of a tie.
            limit = cutoff + margin
            if np.isfinite(limit):
                candidates = np.flatnonzero(row_distances <= limit)
            else:
                candidates = np.arange(len(train_features))
            differences = train_features[candidates] - values
            precise_distances = np.sum(differences * differences, axis=1)
            yield candidates[_nearest_indices(precise_distances, train_labels[candidates], k)]


def predict_knn(X_train, y_train, X_eval, k_value, p_value):
    """Return one rain score and one voted label per evaluation row."""

    train_size = min(len(X_train), len(y_train))

     # Nothing to learn from, or a useless k so we return "no rain" for every test row.
    if train_size == 0 or k_value <= 0:
        rain_scores = [0.0 for _ in X_eval]
        return rain_scores, [0] * len(rain_scores)

    # Convert to numpy arrays for fast math.
    train_features = np.asarray(X_train[:train_size], dtype=float)
    train_labels = np.asarray(y_train[:train_size])
    rain_scores = []
    predictions = []

    k = min(k_value, train_size)
    for neighbors in _neighbors_by_row(train_features, train_labels, X_eval, k, p_value):

        label_count = {}
        for index in neighbors:
            label = y_train[index]
            label_count[label] = label_count.get(label, 0) + 1

        rain_score = label_count.get(1, 0) / len(neighbors)
        rain_scores.append(rain_score)

        predicted_label = 0
        highest_count = 0
        for label, count in label_count.items():
            if count > highest_count:
                highest_count = count
                predicted_label = label
        predictions.append(predicted_label)

    return rain_scores, predictions


def KNN(preprocessed_data, k_value, p_value):
    """Return test-set rain scores grouped by location."""
    rain_scores = {}
    for location, data in preprocessed_data.items():
        scores, _ = predict_knn(
            data["X_train"], data["y_train"], data["X_test"], k_value, p_value
        )
        rain_scores[location] = scores
    return rain_scores
