import numpy as np


def predict_knn(X_train, y_train, X_eval, k_value, p_value):
    """Return one rain score and one voted label per evaluation row."""
    train_size = min(len(X_train), len(y_train))
    if train_size == 0 or k_value <= 0:
        rain_scores = [0.0 for _ in X_eval]
        return rain_scores, [0] * len(rain_scores)

    train_features = np.asarray(X_train[:train_size], dtype=float)
    train_labels = np.asarray(y_train[:train_size])
    rain_scores = []
    predictions = []

    for values in X_eval:
        differences = np.abs(train_features - values)
        distances = np.sum(differences ** p_value, axis=1) ** (1 / p_value)

        # Order by distance first and by label when distances tie.
        neighbors = np.lexsort((train_labels, distances))[:k_value]

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
