def KNN(preprocessed_data, k_value, p_value):
    print("K-nearest neighbour output")
    print("k-value: ", k_value, "\t", "p-value: ", p_value)

    counts = {}

    # Process each location
    for location, data in preprocessed_data.items():

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        print(location, "train:", len(X_train), "-> test:", len(X_test))

        counts[location] = [0, 0, 0, 0] #TP TN FP FN

        # KNN algorithm
        for values, actual in zip(X_test, y_test):
            neighbors = []

            for train_values, label in zip(X_train, y_train):

                # Minkowski distance
                distance = 0
                for i in range(len(values)):
                    distance += abs(train_values[i] - values[i]) ** p_value

                distance = distance ** (1 / p_value)

                neighbors.append([distance, label])

                # Sort from nearest to farthest
                neighbors.sort()
                if len(neighbors) > k_value:
                    neighbors.pop()

            #Get the highest
            label_count = {}

            for neighbor in neighbors:
                if neighbor[1] not in label_count:
                    label_count[neighbor[1]] = 1
                else:
                    label_count[neighbor[1]] += 1

            predicted_label = 0
            highest_count = 0

            for label, count in label_count.items():
                if count > highest_count:
                    highest_count = count
                    predicted_label = label

            # Confusion matrix
            if predicted_label == 1 and actual == 1:
                counts[location][0] += 1  # TP
            elif predicted_label == 0 and actual == 0:
                counts[location][1] += 1  # TN
            elif predicted_label == 1 and actual == 0:
                counts[location][2] += 1  # FP
            else:
                counts[location][3] += 1  # FN

    # Metrics (per location, then overall)
    counts["OVERALL"] = [
        sum(c[i] for c in counts.values())
        for i in range(4)
    ]

    for name, (tp, tn, fp, fn) in counts.items():

        total = tp + tn + fp + fn

        #Calculate the metrics
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )


        print("\n" + name)
        print("  Confusion matrix [[TN FP] [FN TP]]:", [[tn, fp], [fn, tp]])
        print("  Accuracy :", round(accuracy, 4))
        print("  Precision:", round(precision, 4))
        print("  Recall   :", round(recall, 4))
        print("  F1-score :", round(f1, 4))

