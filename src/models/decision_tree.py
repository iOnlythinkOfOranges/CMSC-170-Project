# CART Decision Tree Implementation
#
# GOAL: predict whether it will rain tomorrow (1) or stay dry (0) / Probability of Precipitation (PoP)
#
# IMPLEMENTATION:
#   1. FIT:     Recursively partition feature space to maximize variance reduction (MSE) or Gini gain.
#               Uses vectorized NumPy prefix sums for fast candidate split evaluations.
#   2. SCORE:   Traverse tree to the terminal leaf node; return the empirical sample mean of target
#               at the leaf node (representing the forecasted PoP / rain probability).
#   3. PREDICT: If PoP >= threshold, predict rain (1), else dry (0).
#
# Input format (from data/preprocess.py -> preprocess_location):
#     X = list of lists or 2D array, one row per example, already standardized
#     y = list of 0/1 labels or continuous PoP values in [0.0, 1.0]
#
# Output format (what metrics.py and comparison.py expect):
#     predict_proba -> list of P(rain) in [0.0, 1.0], one number per example
#     predict       -> list of 0/1 predictions (or continuous PoP if threshold is None)
#
# References:
#   [1] Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
#       Classification and Regression Trees. Wadsworth & Brooks/Cole.
#   [2] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
#       Monthly Weather Review, 78(1), 1-3.

import numpy as np


class DecisionTreeNode:
    """A single node (internal decision split or terminal leaf) in the CART tree."""

    def __init__(
        self,
        feature_idx=None,
        threshold=None,
        left=None,
        right=None,
        value=0.0,
        n_samples=0,
        impurity=0.0,
        gain=0.0,
        depth=0,
        is_leaf=False,
    ):
        self.feature_idx = feature_idx      # Feature index chosen for binary split
        self.threshold = threshold          # Cutoff threshold (X[:, j] <= threshold -> left)
        self.left = left                    # Left child subtree (<= threshold)
        self.right = right                  # Right child subtree (> threshold)
        self.value = float(value)           # Predicted target (sample mean at this node)
        self.n_samples = int(n_samples)     # Number of training observations in this node
        self.impurity = float(impurity)     # Node impurity (variance or Gini)
        self.gain = float(gain)             # Impurity reduction achieved by splitting this node
        self.depth = int(depth)             # Node depth in the tree (root is depth 0)
        self.is_leaf = bool(is_leaf)        # True if terminal leaf node

    def __repr__(self):
        if self.is_leaf:
            return (
                f"DecisionTreeNode(leaf=True, value={self.value:.4f}, "
                f"n_samples={self.n_samples}, depth={self.depth})"
            )
        return (
            f"DecisionTreeNode(feature={self.feature_idx}, threshold={self.threshold:.4f}, "
            f"gain={self.gain:.4f}, n_samples={self.n_samples}, depth={self.depth})"
        )


class CARTDecisionTree:
    """
    Classification and Regression Tree (CART) implemented strictly from scratch using NumPy.

    Parameters:
    -----------
    max_depth : int or None, default=5
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or until all leaves contain fewer than min_samples_split samples.
    min_samples_split : int, default=50
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=25
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at least
        min_samples_leaf training samples in each of the left and right branches.
    min_impurity_decrease : float, default=1e-7
        A node will be split if this split induces a decrease of impurity greater than
        or equal to this value.
    criterion : str, default="mse"
        The function to measure the quality of a split. Supported criteria are:
        - "mse" / "variance": Mean squared error / variance reduction (recommended for
          Probability of Precipitation continuous regression and binary 0/1 labels).
        - "gini": Gini impurity reduction (for classification).
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=25,
        min_impurity_decrease=1e-7,
        criterion="mse",
    ):
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be non-negative or None, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion.lower() if isinstance(criterion, str) else "mse"

        # Learned attributes
        self.root_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Build a CART decision tree from the training set (X, y).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training feature vectors.
        y : array-like of shape (n_samples,)
            Target values (0/1 binary labels or continuous PoP values).

        Returns:
        --------
        self : CARTDecisionTree
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows: X={X.shape[0]}, y={y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("Cannot fit decision tree on empty dataset.")

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)

        # Allocate feature importances accumulator
        raw_importances = np.zeros(n_features, dtype=float)

        # Build recursive tree
        self.root_ = self._build_tree(
            X, y, depth=0, total_samples=n_samples, raw_importances=raw_importances
        )

        # Normalize feature importances so they sum to 1.0 (if any splits occurred)
        total_imp = np.sum(raw_importances)
        if total_imp > 0:
            self.feature_importances_ = raw_importances / total_imp
        else:
            self.feature_importances_ = np.zeros(n_features, dtype=float)

        return self

    def _calculate_impurity(self, y):
        """Calculate node impurity according to self.criterion."""
        n = len(y)
        if n == 0:
            return 0.0

        if self.criterion == "gini":
            # Multi-class or binary Gini impurity: 1 - sum(p_c^2)
            classes, counts = np.unique(y, return_counts=True)
            probs = counts / n
            return float(1.0 - np.sum(probs ** 2))

        # Default: sample variance (Mean Squared Error)
        mean_y = np.mean(y)
        return float(np.mean((y - mean_y) ** 2))

    def _best_split(self, X, y, current_impurity):
        """
        Find the optimal feature and threshold that maximizes variance/Gini reduction.

        Uses vectorized NumPy cumulative sums to evaluate all candidate thresholds
        along each feature in O(D * N log N) time.
        """
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feat = None
        best_thresh = None

        if self.criterion == "gini" and len(self.classes_) > 2:
            # Multi-class Gini optimization
            K = len(self.classes_)
            Y_oh = (y[:, None] == self.classes_).astype(float)
            total_sum = np.sum(Y_oh, axis=0)
            tot_sq = np.sum(total_sum ** 2) / n_samples
            k_arr = np.arange(1, n_samples)

            for j in range(n_features):
                xj = X[:, j]
                order = np.argsort(xj)
                x_sorted = xj[order]

                diff_mask = x_sorted[:-1] != x_sorted[1:]
                valid_mask = (
                    diff_mask
                    & (k_arr >= self.min_samples_leaf)
                    & ((n_samples - k_arr) >= self.min_samples_leaf)
                )
                if not np.any(valid_mask):
                    continue

                cumsum = np.cumsum(Y_oh[order], axis=0)
                sL = cumsum[:-1][valid_mask]
                nL = k_arr[valid_mask]
                nR = n_samples - nL
                sR = total_sum - sL

                sum_sq_L = np.sum(sL ** 2, axis=1) / nL
                sum_sq_R = np.sum(sR ** 2, axis=1) / nR
                gains = (sum_sq_L + sum_sq_R - tot_sq) / n_samples

                max_idx = np.argmax(gains)
                max_gain = gains[max_idx]

                if max_gain > best_gain:
                    best_gain = max_gain
                    best_feat = j
                    valid_k = k_arr[valid_mask]
                    k_best = valid_k[max_idx]
                    best_thresh = (x_sorted[k_best - 1] + x_sorted[k_best]) / 2.0
        else:
            # Variance reduction (MSE) & Binary Gini:
            # Gain = (sum_L^2 / n_L + sum_R^2 / n_R - total_sum^2 / n) / n
            total_sum = np.sum(y)
            tot_term = (total_sum ** 2) / n_samples
            k_arr = np.arange(1, n_samples)

            for j in range(n_features):
                xj = X[:, j]
                order = np.argsort(xj)
                x_sorted = xj[order]
                y_sorted = y[order]

                diff_mask = x_sorted[:-1] != x_sorted[1:]
                valid_mask = (
                    diff_mask
                    & (k_arr >= self.min_samples_leaf)
                    & ((n_samples - k_arr) >= self.min_samples_leaf)
                )
                if not np.any(valid_mask):
                    continue

                cumsum = np.cumsum(y_sorted)
                sL = cumsum[:-1][valid_mask]
                nL = k_arr[valid_mask]
                nR = n_samples - nL
                sR = total_sum - sL

                gains = (sL ** 2 / nL + sR ** 2 / nR - tot_term) / n_samples
                if self.criterion == "gini":
                    # For binary {0, 1}, Gini gain is exactly 2 * variance reduction
                    gains *= 2.0

                max_idx = np.argmax(gains)
                max_gain = gains[max_idx]

                if max_gain > best_gain:
                    best_gain = max_gain
                    best_feat = j
                    valid_k = k_arr[valid_mask]
                    k_best = valid_k[max_idx]
                    best_thresh = (x_sorted[k_best - 1] + x_sorted[k_best]) / 2.0

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y, depth, total_samples, raw_importances):
        """Recursively construct the CART tree."""
        n_samples = len(y)
        node_value = float(np.mean(y)) if n_samples > 0 else 0.0
        impurity = self._calculate_impurity(y)

        # Base case 1: pure node
        if impurity <= 1e-12:
            return DecisionTreeNode(
                value=node_value, n_samples=n_samples, impurity=impurity, depth=depth, is_leaf=True
            )

        # Base case 2: maximum depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(
                value=node_value, n_samples=n_samples, impurity=impurity, depth=depth, is_leaf=True
            )

        # Base case 3: insufficient samples to split
        if n_samples < self.min_samples_split:
            return DecisionTreeNode(
                value=node_value, n_samples=n_samples, impurity=impurity, depth=depth, is_leaf=True
            )

        # Find the best split
        best_feat, best_thresh, best_gain = self._best_split(X, y, impurity)

        # Base case 4: no valid split found or gain is below threshold
        if (
            best_feat is None
            or best_thresh is None
            or best_gain < self.min_impurity_decrease
        ):
            return DecisionTreeNode(
                value=node_value, n_samples=n_samples, impurity=impurity, depth=depth, is_leaf=True
            )

        # Partition dataset
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # Fallback check for degenerate numerical partitions
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return DecisionTreeNode(
                value=node_value, n_samples=n_samples, impurity=impurity, depth=depth, is_leaf=True
            )

        # Accumulate feature importance: weight * gain
        raw_importances[best_feat] += (n_samples / total_samples) * best_gain

        # Recursively construct child branches
        left_child = self._build_tree(
            X[left_mask], y[left_mask], depth + 1, total_samples, raw_importances
        )
        right_child = self._build_tree(
            X[right_mask], y[right_mask], depth + 1, total_samples, raw_importances
        )

        return DecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=node_value,
            n_samples=n_samples,
            impurity=impurity,
            gain=best_gain,
            depth=depth,
            is_leaf=False,
        )

    def _predict_sample(self, node, x):
        """Traverse the tree to a leaf node for a single sample."""
        curr = node
        while not curr.is_leaf:
            if x[curr.feature_idx] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value

    def predict_proba(self, X):
        """
        Return a list with the predicted Probability of Precipitation (PoP) in [0.0, 1.0]
        for each example, matching the expected output format of metrics.py and comparison.py.
        """
        if self.root_ is None:
            raise ValueError("This CARTDecisionTree instance is not fitted yet. Call 'fit' first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = [self._predict_sample(self.root_, row) for row in X]
        # Guarantee probabilities stay within the valid [0.0, 1.0] interval
        return [float(np.clip(s, 0.0, 1.0)) for s in scores]

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels (1 = rain, 0 = dry) or continuous PoP values.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Evaluation samples.
        threshold : float or None, default=0.5
            If a float in [0, 1], returns a list of binary integers (1 if P(rain) >= threshold else 0).
            If None, returns the continuous predicted probability values (same as predict_proba).
        """
        scores = self.predict_proba(X)
        if threshold is None:
            return scores
        return [1 if s >= threshold else 0 for s in scores]

    def get_depth(self):
        """Return the maximum depth of the fitted tree."""
        if self.root_ is None:
            return 0

        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root_)

    def get_n_leaves(self):
        """Return the number of terminal leaf nodes in the fitted tree."""
        if self.root_ is None:
            return 0

        def _count(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root_)

    def export_text(self, feature_names=None, max_depth=None, spacing=3):
        """
        Return a human-readable text representation of the decision tree rules.

        Parameters:
        -----------
        feature_names : list of str, optional
            Names of the features. If None, features are labeled as 'feature_0', 'feature_1', etc.
        max_depth : int, optional
            Maximum depth to display.
        spacing : int, default=3
            Number of spaces per indentation level.
        """
        if self.root_ is None:
            return "Empty Tree (not fitted yet)"

        lines = []

        def _recurse(node, prefix=""):
            if max_depth is not None and node.depth > max_depth:
                return

            indent = " " * (node.depth * spacing)
            if node.is_leaf:
                lines.append(
                    f"{indent}|--- PoP: {node.value:.4f} (samples={node.n_samples}, impurity={node.impurity:.4f})"
                )
                return

            if feature_names is not None and node.feature_idx < len(feature_names):
                feat_name = feature_names[node.feature_idx]
            else:
                feat_name = f"feature_{node.feature_idx}"

            lines.append(
                f"{indent}|--- {feat_name} <= {node.threshold:.4f} "
                f"[gain={node.gain:.4f}, samples={node.n_samples}]"
            )
            _recurse(node.left)
            lines.append(
                f"{indent}|--- {feat_name} >  {node.threshold:.4f} "
                f"[gain={node.gain:.4f}, samples={node.n_samples}]"
            )
            _recurse(node.right)

        _recurse(self.root_)
        return "\n".join(lines)

    def print_tree(self, feature_names=None, max_depth=None):
        """Print the human-readable decision rules to stdout."""
        print(self.export_text(feature_names=feature_names, max_depth=max_depth))


def predict_cart(
    X_train,
    y_train,
    X_eval,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=25,
    threshold=0.5,
    criterion="mse",
):
    """
    Functional interface to train a CART Decision Tree and predict on evaluation data.

    Returns:
    --------
    rain_scores : list of float
        Predicted Probability of Precipitation (PoP) in [0.0, 1.0].
    predictions : list of int
        Predicted binary labels (1 = rain, 0 = dry).
    """
    tree = CARTDecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
    ).fit(X_train, y_train)

    rain_scores = tree.predict_proba(X_eval)
    predictions = tree.predict(X_eval, threshold=threshold)
    return rain_scores, predictions


# Clean aliases
DecisionTree = CARTDecisionTree
CART = CARTDecisionTree
DecisionTreeRegressor = CARTDecisionTree
DecisionTreeClassifier = CARTDecisionTree
