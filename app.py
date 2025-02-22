from flask import Flask, request, jsonify
import dill as pickle  # Using dill for better handling of custom classes
import numpy as np
from flask_cors import CORS
import sys

# ----- Define your custom classes (must match those used when training) -----

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.feature_index = None
        self.split_value = None
        self.left = None
        self.right = None

    def fit(self, X, depth=0):
        if depth >= self.max_depth or X.shape[0] <= 1:
            return
        self.feature_index = np.random.randint(X.shape[1])
        min_val, max_val = X[:, self.feature_index].min(), X[:, self.feature_index].max()
        if min_val == max_val:
            return
        self.split_value = np.random.uniform(min_val, max_val)
        left_mask = X[:, self.feature_index] < self.split_value
        X_left, X_right = X[left_mask], X[~left_mask]
        self.left, self.right = IsolationTree(self.max_depth), IsolationTree(self.max_depth)
        self.left.fit(X_left, depth + 1)
        self.right.fit(X_right, depth + 1)

    def path_length(self, X):
        if self.left is None or self.right is None:
            return 1
        if X[self.feature_index] < self.split_value:
            return 1 + self.left.path_length(X)
        else:
            return 1 + self.right.path_length(X)

class IsolationForestCustom:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        self.trees = []
        for _ in range(self.n_trees):
            subsample = X[np.random.choice(X.shape[0], size=min(256, X.shape[0]), replace=False)]
            tree = IsolationTree(self.max_depth)
            tree.fit(subsample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            score = 2 ** (-avg_path_length / self.max_depth)
            scores.append(score)
        return np.array(scores)

    def predict(self, X, threshold=0.6):
        scores = self.anomaly_score(X)
        return np.where(scores > threshold, -1, 1)

# ----- Hack: Ensure that the custom classes are available in __main__ -----
# This is necessary because the model was saved with __main__ as the module name.
sys.modules['__main__'].IsolationForestCustom = IsolationForestCustom
sys.modules['__main__'].IsolationTree = IsolationTree

# ----- Load the model and encoder using dill -----
with open("custom_isolation_forest.pkl", "rb") as file:
    iso_forest = pickle.load(file)

with open("category_encoder.pkl", "rb") as file:
    category_encoder = pickle.load(file)

# ----- Flask API Setup -----
app = Flask(__name__)
CORS(app)

@app.route("/detect_anomalies", methods=["POST"])
def detect_anomalies():
    try:
        transactions = request.json
        if not isinstance(transactions, list):
            return jsonify({"error": "Expected a list of transactions"}), 400

        results = []
        for transaction in transactions:
            cuid = transaction.get("cuid")
            date = transaction.get("date")
            category = transaction.get("category")
            amount = transaction.get("amount")
            description = transaction.get("description")

            if not cuid or not date or not category or amount is None or not description:
                results.append({"cuid": cuid, "status": "error: missing values"})
                continue

            # Convert category to numeric encoding
            if category in category_encoder:
                category_encoded = category_encoder[category]
            else:
                transaction["status"] = "unknown"
                results.append(transaction)
                continue

            # Prepare input data
            X_input = np.array([[category_encoded, amount]])

            # Predict anomaly (-1 = anomalous, 1 = normal)
            prediction = iso_forest.predict(X_input)
            status = "anomalous" if prediction[0] == -1 else "normal"

            transaction["status"] = status
            results.append(transaction)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
