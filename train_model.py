"""
train_model.py
Trains a simple RandomForest classifier on the classic Iris dataset
and saves it as iris_model.joblib for the Gradio app to load.

Run this once: python train_model.py
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.2%}")

# Save model + target names together
joblib.dump({"model": model, "target_names": target_names}, "iris_model.joblib")
print("Saved model to iris_model.joblib")
