# Run this file once to create decision_tree_model.pkl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# This matches the final model in the uploaded notebook:
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))

joblib.dump(model, "decision_tree_model.pkl")
print("Saved: decision_tree_model.pkl")
