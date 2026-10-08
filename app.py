from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Decision Tree model
model = joblib.load("decision_tree_model.pkl")

CLASS_NAMES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {
        "sepal_length": "",
        "sepal_width": "",
        "petal_length": "",
        "petal_width": ""
    }

    if request.method == "POST":
        try:
            values = {
                "sepal_length": request.form["sepal_length"],
                "sepal_width": request.form["sepal_width"],
                "petal_length": request.form["petal_length"],
                "petal_width": request.form["petal_width"]
            }

            features = np.array([[
                float(values["sepal_length"]),
                float(values["sepal_width"]),
                float(values["petal_length"]),
                float(values["petal_width"])
            ]])

            result = int(model.predict(features)[0])
            prediction = CLASS_NAMES.get(result, f"Class {result}")

        except (ValueError, KeyError):
            error = "Please enter valid numeric values for all four fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )

if __name__ == "__main__":
    app.run(debug=True)
