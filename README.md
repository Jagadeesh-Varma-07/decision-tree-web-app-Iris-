# Decision Tree Web Prediction App

This connects an HTML webpage to the Decision Tree model from the uploaded Jupyter Notebook.

## Inputs
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

## Output
The model predicts:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Run

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open:

http://127.0.0.1:5000

## Architecture

HTML form
    ↓
Flask backend (app.py)
    ↓
decision_tree_model.pkl
    ↓
Prediction
    ↓
HTML result
