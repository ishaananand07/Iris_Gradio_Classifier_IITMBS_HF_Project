# Iris Flower Species Classifier 🌸

A simple machine learning web app that predicts the species of an Iris flower (Setosa, Versicolor, or Virginica) based on four measurements, built with **scikit-learn** and **Gradio**.

Built as a hands-on project after completing the IIT Madras BS Degree Program workshop *"Getting Started with Hugging Face: Models, Datasets and AI Applications."*

## Demo

Move the sliders to enter sepal length, sepal width, petal length, and petal width (in cm) — the app predicts the most likely species along with confidence scores for all three classes.

## How it works

1. **`train_model.py`** — trains a `RandomForestClassifier` on the classic Iris dataset (built into scikit-learn) and saves it as `iris_model.joblib`
2. **`app.py`** — loads the saved model and wraps it in a Gradio `Interface`, taking four slider inputs and returning a `Label` output with per-class probabilities
3. **`requirements.txt`** — lists all dependencies needed to run the app

## Project structure

```
.
├── app.py              # Gradio app (run this)
├── train_model.py      # Trains and saves the model
├── iris_model.joblib    # Pre-trained model (already included)
└── requirements.txt    # Python dependencies
```

## Setup and running locally

Clone the repo and move into it:

```bash
git clone <your-repo-url>
cd iris-gradio-classifier
```

(Optional but recommended) create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python3 app.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`) in your browser.

> The trained model (`iris_model.joblib`) is already included, so you can run `app.py` directly. If you'd like to retrain it yourself, run `python3 train_model.py` first.

## Tech stack

- Python
- scikit-learn (RandomForestClassifier)
- Gradio (web interface)
- joblib (model persistence)

## License

Free to use for learning purposes.
