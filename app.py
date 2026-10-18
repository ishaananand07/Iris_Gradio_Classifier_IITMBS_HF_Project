"""
app.py
Gradio web app that predicts the species of an Iris flower
based on four measurements, using a pre-trained RandomForest model.
"""

import gradio as gr
import joblib
import numpy as np

# Load the trained model + class names
saved = joblib.load("iris_model.joblib")
model = saved["model"]
target_names = saved["target_names"]


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    probs = model.predict_proba(features)[0]
    # Return a dict of {class_name: probability} for gr.Label
    return {target_names[i]: float(probs[i]) for i in range(len(target_names))}


demo = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.8, label="Sepal Length (cm)"),
        gr.Slider(2.0, 4.5, value=3.0, label="Sepal Width (cm)"),
        gr.Slider(1.0, 7.0, value=4.0, label="Petal Length (cm)"),
        gr.Slider(0.1, 2.5, value=1.2, label="Petal Width (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Predicted Species"),
    title="Iris Flower Species Classifier",
    description=(
        "Adjust the sliders to enter flower measurements and predict "
        "whether it's a Setosa, Versicolor, or Virginica iris."
    ),
)

if __name__ == "__main__":
    demo.launch()
