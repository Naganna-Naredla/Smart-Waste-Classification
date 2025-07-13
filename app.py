import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
import numpy as np
import collections

# Labels map
labels_map = {
    "glass": "Dispose in glass bin.",
    "paper": "Recycle in paper bin.",
    "metal": "Recycle in metal bin.",
    "cardboard": "Recycle in cardboard bin.",
    "plastic": "Recycle in plastic bin.",
    "general": "Place in general waste bin."
}

# Load model
MODEL_PATH = "waste_model.keras"
model = load_model(MODEL_PATH)

# Initialize class counts for the Pie chart
class_counts = collections.defaultdict(int)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    instruction_text = None
    filename = None

    if request.method == "POST":
        f = request.files["file"]

        if f and f.filename != "":
            filepath = os.path.join("static", f.filename)
            f.save(filepath)

            try:
                # Preprocess image
                img = image.load_img(filepath, target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                preds = model.predict(img_array)
                confidence = float(np.max(preds[0]))
                class_idx = int(np.argmax(preds[0]))

                # Class labels in correct order
                class_labels = ["cardboard", "glass", "metal", "paper", "plastic"]

                # Confidence threshold logic
                threshold = 0.7

                if confidence >= threshold:
                    class_label = class_labels[class_idx]
                else:
                    class_label = "general"

                instruction_text = labels_map.get(class_label, "Recycle properly.")
                prediction_text = f"Predicted Class: {class_label} (Confidence: {confidence:.2f})"

                # Update pie chart counts (optional)
                class_counts[class_label] += 1
                filename = f.filename

            except UnidentifiedImageError:
                prediction_text = "Error: Uploaded file is not a valid image."
                instruction_text = ""
                filename = None

        else:
            prediction_text = "Error: No file selected."
            instruction_text = ""
            filename = None

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        instruction_text=instruction_text,
        filename=filename,
        class_counts=dict(class_counts) if class_counts else {}
    )

if __name__ == "__main__":
    app.run(debug=True)
