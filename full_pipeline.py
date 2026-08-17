"""
full_pipeline.py
------------------
The single function the website (Flask/Django) should call.
Ties together: prediction -> Grad-CAM -> severity -> knowledge base
-> (optional) weather context, into one clean result dict.

Usage:
    python full_pipeline.py path/to/photo.jpg
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.predict import predict_image          # noqa: E402
from gradcam.gradcam_fixed import run_gradcam         # noqa: E402
from severity.severity_estimation import estimate_severity  # noqa: E402

KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base", "disease_database.json")

# Below this confidence, the system should say "uncertain" instead
# of forcing a diagnosis. Tune based on your validation results.
CONFIDENCE_THRESHOLD = 60.0


def _load_knowledge_base():
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return {}
    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        return json.load(f)


def humanize_class_name(class_name):
    """
    Converts a raw model class label like 'Tomato___Early_blight'
    into farmer-readable pieces:
        plant   -> "Tomato"
        disease -> "Early Blight"
        display -> "Tomato — Early Blight"  (or "Healthy Tomato Leaf")
    """
    if "___" in class_name:
        plant_raw, disease_raw = class_name.split("___", 1)
    else:
        plant_raw, disease_raw = class_name, ""

    plant = plant_raw.replace("_", " ").replace(",", "").strip()
    disease = disease_raw.replace("_", " ").strip()

    if disease.lower() == "healthy":
        display = f"Healthy {plant} Leaf"
    elif disease:
        display = f"{plant} — {disease}"
    else:
        display = plant

    return {"plant": plant, "disease": disease or "Healthy", "display": display}


def run_full_pipeline(image_path, include_gradcam=True, include_severity=True,
                       lat=None, lon=None):
    """
    Returns a single dict combining every stage of the pipeline.
    Any stage that fails is caught and reported instead of crashing
    the whole pipeline, so a farmer-facing website can still show a
    partial result gracefully.
    """
    result = {"image": os.path.basename(image_path)}
    # Kept so the PDF report generator can embed the actual photo.
    result["_local_image_path"] = os.path.abspath(image_path)

    # 1. Prediction
    prediction = predict_image(image_path)
    names = humanize_class_name(prediction["predicted_class"])
    result["prediction"] = {
        "class": prediction["predicted_class"],
        "plant": names["plant"],
        "disease": names["disease"],
        "display_name": names["display"],
        "confidence": prediction["confidence"],
        "top_k": [
            {
                **entry,
                "display_name": humanize_class_name(entry["class"])["display"],
            }
            for entry in prediction["top_k"]
        ],
    }
    result["uncertain"] = prediction["confidence"] < CONFIDENCE_THRESHOLD

    # 2. Knowledge base lookup
    kb = _load_knowledge_base()
    disease_info = kb.get(prediction["predicted_class"])
    result["disease_info"] = disease_info if disease_info else {
        "note": "No knowledge-base entry yet for this class. Add one to disease_database.json."
    }

    # 3. Severity (only meaningful if not a "healthy" class)
    if include_severity:
        try:
            result["severity"] = estimate_severity(image_path)
        except Exception as e:
            result["severity"] = {"error": str(e)}

    # 4. Grad-CAM
    if include_gradcam:
        try:
            _overlay, _pred_class = run_gradcam(image_path, save=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result["gradcam_image"] = f"outputs/gradcam_{base_name}.jpg"
        except Exception as e:
            result["gradcam_image"] = None
            result["gradcam_error"] = str(e)

    # 5. Weather (optional — only if lat/lon provided)
    if lat is not None and lon is not None:
        try:
            from weather.weather_risk import get_weather, assess_disease_risk
            weather = get_weather(lat, lon)
            result["weather"] = weather
            result["weather_risk"] = assess_disease_risk(weather)
        except Exception as e:
            result["weather"] = None
            result["weather_error"] = str(e)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python full_pipeline.py path/to/image.jpg")
        sys.exit(1)

    output = run_full_pipeline(sys.argv[1])
    print(json.dumps(output, indent=2, default=str))