"""
app.py
------
Flask website for AgriVision-AI.
Run from the website/ folder:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os
import sys
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from full_pipeline import run_full_pipeline  # noqa: E402
from reports.pdf_report_generator import generate_pdf_report  # noqa: E402

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
PDF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "reports")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


# Serves Grad-CAM images, which are saved to the project's outputs/
# folder (outside website/static/, so Flask needs an explicit route).
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Optional: browser geolocation, sent as hidden form fields.
    # If missing/empty, weather is simply skipped -- not an error.
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    try:
        lat = float(lat) if lat else None
        lon = float(lon) if lon else None
    except ValueError:
        lat, lon = None, None

    try:
        result = run_full_pipeline(
            save_path, include_gradcam=True, include_severity=True,
            lat=lat, lon=lon,
        )
    except Exception as e:
        return render_template("index.html", error=str(e))

    result["uploaded_image_url"] = url_for("static", filename=f"uploads/{unique_name}")

    # Generate the PDF report right away so the download link is ready.
    try:
        pdf_name = f"{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(PDF_FOLDER, pdf_name)
        generate_pdf_report(result, pdf_path)
        result["pdf_url"] = url_for("static", filename=f"reports/{pdf_name}")
    except Exception as e:
        result["pdf_url"] = None
        result["pdf_error"] = str(e)

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)