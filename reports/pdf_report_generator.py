"""
pdf_report_generator.py
-------------------------
Generates a real downloadable PDF report from a full_pipeline.py
result dict -- for farmers to save/print, and for your conference
demo.

Install once:
    pip install reportlab

Usage:
    from reports.pdf_report_generator import generate_pdf_report
    generate_pdf_report(result, "reports/output/report.pdf")
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _bullet_list(items, styles):
    """Turns a Python list of strings into a real bulleted list in the PDF,
    instead of printing the raw Python list repr."""
    if not items:
        return Paragraph("-", styles["BodyText"])
    return ListFlowable(
        [ListItem(Paragraph(str(item), styles["BodyText"])) for item in items],
        bulletType="bullet",
        leftIndent=14,
    )


def generate_pdf_report(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#2C4C28")
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#3F6B3A")
    )

    elements = []

    elements.append(Paragraph("AgriVision-AI Diagnosis Report", title_style))
    elements.append(Spacer(1, 10))

    # Image (if it exists on disk)
    uploaded_path = result.get("_local_image_path")
    if uploaded_path and os.path.exists(uploaded_path):
        elements.append(Image(uploaded_path, width=80 * mm, height=80 * mm))
        elements.append(Spacer(1, 10))

    elements.append(Paragraph("Diagnosis", heading_style))
    pred = result.get("prediction", {})
    display_name = pred.get("display_name", pred.get("class", "-"))
    data = [
        ["Result", display_name],
        ["Confidence", f'{pred.get("confidence", "-")}%'],
        ["Needs a second look?", "Yes - low confidence" if result.get("uncertain") else "No"],
    ]
    table = Table(data, colWidths=[60 * mm, 100 * mm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E8DCC0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FBF8F2")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 14))

    # Severity
    severity = result.get("severity")
    if severity and "error" not in severity:
        elements.append(Paragraph("How Bad Is It", heading_style))
        elements.append(Paragraph(
            f'{severity.get("severity_label")} - approx. '
            f'{severity.get("severity_percent")}% of leaf area affected.',
            styles["BodyText"]
        ))
        elements.append(Spacer(1, 14))

    # Weather risk (only present if location was available)
    weather_risk = result.get("weather_risk")
    if weather_risk:
        elements.append(Paragraph("Weather Risk", heading_style))
        elements.append(Paragraph(
            f'Risk level: {weather_risk.get("risk_level")}. {weather_risk.get("note", "")}',
            styles["BodyText"]
        ))
        elements.append(Spacer(1, 14))

    # Disease info -- now rendered as real bullet lists, not raw Python lists
    disease_info = result.get("disease_info", {})
    elements.append(Paragraph("What To Do", heading_style))
    if "note" in disease_info:
        elements.append(Paragraph(disease_info["note"], styles["BodyText"]))
    else:
        label_map = [
            ("symptoms", "What you'll see"),
            ("treatment", "What to do about it"),
            ("prevention", "How to stop it coming back"),
        ]
        for field, label in label_map:
            if field in disease_info:
                elements.append(Paragraph(f"<b>{label}:</b>", styles["BodyText"]))
                elements.append(_bullet_list(disease_info[field], styles))
                elements.append(Spacer(1, 8))

    elements.append(Spacer(1, 16))
    elements.append(Paragraph(
        "This report is AI-generated decision support and does not "
        "replace advice from an agricultural expert.",
        ParagraphStyle("Disclaimer", parent=styles["BodyText"],
                        textColor=colors.grey, fontSize=8)
    ))

    doc.build(elements)
    return output_path


if __name__ == "__main__":
    # Quick manual test with a fake result
    fake_result = {
        "prediction": {"class": "Tomato___Early_blight", "display_name": "Tomato — Early Blight", "confidence": 94.32},
        "uncertain": False,
        "severity": {"severity_label": "Moderate", "severity_percent": 18.4},
        "disease_info": {
            "symptoms": ["Dark concentric spots on lower leaves.", "Yellowing around lesions."],
            "treatment": ["Remove affected leaves.", "Apply an approved fungicide per label."],
            "prevention": ["Improve airflow.", "Avoid overhead irrigation."],
        },
    }
    path = generate_pdf_report(fake_result, "reports/output/test_report.pdf")
    print("Saved:", path)