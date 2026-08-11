import os
from datetime import datetime


class ReportGenerator:

    def __init__(
        self,
        image_name,
        predicted_class,
        plant,
        disease,
        confidence,
        severity,
        symptoms,
        cause,
        treatment,
        prevention,
        fertilizer,
        pesticide,
        organic,
        watering,
        weather
    ):

        self.image_name = image_name
        self.predicted_class = predicted_class
        self.plant = plant
        self.disease = disease
        self.confidence = confidence
        self.severity = severity

        self.symptoms = symptoms
        self.cause = cause
        self.treatment = treatment
        self.prevention = prevention

        self.fertilizer = fertilizer
        self.pesticide = pesticide
        self.organic = organic
        self.watering = watering
        self.weather = weather

    def generate(self):

        report = f"""
===========================================================
              PLANT DISEASE DETECTION REPORT
===========================================================

Date :
{datetime.now()}

-----------------------------------------------------------

Image
{self.image_name}

Predicted Class
{self.predicted_class}

Plant
{self.plant}

Disease
{self.disease}

Confidence
{self.confidence:.2f} %

Severity
{self.severity}

-----------------------------------------------------------

Symptoms

{self.symptoms}

-----------------------------------------------------------

Cause

{self.cause}

-----------------------------------------------------------

Treatment

{self.treatment}

-----------------------------------------------------------

Prevention

{self.prevention}

-----------------------------------------------------------

Recommended Fertilizer

{self.fertilizer}

-----------------------------------------------------------

Recommended Pesticide

{self.pesticide}

-----------------------------------------------------------

Organic Control

{self.organic}

-----------------------------------------------------------

Water Requirement

{self.watering}

-----------------------------------------------------------

Weather Recommendation

{self.weather}

===========================================================
End of Report
===========================================================
"""

        os.makedirs("reports", exist_ok=True)

        report_path = os.path.join(
            "reports",
            "prediction_report.txt"
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print("\nReport Generated Successfully")

        print("Saved at :", report_path)

        return report_path