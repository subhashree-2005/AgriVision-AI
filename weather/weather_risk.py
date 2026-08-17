"""
weather_risk.py
-----------------
Fetches current weather for a location and returns a simple,
rule-based disease-favorability note. This does NOT diagnose
disease from weather -- it only adds supporting context.

Requires a free OpenWeatherMap API key:
https://openweathermap.org/api

Set it as an environment variable before running:
    export OPENWEATHER_API_KEY=your_key_here     (Linux/macOS)
    setx OPENWEATHER_API_KEY "your_key_here"      (Windows)
"""

import os
import requests

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(lat, lon):
    if not API_KEY:
        raise EnvironmentError(
            "OPENWEATHER_API_KEY environment variable is not set."
        )

    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric",
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "temperature_c": data["main"]["temp"],
        "humidity_percent": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "rain_expected": "rain" in data,
    }


def assess_disease_risk(weather):
    """
    Rule-based check: fungal spores generally favor humidity above
    ~80% and temperatures in the ~20-30C range. Cite a plant
    pathology source for these thresholds in your paper.
    """
    humidity = weather["humidity_percent"]
    temp = weather["temperature_c"]

    high_humidity = humidity >= 80
    favorable_temp = 20 <= temp <= 30

    if high_humidity and favorable_temp:
        risk = "High"
        note = (
            "Current humidity and temperature conditions may favor "
            "fungal disease development. Monitor affected leaves and "
            "avoid prolonged leaf wetness where possible."
        )
    elif high_humidity or favorable_temp:
        risk = "Moderate"
        note = "Conditions are partially favorable for disease spread. Keep monitoring."
    else:
        risk = "Low"
        note = "Current conditions are less favorable for rapid disease spread."

    return {"risk_level": risk, "note": note}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python weather/weather_risk.py <lat> <lon>")
        sys.exit(1)
    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    weather = get_weather(lat, lon)
    risk = assess_disease_risk(weather)
    print(weather)
    print(risk)
