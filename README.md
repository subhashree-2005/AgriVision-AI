# AgriVision-AI — Recovered Files

These are rebuilt versions of everything we designed together in chat.
They are NOT pulled from your actual disk (I can't access your PC) —
place them into your real `D:\AgriVision-AI\` project, in the matching
folders, alongside your existing `datasets/`, `saved_models/`, and
`knowledge_base/disease_database.json`.

## Where each file goes

```
D:\AgriVision-AI\
├── inference\
│   ├── preprocessing.py
│   └── predict.py
├── gradcam\
│   └── gradcam_fixed.py
├── severity\
│   └── severity_estimation.py
├── weather\
│   └── weather_risk.py
├── full_pipeline.py          <- project root
├── website\
│   ├── app.py
│   └── templates\
│       ├── index.html
│       └── result.html
├── reports\
│   └── pdf_report_generator.py
└── requirements.txt           <- project root
```

## Before running anything

1. Confirm `saved_models/cnn.keras` and `saved_models/class_indices.json`
   are still where they were — these were never deleted since they're
   gitignored and only ever existed locally.
2. Open `cnn.keras` and check `model.summary()` — confirm the last
   Conv2D layer is still named `conv2d_2`. If it's different, update
   `LAST_CONV_LAYER_NAME` in `gradcam/gradcam_fixed.py`.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Recommended order to verify everything still works

1. `python inference/predict.py sample_images/early_blight.jpg`
   — confirms the model loads and predicts.
2. `python gradcam/gradcam_fixed.py sample_images/early_blight.jpg`
   — confirms Grad-CAM no longer throws "gradient is None".
3. `python severity/severity_estimation.py sample_images/early_blight.jpg`
   — confirms severity estimation runs.
4. `python full_pipeline.py sample_images/early_blight.jpg`
   — confirms everything ties together into one JSON result.
5. `cd website && python app.py`
   — open http://127.0.0.1:5000, upload a photo, confirm the result
   page renders with the image, Grad-CAM overlay, and disease info.

## What's intentionally NOT included here

- Your trained `.keras` model files (too large, never went through chat)
- Your dataset
- Your existing `knowledge_base/disease_database.json` entries — the
  pipeline reads whatever is already there; nothing here overwrites it
- Weather module needs your own free OpenWeatherMap API key
  (see `weather/weather_risk.py` for setup)

## Recommended immediate next step

Run step 1–4 above one at a time, in order, and paste me the exact
output or error for each — that lets me fix the real problem instead
of guessing, and confirms nothing important is actually missing
beyond what's rebuilt here.
