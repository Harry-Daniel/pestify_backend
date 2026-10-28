# Pestify Backend

A Django REST backend that identifies crop pests from a photo. It serves a
custom-trained image classification model (exported to TensorFlow Lite) and
returns the most likely pest along with confidence scores. It powers the
Pestify mobile app.

Pestify was built as a final-year university project to help farmers identify
common crop pests early, so they can respond before an infestation spreads.

**Detectable classes:** aphid, armyworm, diamondback moth, grasshopper,
whitefly, and `unknown` (no recognised pest).

## Tech stack

- **Python / Django 5.2** — HTTP API
- **LiteRT** (`ai-edge-litert`, the successor to TensorFlow Lite) — on-device-style inference without a full TensorFlow install
- **NumPy / Pillow** — image preprocessing

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/Harry-Daniel/pestify_backend.git
cd pestify_backend

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export DJANGO_DEBUG=True         # Windows (PowerShell): $env:DJANGO_DEBUG="True"
python manage.py runserver 0.0.0.0:8000
```

### Configuration

Settings are read from environment variables (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `DJANGO_DEBUG` | `False` | Enable debug mode for local development |
| `DJANGO_SECRET_KEY` | — | Required when `DJANGO_DEBUG` is off |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,10.0.2.2` | Comma-separated hostnames/IPs the server answers to |

`10.0.2.2` is how the Android emulator reaches the host machine. To test from a
physical phone on the same network, add your computer's LAN IP to
`DJANGO_ALLOWED_HOSTS`.

## API

### `POST /api/predict/`

Send an image as `multipart/form-data` in a field named `image`.

```bash
curl -F image=@leaf.jpg http://localhost:8000/api/predict/
```

**200 OK**

```json
{
  "label": "aphid",
  "confidence": 0.97,
  "top_k": [
    {"label": "aphid", "confidence": 0.97},
    {"label": "whitefly", "confidence": 0.02},
    {"label": "unknown", "confidence": 0.01}
  ]
}
```

**400 Bad Request** — no image supplied, wrong HTTP method, or the file is not
a valid image:

```json
{"error": "POST request with image required"}
```

## How it works

1. The uploaded image is converted to RGB and resized to the model's input
   size (224×224), keeping raw 0–255 pixel values.
2. The LiteRT interpreter (loaded once at startup) runs inference.
3. The class probabilities are ranked and the top 3 are returned.

Images are processed in memory and never written to disk.

## Running tests

```bash
DJANGO_DEBUG=True python manage.py test
```

## Project structure

```
pest_detect/          Django project settings and root URLs
detector/
  views.py            Model loading, preprocessing and the predict endpoint
  urls.py             /api/predict/ route
  tests.py            Endpoint tests
  tflite_model/       Trained model (best_float32.tflite)
```

## Credits

- **Backend & API** — [Harry-Daniel](https://github.com/Harry-Daniel)
- **Model training** — my project teammate, who trained the pest classification model

## License

Copyright © 2025 the Pestify authors. All rights reserved.

This repository is published for viewing as part of a portfolio. No license is
granted to copy, modify, or redistribute the code or the trained model.
