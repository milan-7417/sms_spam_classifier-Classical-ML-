"""
SpamGuard — SMS Spam Classifier Web Application.

FastAPI backend serving the web interface, static assets,
and JSON REST API for SMS spam detection.
Loads saved ML models once at startup and evaluates messages dynamically.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.predict import load_model_and_vectorizer, predict_message

app = FastAPI(
    title="SpamGuard API",
    description="Intelligent SMS Spam Detection using Classical NLP and Machine Learning",
    version="1.0.0",
)

# Static files and template engine
static_dir = project_root / "static"
templates_dir = project_root / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Global model state (loaded once at startup)
classifier: Optional[Any] = None
vectorizer: Optional[Any] = None
metadata: Dict[str, Any] = {}
load_error: Optional[str] = None

try:
    classifier, vectorizer, metadata = load_model_and_vectorizer()
    print(f" Loaded model successfully: {metadata.get('best_model_name', 'Classifier')}")
except Exception as e:
    load_error = str(e)
    print(f" Warning: Could not load model artifacts on startup: {load_error}")
    print(" Please run 'python src/train.py' to generate trained model artifacts.")


class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1600,
        description="The SMS message text to analyze (1 to 1600 characters).",
        json_schema_extra={"example": "WINNER!! You have won a £900 prize reward! Call 09061701461 to claim."},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the clean main classifier playground interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "metadata": metadata,
            "model_loaded": (classifier is not None and vectorizer is not None),
            "load_error": load_error,
        },
    )


@app.get("/documentation", response_class=HTMLResponse)
async def documentation(request: Request):
    """
    Render dedicated documentation page explaining in short:
    - How SpamGuard was made (dataset, preprocessing, feature engineering, models)
    - How it works (runtime NLP classification workflow)
    - Model evaluation and accuracy summary
    """
    return templates.TemplateResponse(
        request=request,
        name="docs.html",
        context={
            "metadata": metadata,
            "model_loaded": (classifier is not None and vectorizer is not None),
            "load_error": load_error,
        },
    )


@app.get("/metrics")
async def get_metrics():
    """Return trained model evaluation metrics and comparison table."""
    if not metadata:
        raise HTTPException(status_code=404, detail="Metrics not available. Run training pipeline first.")
    return JSONResponse(content=metadata)


@app.post("/predict")
async def predict(payload: MessageRequest):
    """
    Classify an incoming SMS message.

    Expected JSON body:
    {
        "message": "Free entry into competition..."
    }

    Returns JSON prediction object.
    """
    if classifier is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Machine learning model is not loaded. "
                "Please run 'python src/train.py' in the project directory first."
            ),
        )

    message_text = payload.message.strip()
    if not message_text:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty or whitespace only.",
        )

    try:
        result = predict_message(message_text, classifier, vectorizer, metadata)
        return JSONResponse(content=result, status_code=200)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as ex:
        print(f"Prediction error: {ex}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during classification.",
        )


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    print(f"Starting SpamGuard on FastAPI server (http://{host}:{port})...")
    uvicorn.run("app:app", host=host, port=port, reload=False)
