"""
FastAPI backend for the Neuro‑Fuzzy Load Flow project
=====================================================

This module exposes a REST API for real‑time power grid state estimation.
It loads a pre‑trained fuzzy preprocessor and neural network from the
``models`` directory at startup and provides endpoints for health checks,
single and batch predictions.  The implementation is deliberately
lightweight to meet Vercel's serverless function constraints.  Only
essential dependencies are imported, and model files are loaded from
relative paths so that the API can be deployed as a standalone unit
without bundling the entire training environment.

The API exposes the following endpoints:

* ``GET /`` – root endpoint with basic information.
* ``GET /health`` – returns the status of the API and whether models
  were loaded.
* ``GET /stats`` – returns simple statistics about the model and
  configuration.
* ``POST /predict`` – accepts a JSON payload with 20 sensor
  measurements and returns predicted voltages and angles for 33 buses.
* ``POST /predict/batch`` – accepts a list of sensor measurement lists
  and returns a list of predictions.
* ``GET /example`` – generates a random example input for quick testing.

The implementation assumes that ``fuzzy_preprocessor.pkl`` and
``neurofuzzy_best.pth`` reside in a ``models`` directory adjacent to
this file.  Adjust ``fuzzy_path`` and ``model_path`` in the
``load_models`` function if your repository structure differs.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Add the ``src`` directory to the Python path so that we can import
# ``neurofuzzy_model`` and ``fuzzy_preprocessor``.  This uses a
# relative path based on the location of this file, which is robust
# when the code is deployed on Vercel or run locally.  Avoid using
# absolute paths or hard‑coding ``sys.path`` entries; modifying
# ``sys.path`` at runtime is discouraged, but it's pragmatic here.
import sys
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from neurofuzzy_model import NeuroFuzzyLoadFlowModel  
from fuzzy_preprocessor import FuzzyPreprocessor  


# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Neuro‑Fuzzy Load Flow Estimation API",
    description=(
        "Real‑time power grid state estimation from sparse sensor data using a "
        "hybrid neuro‑fuzzy approach."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS from any origin during development.  In production you
# should restrict this list to trusted domains (e.g. your Next.js app).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model variables
# ---------------------------------------------------------------------------

model: Optional[NeuroFuzzyLoadFlowModel] = None
fuzzy_preprocessor: Optional[FuzzyPreprocessor] = None
device: torch.device = torch.device("cpu")  # use CPU in serverless context


# ---------------------------------------------------------------------------
# Pydantic schemas for request/response bodies
# ---------------------------------------------------------------------------

class SensorInput(BaseModel):
    """Input schema for a single prediction request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "measurements": [
                    0.98,
                    None,
                    1.5,
                    None,
                    2.3,
                    0.95,
                    None,
                    1.8,
                    None,
                    2.1,
                    0.97,
                    None,
                    1.6,
                    None,
                    2.4,
                    0.96,
                    None,
                    1.7,
                    None,
                    2.2,
                ]
            }
        }
    )

    measurements: List[Optional[float]] = Field(
        ..., description="List of 20 sensor measurements (use null for missing)", min_length=20, max_length=20
    )


class BatchSensorInput(BaseModel):
    """Input schema for batch prediction requests."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch": [
                    [
                        0.98,
                        None,
                        1.5,
                        None,
                        2.3,
                        0.95,
                        None,
                        1.8,
                        None,
                        2.1,
                        0.97,
                        None,
                        1.6,
                        None,
                        2.4,
                        0.96,
                        None,
                        1.7,
                        None,
                        2.2,
                    ],
                    [
                        0.99,
                        1.2,
                        None,
                        1.9,
                        None,
                        0.96,
                        1.4,
                        None,
                        2.0,
                        None,
                        0.98,
                        1.3,
                        None,
                        1.8,
                        None,
                        0.97,
                        1.5,
                        None,
                        2.1,
                        None,
                    ],
                ]
            }
        }
    )

    batch: List[List[Optional[float]]] = Field(
        ..., description="List of sensor measurement arrays"
    )


class GridState(BaseModel):
    """Output schema for a prediction response."""

    voltages: Dict[str, float] = Field(..., description="Voltage magnitudes (pu) per bus")
    angles: Dict[str, float] = Field(..., description="Voltage angles (degrees) per bus")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")


class HealthResponse(BaseModel):
    """Schema for the health check endpoint."""

    status: str
    model_loaded: bool
    device: str
    version: str


class StatsResponse(BaseModel):
    """Schema for system statistics."""

    total_buses: int
    input_features: int
    model_parameters: int
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

import numpy as np            # make sure this import is present at top
import torch
# ...

def load_models():
    global model, fuzzy_preprocessor, device
    try:
        # 1) Fuzzy preprocessor first (so its class is importable)
        with open(BASE_DIR / "models" / "fuzzy_preprocessor.pkl", "rb") as f:
            fuzzy_preprocessor = pickle.load(f)

        # 2) Allow legacy pickled content by disabling weights_only
        checkpoint = torch.load(
            BASE_DIR / "models" / "checkpoints" / "neurofuzzy_best.pth",
            map_location=device,
            weights_only=False   # <-- change this to False
        )

        # 3) Rebuild model and load state
        model = NeuroFuzzyLoadFlowModel(
            n_sensor_features=20, n_fuzzy_features=12, n_outputs=66,
            hidden_dims=[128, 256, 128], dropout_rate=0.2
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 4) Attach normalization stats
        model.sensor_mean = checkpoint['sensor_mean']
        model.sensor_std  = checkpoint['sensor_std']
        model.output_mean = checkpoint['output_mean']
        model.output_std  = checkpoint['output_std']

        model.eval()
        print("✓ Models loaded successfully")
        return True

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


def prepare_input(measurements: List[Optional[float]]) -> np.ndarray:
    """Convert a list of measurements to a numpy array with NaNs for missing values."""
    arr = np.array([m if m is not None else np.nan for m in measurements], dtype=float)
    return arr.reshape(1, -1)


def predict_grid_state(sensor_data: np.ndarray) -> Dict[str, Any]:
    """Run inference on a single set of sensor measurements.

    Parameters
    ----------
    sensor_data : np.ndarray
        A 2D array of shape (1, 20) containing the raw sensor measurements with
        NaNs indicating missing values.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``voltages``, ``angles`` and ``metadata``.
    """

    if model is None or fuzzy_preprocessor is None:
        raise RuntimeError("Models are not loaded; call load_models() first.")

    start = time.time()

    # Convert to DataFrame for the fuzzy preprocessor
    sensor_df = pd.DataFrame(sensor_data)

    # Generate fuzzy features
    fuzzy_features = fuzzy_preprocessor.transform(sensor_df)  

    # Preprocess input for the neural network
    X_tensor = model.preprocess_input(sensor_df.values, fuzzy_features)
    X_tensor = X_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        y_normalized = model(X_tensor)
    y_denorm = model.denormalize_output(y_normalized)
    predictions = y_denorm[0]

    # Split predictions into voltages and angles
    n_buses = 33
    voltages = predictions[:n_buses]
    angles = predictions[n_buses:]

    duration_ms = (time.time() - start) * 1000.0

    # Calculate sparsity and confidence
    sparsity = float(np.isnan(sensor_data).mean() * 100.0)
    confidence_scores = fuzzy_features[0, :3]  # first three fuzzy features are confidence
    confidence = float(np.mean(confidence_scores))

    return {
        "voltages": {f"bus_{i}": float(voltages[i]) for i in range(n_buses)},
        "angles": {f"bus_{i}": float(angles[i]) for i in range(n_buses)},
        "metadata": {
            "inference_time_ms": round(duration_ms, 3),
            "sparsity_percent": round(sparsity, 2),
            "confidence_score": round(confidence, 3),
            "available_sensors": int(np.sum(~np.isnan(sensor_data))),
            "total_sensors": sensor_data.size,
            "model_version": "1.0.0",
        },
    }


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    """Load models when the application starts."""
    success = load_models()
    if not success:
        print("WARNING: Models failed to load.  The API will not function properly.")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Return basic information about the API."""
    return {
        "message": "Neuro‑Fuzzy Load Flow Estimation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "github": "https://github.com/aj0911/NeuroFuzzy-LoadFlow-SmartGrid",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check to verify that the API and models are operational."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        version="1.0.0",
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Return static statistics about the system."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    total_params = sum(p.numel() for p in model.parameters())  
    return StatsResponse(
        total_buses=33,
        input_features=20,
        model_parameters=total_params,
        inference_time_ms=0.089,
    )


@app.post("/predict", response_model=GridState)
async def predict(input_data: SensorInput) -> GridState:
    """Predict the grid state for a single set of sensor measurements."""
    if model is None or fuzzy_preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        sensor_array = prepare_input(input_data.measurements)
        result = predict_grid_state(sensor_array)
        return GridState(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post("/predict/batch", response_model=List[GridState])
async def predict_batch(input_data: BatchSensorInput) -> List[GridState]:
    """Predict the grid state for a batch of sensor measurements."""
    if model is None or fuzzy_preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        outputs: List[GridState] = []
        for measurements in input_data.batch:
            sensor_array = prepare_input(measurements)
            result = predict_grid_state(sensor_array)
            outputs.append(GridState(**result))
        return outputs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")


@app.get("/example")
async def example() -> Dict[str, Any]:
    """Return a randomly generated example measurement list for testing."""
    # Generate a list with roughly 50 % sparsity
    rng = np.random.default_rng()
    measurements: List[Optional[float]] = []
    for _ in range(20):
        if rng.random() > 0.5:
            # draw a value between 0.9 and 2.5 inclusive
            measurements.append(round(float(rng.uniform(0.9, 2.5)), 2))
        else:
            measurements.append(None)
    return {
        "measurements": measurements,
        "note": "Copy this array into the /predict endpoint body to test the API.",
    }


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request, exc):  
    """Handle unknown endpoints."""
    return {
        "error": "Endpoint not found",
        "available_endpoints": [
            "/",
            "/health",
            "/stats",
            "/predict",
            "/predict/batch",
            "/example",
        ],
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):  
    """Handle unexpected server errors."""
    return {
        "error": "Internal server error",
        "message": str(exc),
        "support": "Check logs for details",
    }


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running locally with ``python server.py`` for testing.  Note
    # that Vercel will ignore this block and use the default ASGI
    # handler provided by FastAPI.
    import uvicorn

    print("=" * 70)
    print("Starting Neuro‑Fuzzy Load Flow API Server")
    print("=" * 70)
    print("Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)