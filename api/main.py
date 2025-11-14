"""
FastAPI Backend for Neuro-Fuzzy Load Flow Prediction
Deployment-ready API for Next.js frontend integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
import numpy as np
import torch
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from neurofuzzy_model import NeuroFuzzyLoadFlowModel
from fuzzy_preprocessor import FuzzyPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Neuro-Fuzzy Load Flow Estimation API",
    description="Real-time power grid state estimation from sparse sensor data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
fuzzy_preprocessor = None
device = torch.device('cpu')  # Use CPU for serverless deployment

# =============================================================================
# Pydantic Models
# =============================================================================

class SensorInput(BaseModel):
    """Single sensor measurement input"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "measurements": [
                    0.98, None, 1.5, None, 2.3, 0.95, None, 1.8,
                    None, 2.1, 0.97, None, 1.6, None, 2.4, 0.96,
                    None, 1.7, None, 2.2
                ]
            }
        }
    )
    
    measurements: List[Optional[float]] = Field(
        ..., 
        description="List of 20 sensor measurements (use null for missing data)",
        min_length=20,
        max_length=20
    )

class BatchSensorInput(BaseModel):
    """Batch of sensor measurements"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch": [
                    [0.98, None, 1.5, None, 2.3, 0.95, None, 1.8,
                     None, 2.1, 0.97, None, 1.6, None, 2.4, 0.96,
                     None, 1.7, None, 2.2],
                    [0.99, 1.2, None, 1.9, None, 0.96, 1.4, None,
                     2.0, None, 0.98, 1.3, None, 1.8, None, 0.97,
                     1.5, None, 2.1, None]
                ]
            }
        }
    )
    
    batch: List[List[Optional[float]]] = Field(
        ...,
        description="List of sensor measurement arrays"
    )

class GridState(BaseModel):
    """Predicted grid state output"""
    voltages: Dict[str, float] = Field(..., description="Voltage magnitudes (pu) per bus")
    angles: Dict[str, float] = Field(..., description="Voltage angles (degrees) per bus")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    device: str
    version: str

class StatsResponse(BaseModel):
    """System statistics"""
    total_buses: int
    input_features: int
    model_parameters: int
    inference_time_ms: float

# =============================================================================
# Utility Functions
# =============================================================================

def load_models():
    """Load trained models on startup"""
    global model, fuzzy_preprocessor, device
    
    try:
        # Load fuzzy preprocessor
        fuzzy_path = Path(__file__).parent.parent / 'models' / 'fuzzy_preprocessor.pkl'
        with open(fuzzy_path, 'rb') as f:
            fuzzy_preprocessor = pickle.load(f)
        
        # Load neural network model
        model_path = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'neurofuzzy_best.pth'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = NeuroFuzzyLoadFlowModel(
            n_sensor_features=20,
            n_fuzzy_features=12,
            n_outputs=66,
            hidden_dims=[128, 256, 128],
            dropout_rate=0.2
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalization statistics
        model.sensor_mean = checkpoint['sensor_mean']
        model.sensor_std = checkpoint['sensor_std']
        model.output_mean = checkpoint['output_mean']
        model.output_std = checkpoint['output_std']
        
        model.eval()
        
        print("✓ Models loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def prepare_input(measurements: List[Optional[float]]) -> np.ndarray:
    """Convert sensor measurements to model input format"""
    # Convert None to np.nan
    sensor_array = np.array([m if m is not None else np.nan for m in measurements])
    return sensor_array.reshape(1, -1)

def predict_grid_state(sensor_data: np.ndarray) -> Dict:
    """Run prediction on sensor data"""
    import time
    
    start_time = time.time()
    
    # Convert to pandas DataFrame
    import pandas as pd
    sensor_df = pd.DataFrame(sensor_data)
    
    # Generate fuzzy features
    fuzzy_features = fuzzy_preprocessor.transform(sensor_df)
    
    # Prepare input for model (pass both sensor data and fuzzy features)
    X_tensor = model.preprocess_input(sensor_df.values, fuzzy_features)
    X_tensor = X_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        predictions_normalized = model(X_tensor)
    
    # Denormalize predictions using model's method
    predictions = model.denormalize_output(predictions_normalized)
    predictions_np = predictions[0]
    
    # Split into voltages and angles
    n_buses = 33
    voltages = predictions_np[:n_buses]
    angles = predictions_np[n_buses:]
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Calculate sparsity
    sparsity = np.isnan(sensor_data).mean() * 100
    
    # Calculate confidence (using fuzzy preprocessor's confidence)
    confidence_scores = fuzzy_features[0, :3]  # First 3 fuzzy features are confidence
    overall_confidence = float(np.mean(confidence_scores))
    
    return {
        'voltages': {f'bus_{i}': float(voltages[i]) for i in range(n_buses)},
        'angles': {f'bus_{i}': float(angles[i]) for i in range(n_buses)},
        'metadata': {
            'inference_time_ms': round(inference_time, 3),
            'sparsity_percent': round(sparsity, 2),
            'confidence_score': round(overall_confidence, 3),
            'available_sensors': int(np.sum(~np.isnan(sensor_data))),
            'total_sensors': sensor_data.size,
            'model_version': '1.0.0'
        }
    }

# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    success = load_models()
    if not success:
        print("WARNING: Models failed to load. API will not function properly.")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Neuro-Fuzzy Load Flow Estimation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "github": "https://github.com/aj0911/NeuroFuzzy-LoadFlow-SmartGrid"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        version="1.0.0"
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    return StatsResponse(
        total_buses=33,
        input_features=20,
        model_parameters=total_params,
        inference_time_ms=0.089  # Average from benchmarks
    )

@app.post("/predict", response_model=GridState)
async def predict(input_data: SensorInput):
    """
    Predict grid state from sparse sensor measurements
    
    - **measurements**: List of 20 sensor values (use null for missing data)
    
    Returns voltage magnitudes (pu) and angles (degrees) for all 33 buses
    """
    if model is None or fuzzy_preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare input
        sensor_array = prepare_input(input_data.measurements)
        
        # Predict
        result = predict_grid_state(sensor_array)
        
        return GridState(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[GridState])
async def predict_batch(input_data: BatchSensorInput):
    """
    Predict grid states for multiple sensor measurements (batch processing)
    
    - **batch**: List of sensor measurement arrays
    
    Returns predictions for each input in the batch
    """
    if model is None or fuzzy_preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = []
        for measurements in input_data.batch:
            sensor_array = prepare_input(measurements)
            result = predict_grid_state(sensor_array)
            results.append(GridState(**result))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/example")
async def get_example():
    """Get an example sensor input for testing"""
    # Generate random example with ~50% sparsity
    np.random.seed(42)
    measurements = []
    for i in range(20):
        if np.random.random() > 0.5:
            measurements.append(round(np.random.uniform(0.9, 2.5), 2))
        else:
            measurements.append(None)
    
    return {
        "measurements": measurements,
        "note": "Copy this to /predict endpoint for testing"
    }

# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/stats", "/predict", "/predict/batch", "/example"]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": str(exc),
        "support": "Check logs for details"
    }

# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("="*70)
    print("Starting Neuro-Fuzzy Load Flow API Server")
    print("="*70)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
