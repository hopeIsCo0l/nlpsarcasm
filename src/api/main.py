"""
FastAPI application for sarcasm detection API.
Provides endpoints for model predictions and health checks.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import os
import sys
import joblib
from datetime import datetime

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our model with error handling
try:
    from models.sarcasm_detector import SarcasmDetector
except ImportError:
    # Fallback import
    sys.path.append(os.path.dirname(parent_dir))
    from src.models.sarcasm_detector import SarcasmDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sarcasm Detection API",
    description="A machine learning API for detecting sarcasm in headlines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
sarcasm_detector = None
model_loaded = False

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    headline: str = Field(..., description="The headline to analyze for sarcasm", min_length=1, max_length=500)
    
class PredictionResponse(BaseModel):
    headline: str
    is_sarcastic: bool
    confidence: float
    sarcastic_probability: float
    non_sarcastic_probability: float
    processed_text: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    headlines: List[str] = Field(..., description="List of headlines to analyze", min_items=1, max_items=100)

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    sarcastic_count: int
    non_sarcastic_count: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]]
    timestamp: str

class ModelInfoResponse(BaseModel):
    status: str
    model_type: Optional[str]
    training_date: Optional[str]
    performance_metrics: Optional[Dict[str, float]]
    vectorizer_type: Optional[str]
    model_type_name: Optional[str]

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global sarcasm_detector, model_loaded
    
    logger.info("Starting Sarcasm Detection API...")
    
    # Try to load the best model
    model_path = "../../models/best_model.pkl"
    if os.path.exists(model_path):
        try:
            sarcasm_detector = SarcasmDetector()
            sarcasm_detector.load_model(model_path)
            model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model_loaded = False
    else:
        logger.warning("No trained model found. Please train a model first.")
        model_loaded = False

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sarcasm Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global sarcasm_detector, model_loaded
    
    model_info = None
    if sarcasm_detector and model_loaded:
        model_info = sarcasm_detector.get_model_info()
    
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed information about the loaded model."""
    global sarcasm_detector, model_loaded
    
    if not model_loaded or sarcasm_detector is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    model_info = sarcasm_detector.get_model_info()
    
    return ModelInfoResponse(
        status=model_info.get('status', 'unknown'),
        model_type=model_info.get('model_type'),
        training_date=model_info.get('training_history', {}).get('training_date'),
        performance_metrics=model_info.get('training_history', {}).get('metrics'),
        vectorizer_type=model_info.get('vectorizer_type'),
        model_type_name=model_info.get('model_type_name')
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_sarcasm(request: PredictionRequest):
    """Predict sarcasm for a single headline."""
    global sarcasm_detector, model_loaded
    
    if not model_loaded or sarcasm_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        result = sarcasm_detector.predict(request.headline)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_sarcasm_batch(request: BatchPredictionRequest):
    """Predict sarcasm for multiple headlines."""
    global sarcasm_detector, model_loaded
    
    if not model_loaded or sarcasm_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make batch predictions
        results = sarcasm_detector.predict_batch(request.headlines)
        
        # Add timestamps
        for result in results:
            result['timestamp'] = datetime.now().isoformat()
        
        # Calculate summary statistics
        sarcastic_count = sum(1 for r in results if r['is_sarcastic'])
        non_sarcastic_count = len(results) - sarcastic_count
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**result) for result in results],
            total_count=len(results),
            sarcastic_count=sarcastic_count,
            non_sarcastic_count=non_sarcastic_count,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model from disk."""
    global sarcasm_detector, model_loaded
    
    def _reload_model():
        global sarcasm_detector, model_loaded
        
        try:
            model_path = "../../models/best_model.pkl"
            if os.path.exists(model_path):
                sarcasm_detector = SarcasmDetector()
                sarcasm_detector.load_model(model_path)
                model_loaded = True
                logger.info("Model reloaded successfully")
            else:
                logger.error("Model file not found")
                model_loaded = False
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            model_loaded = False
    
    background_tasks.add_task(_reload_model)
    
    return {"message": "Model reload initiated", "timestamp": datetime.now().isoformat()}

@app.get("/examples")
async def get_examples():
    """Get example headlines for testing."""
    examples = {
        "sarcastic_examples": [
            "Scientists discover that coffee is actually good for you",
            "Breaking: Water is wet, study confirms",
            "Man shocked to learn that his cat doesn't actually care about his problems",
            "Local man surprised that his diet of pizza and beer isn't working",
            "Study finds that people who exercise regularly are healthier"
        ],
        "non_sarcastic_examples": [
            "New study shows benefits of Mediterranean diet",
            "Scientists discover new species of deep-sea creatures",
            "Global temperatures continue to rise, climate report shows",
            "Tech company announces breakthrough in renewable energy",
            "Medical researchers develop new treatment for rare disease"
        ]
    }
    return examples

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 