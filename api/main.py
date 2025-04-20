from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any
import joblib
import numpy as np
import os
import re
import uvicorn
import logging
from datetime import datetime
import json
from contextlib import asynccontextmanager
import cloudpickle

from fusion import FusionModel
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cybersecurity-api")

# Get current directory (the "api" folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# Change the model directory to point one level up from CURRENT_DIR, then "models"
###############################################################################
if os.environ.get("DOCKER_ENV", "0") == "1":
    MODEL_DIR = "/models"  # Used inside Docker
else:
    MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models"))  # Used locally


# Model paths
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model2_cicids.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler.joblib')
PCA_PATH = os.path.join(MODEL_DIR, 'incremental_pca.joblib')
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, 'model1_feature_order.joblib')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'logistic_regression_vectorizer.joblib')
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, 'fusion_model.joblib')

# Print out paths for debugging
logger.info(f"Current directory: {CURRENT_DIR}")
logger.info(f"Model directory: {MODEL_DIR}")
logger.info("Model paths:")
logger.info(f"  RF_MODEL_PATH: {RF_MODEL_PATH}")
logger.info(f"  SCALER_PATH: {SCALER_PATH}")
logger.info(f"  PCA_PATH: {PCA_PATH}")
logger.info(f"  FEATURE_ORDER_PATH: {FEATURE_ORDER_PATH}")
logger.info(f"  LR_MODEL_PATH: {LR_MODEL_PATH}")
logger.info(f"  VECTORIZER_PATH: {VECTORIZER_PATH}")
logger.info(f"  FUSION_MODEL_PATH: {FUSION_MODEL_PATH}")

# Using lifespan context manager instead of on_event (modern FastAPI approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    logger.info("Loading models...")
    try:
        # Check if files exist before loading
        for path, name in [
            (RF_MODEL_PATH, "RF Model"),
            (SCALER_PATH, "Scaler"),
            (PCA_PATH, "PCA"),
            (FEATURE_ORDER_PATH, "Feature Order"),
            (LR_MODEL_PATH, "LR Model"),
            (VECTORIZER_PATH, "Vectorizer")
        ]:
            if not os.path.exists(path):
                logger.warning(f"{name} file not found at: {path}")
            else:
                logger.info(f"{name} file found at: {path}")
        
        # Load models and preprocessors
        app.state.rf_model = joblib.load(RF_MODEL_PATH)
        app.state.scaler = joblib.load(SCALER_PATH)
        app.state.pca = joblib.load(PCA_PATH)
        app.state.feature_order = joblib.load(FEATURE_ORDER_PATH)
        app.state.lr_model = joblib.load(LR_MODEL_PATH)
        app.state.vectorizer = joblib.load(VECTORIZER_PATH)
        
        # Try to load fusion model, create if it doesn't exist
        if os.path.exists(FUSION_MODEL_PATH):
            logger.info(f"Loading fusion model from: {FUSION_MODEL_PATH}")
            with open(FUSION_MODEL_PATH, 'rb') as f:
               app.state.fusion_model = cloudpickle.load(f)

        else:
            logger.warning(f"Fusion model not found at: {FUSION_MODEL_PATH}. Creating new instance.")
            import sys
            sys.path.append(CURRENT_DIR)  # Make sure fusion.py can be found
            try:
                from fusion import FusionModel
                logger.info("Successfully imported fusion model")
                
                # Create new instance
                app.state.fusion_model = FusionModel(auto_load=False)
                
                # Set attributes required by the fusion model
                app.state.fusion_model.rf_model = app.state.rf_model
                app.state.fusion_model.scaler = app.state.scaler
                app.state.fusion_model.pca = app.state.pca
                app.state.fusion_model.feature_order = app.state.feature_order
                app.state.fusion_model.lr_model = app.state.lr_model
                app.state.fusion_model.vectorizer = app.state.vectorizer
                
                # Add any missing attributes that the fusion model might need
                if not hasattr(app.state.fusion_model, 'MODEL1_WEIGHT'):
                    app.state.fusion_model.MODEL1_WEIGHT = 0.7
                if not hasattr(app.state.fusion_model, 'MODEL2_WEIGHT'):
                    app.state.fusion_model.MODEL2_WEIGHT = 0.3
                if not hasattr(app.state.fusion_model, 'THRESHOLDS'):
                    app.state.fusion_model.THRESHOLDS = {
                        'ignore': (0.0, 0.4),
                        'monitor': (0.4, 0.7),
                        'alert': (0.7, 1.0)
                    }
                
                logger.info("Fusion model initialized successfully")
                
                # Save the model for future use
                with open(FUSION_MODEL_PATH, 'wb') as f:
                   cloudpickle.dump(app.state.fusion_model, f)

            except Exception as e:
                logger.error(f"Error creating fusion model: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    yield
    
    # Clean up resources if needed when shutting down
    logger.info("Shutting down application")

# Create the FastAPI app
app = FastAPI(
    title="Cyber Attack Detection API",
    description="API for detecting cyber attacks using machine learning models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (the rest of your code stays the same) ...


# Pydantic models for request validation
class NetworkTrafficFeatures(BaseModel):
    """Model for network traffic features."""
    features: List[float] = Field(..., title="Network Traffic Features", 
                                description="List of network traffic features")
    feature_names: Optional[List[str]] = Field(None, title="Feature Names",
                                            description="Optional list of feature names")

class AlertMessage(BaseModel):
    """Model for alert message."""
    message: str = Field(..., title="Alert Message",
                        description="Security alert message text")

class CombinedInput(BaseModel):
    """Model for combined network traffic and alert input."""
    # Either traffic_features or network_data can be provided
    traffic_features: Optional[List[float]] = Field(None, title="Network Traffic Features")
    network_data: Optional[List[float]] = Field(None, title="Network Data (Alternative)")
    alert_message: Optional[str] = Field(None, title="Alert Message")
    
    # Root validator to ensure one data source is provided
    @root_validator(pre=True)
    def check_data_provided(cls, values):
        if not values.get('traffic_features') and not values.get('network_data'):
            raise ValueError('Either traffic_features or network_data must be provided')
        # If network_data is provided but traffic_features is not, copy it
        if not values.get('traffic_features') and values.get('network_data'):
            values['traffic_features'] = values['network_data']
        return values

class Model1Response(BaseModel):
    """Response model for network traffic analysis (Model 1)."""
    predicted_class: str
    confidence: float
    prediction_time: str

class Model2Response(BaseModel):
    """Response model for alert text analysis (Model 2)."""
    attack_probability: float
    prediction_time: str

class FusionResponse(BaseModel):
    """Response model for fusion model."""
    predicted_class: str
    threat_score: float
    recommended_action: str
    model1_confidence: float
    model2_confidence: float
    model1_weight: float
    model2_weight: float
    prediction_time: str

# Helper function to map input features to expected format
def map_input_to_feature_names(raw_input_list):
    """Maps raw input list values to their corresponding feature names."""
    # Standard feature names expected by the model
    standard_feature_names = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
        "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
        "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
        "Avg Bwd Segment Size", "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
        "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
        "Idle Std", "Idle Max", "Idle Min"
    ]
    
    # Handle list length mismatches
    if len(raw_input_list) > len(standard_feature_names):
        logger.warning(f"Input has {len(raw_input_list)} values, but only {len(standard_feature_names)} features will be used")
        raw_input_list = raw_input_list[:len(standard_feature_names)]
    elif len(raw_input_list) < len(standard_feature_names):
        logger.warning(f"Input has only {len(raw_input_list)} values, padding with zeros")
        raw_input_list = raw_input_list + [0] * (len(standard_feature_names) - len(raw_input_list))
    
    # Create a dictionary mapping feature names to values
    return dict(zip(standard_feature_names, raw_input_list))

# Helper function to clean alert messages
def clean_alert_message(text):
    """Clean alert messages by removing IPs, ports, etc."""
    # Convert to string if not already
    text = str(text)
    
    # Remove IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
    
    # Remove port numbers
    text = re.sub(r'\b(?:on|to|from|at|using)\s+port\s+\d+\b', ' ', text)
    text = re.sub(r'\bport\s+\d+\b', '', text)
    
    # Remove protocol numbers
    text = re.sub(r'\bprotocol\s+\d+\b', 'protocol', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Helper function to generate alert messages from traffic data
def generate_alert_message(structured_data):
    """Generate a synthetic alert message from network data."""
    # Convert to dictionary if array
    if isinstance(structured_data, (list, np.ndarray)):
        # Convert to dictionary with feature names
        if isinstance(structured_data, np.ndarray):
            structured_data = structured_data.tolist()
        
        # Map to feature dictionary
        structured_data = map_input_to_feature_names(structured_data)
        
        # Extract useful fields for alert generation
        dest_port = structured_data.get('Destination Port', 0)
        flow_packets = structured_data.get('Flow Packets/s', 0)
        
        if flow_packets > 1000:
            return f"High volume traffic detected on port {dest_port} with {flow_packets:.1f} packets/second"
        else:
            return f"Network traffic detected on port {dest_port} requiring analysis"
    
    # Default fallback
    return "Network traffic detected requiring security analysis"

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health status."""
    try:
        # Verify models are loaded
        if not hasattr(app.state, "rf_model") or not hasattr(app.state, "lr_model"):
            return {"status": "unhealthy", "reason": "Models not loaded"}
        
        return {
            "status": "healthy",
            "models_loaded": True,
            "api_version": app.version,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}

# Model 1: Network Traffic Analysis Endpoint
@app.post("/predict/traffic", response_model=Model1Response)
async def predict_traffic(features: NetworkTrafficFeatures):
    """
    Analyze network traffic features to detect attacks.
    
    This endpoint uses the Random Forest model to analyze network traffic and
    classify it as benign or a specific attack type.
    """
    try:
        start_time = datetime.now()
        
        # Prepare input features
        raw_features = features.features
        
        # Map feature names if provided
        if features.feature_names:
            # Create a dictionary from provided names and values
            feature_dict = dict(zip(features.feature_names, raw_features))
            
            # If feature order is loaded, reorder features
            if hasattr(app.state, "feature_order"):
                # Create an array with the features in the expected order
                ordered_features = []
                for feature in app.state.feature_order:
                    ordered_features.append(feature_dict.get(feature, 0))
                raw_features = ordered_features
        
        # Convert to numpy array and reshape
        input_array = np.array(raw_features).reshape(1, -1)
        
        # Apply preprocessing
        scaled_features = app.state.scaler.transform(input_array)
        pca_features = app.state.pca.transform(scaled_features)
        
        # Make prediction
        proba = app.state.rf_model.predict_proba(pca_features)[0]
        class_idx = np.argmax(proba)
        predicted_class = app.state.rf_model.classes_[class_idx]
        confidence = float(proba[class_idx])
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Log the prediction
        logger.info(f"Traffic prediction: class={predicted_class}, confidence={confidence:.4f}, time={prediction_time:.4f}s")
        
        # Return results
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "prediction_time": f"{prediction_time:.4f}s"
        }
    
    except Exception as e:
        logger.error(f"Traffic prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Model 2: Alert Text Analysis Endpoint
@app.post("/predict/alert", response_model=Model2Response)
async def predict_alert(alert: AlertMessage):
    """
    Analyze security alert text to determine if it indicates an attack.
    
    This endpoint uses the Logistic Regression model to analyze alert messages
    and determine the probability that the alert indicates a real attack.
    """
    try:
        start_time = datetime.now()
        
        # Clean alert message
        cleaned_message = clean_alert_message(alert.message)
        
        # Vectorize the message
        vectorized_message = app.state.vectorizer.transform([cleaned_message])
        
        # Get prediction probability
        attack_probability = float(app.state.lr_model.predict_proba(vectorized_message)[0][1])
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Log the prediction
        logger.info(f"Alert prediction: probability={attack_probability:.4f}, time={prediction_time:.4f}s, message='{cleaned_message}'")
        
        # Return results
        return {
            "attack_probability": attack_probability,
            "prediction_time": f"{prediction_time:.4f}s"
        }
    
    except Exception as e:
        logger.error(f"Alert prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Fusion Model: Combined Analysis Endpoint
@app.post("/predict/raw", response_model=FusionResponse)
async def predict_fusion(input_data: CombinedInput):
    """
    Combined analysis using both traffic data and alert text.
    
    This endpoint uses the fusion model to combine predictions from both
    the network traffic analysis and alert text analysis for a more
    accurate threat assessment.
    
    You can provide network traffic data using either 'traffic_features' or 'network_data' field.
    """
    try:
        start_time = datetime.now()
        
        # Log what we received for debugging
        logger.info(f"Input data: traffic_features={input_data.traffic_features is not None}, network_data={input_data.network_data is not None}")
        
        # Extract inputs - traffic_features will be set from network_data if needed (via validator)
        traffic_features = input_data.traffic_features
        
        # Log what we're using
        logger.info(f"Using traffic features of type {type(traffic_features).__name__}, length={len(traffic_features)}")
        
        # Generate alert message if none provided
        alert_message = input_data.alert_message
        if not alert_message:
            alert_message = generate_alert_message(traffic_features)
            logger.info(f"Generated alert message: '{alert_message}'")
        
        # Use fusion model for prediction
        try:
            # Try using the predict_threat method directly
            result = app.state.fusion_model.predict_threat(traffic_features, alert_message)
        except Exception as fusion_error:
            logger.error(f"Error using fusion model directly: {fusion_error}")
            logger.error(f"Input types: traffic_features={type(traffic_features)}, alert_message={type(alert_message)}")
            
            # Fallback: Manual fusion of predictions
            # Get prediction from Model 1
            model1_result = await predict_traffic(NetworkTrafficFeatures(features=traffic_features))
            predicted_class = model1_result["predicted_class"]
            model1_conf = model1_result["confidence"]
            
            # Get prediction from Model 2
            model2_result = await predict_alert(AlertMessage(message=alert_message))
            model2_conf = model2_result["attack_probability"]
            
            # Calculate model1 threat contribution based on class
            model1_weight = getattr(app.state.fusion_model, 'MODEL1_WEIGHT', 0.7)
            model2_weight = getattr(app.state.fusion_model, 'MODEL2_WEIGHT', 0.3)
            
            if predicted_class == "BENIGN":
                if model1_conf > 0.8:
                    model1_threat = 0.0  # Confident it's benign, no threat
                else:
                    model1_threat = 1.0 - model1_conf  # Low confidence in benign → moderate threat
            else:
                model1_threat = model1_conf  # Attack predicted → threat = confidence
                
            # Calculate combined threat score
            threat_score = (model1_weight * model1_threat + model2_weight * model2_conf)*100
            
            # Determine recommended action
            thresholds = getattr(app.state.fusion_model, 'THRESHOLDS', {
                'ignore': (0.0, 0.4),
                'monitor': (0.4, 0.7),
                'alert': (0.7, 1.0)
            })
            
            if threat_score >= thresholds['alert'][0]:
                recommended_action = 'Alert'
            elif threat_score >= thresholds['monitor'][0]:
                recommended_action = 'Monitor'
            else:
                recommended_action = 'Ignore'
            
            # Create result dictionary
            result = {
                "predicted_class": predicted_class,
                "threat_score": round(float(threat_score), 2),
                "recommended_action": recommended_action,
                "model1_confidence": round(float(model1_conf), 2),
                "model2_confidence": round(float(model2_conf), 2),
                "model1_weight": model1_weight,
                "model2_weight": model2_weight
            }
        
        # Add prediction time
        prediction_time = (datetime.now() - start_time).total_seconds()
        result["prediction_time"] = f"{prediction_time:.4f}s"
        
        # Log the prediction
        logger.info(f"Fusion prediction: class={result['predicted_class']}, score={result['threat_score']:.4f}, action={result['recommended_action']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Fusion prediction error: {e}")
        logger.error(f"Input data: {input_data}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Get model info endpoint
@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded models."""
    try:
        # Collect model information
        model_info = {
            "model1": {
                "type": "Random Forest Classifier",
                "classes": list(app.state.rf_model.classes_),
                "n_estimators": app.state.rf_model.n_estimators,
                "max_depth": app.state.rf_model.max_depth,
                "feature_count": app.state.pca.n_components_,
                "feature_count_pre_pca": len(app.state.scaler.mean_)
            },
            "model2": {
                "type": "Logistic Regression",
                "feature_count": len(app.state.vectorizer.get_feature_names_out()),
                "classes": ["BENIGN", "ATTACK"]
            },
            "fusion": {
                "model1_weight": getattr(app.state.fusion_model, 'MODEL1_WEIGHT', 0.7),
                "model2_weight": getattr(app.state.fusion_model, 'MODEL2_WEIGHT', 0.3),
                "thresholds": getattr(app.state.fusion_model, 'THRESHOLDS', {
                    'ignore': (0.0, 0.4),
                    'monitor': (0.4, 0.7),
                    'alert': (0.7, 1.0)
                })
            }
        }
        
        return {
            "models": model_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# Examples endpoint 
@app.get("/examples")
async def get_examples():
    """Get example inputs for testing the API."""
    # DDoS example
    ddos_example = {
        "traffic_features": [
            80, 1293792, 3, 7, 26, 11607, 20, 0, 8.67, 10.26, 5840, 0, 1658.14, 2137.30, 
            8991.40, 7.73, 143754.67, 430865.81, 1292730, 2, 747, 373.5, 523.97, 744, 3, 
            1293746, 215624.33, 527671.93, 1292730, 2, 0, 0, 72, 152, 2.32, 5.41, 0, 5840, 
            1057.55, 1853.44, 3435230.67, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1163.3, 8.67, 1658.14, 
            72, 3, 26, 7, 11607, 8192, 229, 2, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
        ],
        "alert_message": "Critical: High volume traffic flood detected targeting web server on port 80 from multiple source IPs"
    }
    
    # Port scan example
    port_scan_example = {
        "traffic_features": [
            80, 5185118, 7, 7, 1022, 2321, 372, 0, 146.0, 184.08, 1047, 0, 331.57, 439.66, 
            644.73, 2.70, 398855.23, 1372180.71, 4963956, 4, 221162, 36860.33, 56141.02, 
            141434, 4, 5185004, 864167.33, 2027593.31, 5001548, 879, 0, 0, 232, 232, 1.35, 
            1.35, 0, 1047, 222.87, 331.32, 109775.55, 0, 0, 0, 1, 0, 0, 0, 0, 1, 238.79, 
            146.0, 331.57, 232, 7, 1022, 7, 2321, 29200, 252, 3, 32, 0.0, 0.0, 0, 0, 0.0, 
            0.0, 0, 0
        ],
        "alert_message": "ALERT: PortScan from 172.16.0.1 to 192.168.10.50 on port 80 using protocol TCP"
    }
    
    # Benign example
    benign_example = {
        "traffic_features": [
            443, 2500, 10, 8, 2048, 1024, 1500, 64, 800, 200, 1200, 48, 600, 150, 
            1200.0, 10.0, 100.0, 50.0, 200.0, 10.0, 1000.0, 100.0, 50.0, 200.0, 10.0,
            800.0, 100.0, 40.0, 150.0, 20.0, 0, 0, 200, 160, 5.0, 4.0, 48, 1500, 
            700.0, 300.0, 90000.0, 0, 1, 0, 0, 5, 0, 0, 0, 0.8, 750.0, 800.0, 600.0,
            200, 10, 2048, 8, 1024, 8192, 8192, 4, 32, 100.0, 50.0, 200.0, 10.0, 
            500.0, 200.0, 1000.0, 50.0
        ],
        "alert_message": "ALERT: BENIGN from 192.168.10.15 to 104.25.162.101 on port 80 using protocol TCP"
    }
    
    # Alternative format examples (using network_data field)
    alternative_format_examples = [
        {
            "network_data": [
                80, 1293792, 3, 7, 26, 11607, 20, 0, 8.67, 10.26, 5840, 0, 1658.14, 2137.30, 
                8991.40, 7.73, 143754.67, 430865.81, 1292730, 2, 747, 373.5, 523.97, 744, 3, 
                1293746, 215624.33, 527671.93, 1292730, 2, 0, 0, 72, 152, 2.32, 5.41, 0, 5840, 
                1057.55, 1853.44, 3435230.67, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1163.3, 8.67, 1658.14, 
                72, 3, 2672, 3, 26, 7, 11607, 8192, 229, 2, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
            ],
            "alert_message": "Critical: High volume traffic flood detected targeting web server on port 80 from multiple source IPs"
        }
    ]
    
    # Example for Model 2 only
    alert_only_examples = [
        {"message": "ALERT: BENIGN from 192.168.1.1 to 10.0.0.1 using protocol 6 on port 80"},
        {"message": "ALERT: Potential SQL injection attack detected from 45.62.118.34 on port 443"},
        {"message": "ALERT: DoS attack detected, high traffic volume from 108.61.128.15"},
        {"message": "ALERT: Suspicious login attempt detected from unknown source"}
    ]
    
    return {
        "fusion_examples": [
            ddos_example,
            port_scan_example,
            benign_example
        ],
        "alternative_format_examples": alternative_format_examples,
        "alert_examples": alert_only_examples,
        "usage_notes": "Use these examples with their respective endpoints: /predict/raw, /predict/alert.\n" +
                      "Both 'traffic_features' and 'network_data' field names are supported for compatibility."
    }

# Main entry point
if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)