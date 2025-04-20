"""
Modified fusion.py with feature mapping functionality - Corrected for 70 features

This adds a feature mapping function to properly align raw input lists with the expected 70 features.
"""

import joblib
import numpy as np
import pandas as pd
import re
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Completely suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define paths to model artifacts
# This points to the directory containing fusion.py and all the model files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Model 1: Structured Traffic Classifier
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model2_cicids.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler.joblib')
PCA_PATH = os.path.join(MODEL_DIR, 'incremental_pca.joblib')
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, 'model1_feature_order.joblib')

# Model 2: Alert Text Logistic Regression
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'logistic_regression_vectorizer.joblib')

# Action thresholds
THRESHOLDS = {
    'ignore': (0.0, 0.4),
    'monitor': (0.4, 0.7),
    'alert': (0.7, 1.0)
}

# Model fusion weights
MODEL1_WEIGHT = 0.7
MODEL2_WEIGHT = 0.3

def map_input_to_feature_names(raw_input_list):
    """
    Maps raw input list values to their corresponding feature names.
    
    Args:
        raw_input_list: List of raw input values
        
    Returns:
        dict: Dictionary with feature names as keys and values from raw_input_list
    """
    # EXACTLY 70 standard feature names in the order they appear in raw input lists
    # These match the 70 features expected by your model
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
    
    # Verify we have exactly 70 features
    assert len(standard_feature_names) == 70, f"Feature list contains {len(standard_feature_names)} features, but expected 70"
    
    # Handle case where raw_input_list is longer than standard_feature_names
    if len(raw_input_list) > len(standard_feature_names):
        print(f"Warning: Input has {len(raw_input_list)} values, but only {len(standard_feature_names)} features used by the model.")
        raw_input_list = raw_input_list[:len(standard_feature_names)]
    # Handle case where raw_input_list is shorter than standard_feature_names
    elif len(raw_input_list) < len(standard_feature_names):
        print(f"Warning: Input has only {len(raw_input_list)} values, but {len(standard_feature_names)} features expected.")
        # Pad with zeros
        raw_input_list = raw_input_list + [0] * (len(standard_feature_names) - len(raw_input_list))
    
    # Create a dictionary mapping feature names to values
    return dict(zip(standard_feature_names, raw_input_list))

class FusionModel:
    """
    A fusion model that combines predictions from a structured traffic classifier
    and an alert text classifier to produce a unified threat score.
    """
    
   
    def __init__(self, auto_load=True):
        if auto_load:
            self.load_models()
        
    def load_models(self):
        """Load all model artifacts and preprocessors."""
        
        # Load Model 1 (Structured Traffic) preprocessors and model
        self.scaler = joblib.load(SCALER_PATH)
        self.pca = joblib.load(PCA_PATH)
        self.rf_model = joblib.load(RF_MODEL_PATH)
        
        # Load feature order for proper alignment
        try:
            self.feature_order = joblib.load(FEATURE_ORDER_PATH)
            print(f"Loaded feature order with {len(self.feature_order)} features")
        except Exception as e:
            print(f"Warning: Could not load feature order: {e}")
            self.feature_order = None
        
        # Load Model 2 (Alert Text) preprocessor and model
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.lr_model = joblib.load(LR_MODEL_PATH)
        
        # Get class labels from the random forest model
        self.class_labels = self.rf_model.classes_
        
    def preprocess_structured_input(self, structured_input):
        """
        Preprocess structured input by scaling and applying PCA.
        
        Args:
            structured_input: Raw feature array (pre-PCA) or dictionary
            
        Returns:
            Preprocessed input ready for the random forest model
        """
        try:
            # Convert to numpy array if it's a list
            if isinstance(structured_input, list):
                # Map the raw list to a dictionary with feature names
                structured_input = map_input_to_feature_names(structured_input)
                
                # Now handle it as a dictionary
                if self.feature_order is not None:
                    # Create ordered feature array
                    feature_array = []
                    for feature in self.feature_order:
                        if feature in structured_input:
                            feature_array.append(structured_input[feature])
                        else:
                            # Feature not found, use 0 as default
                            feature_array.append(0)
                    structured_input = np.array(feature_array).reshape(1, -1)
                else:
                    # No feature order available, use values as-is
                    structured_input = np.array(list(structured_input.values())).reshape(1, -1)
            elif isinstance(structured_input, np.ndarray):
                if len(structured_input.shape) == 1:
                    # 1D array, convert to list then to dictionary
                    structured_input = map_input_to_feature_names(structured_input.tolist())
                    
                    # Now handle it as a dictionary
                    if self.feature_order is not None:
                        # Create ordered feature array
                        feature_array = []
                        for feature in self.feature_order:
                            if feature in structured_input:
                                feature_array.append(structured_input[feature])
                            else:
                                # Feature not found, use 0 as default
                                feature_array.append(0)
                        structured_input = np.array(feature_array).reshape(1, -1)
                    else:
                        # No feature order available, use values as-is
                        structured_input = np.array(list(structured_input.values())).reshape(1, -1)
                else:
                    # Already a 2D array, reshape if needed
                    structured_input = structured_input.reshape(1, -1) if structured_input.shape[0] != 1 else structured_input
            elif isinstance(structured_input, dict):
                # Handle dictionary input - align with expected feature order
                if self.feature_order is not None:
                    # Create ordered feature array
                    feature_array = []
                    for feature in self.feature_order:
                        if feature in structured_input:
                            feature_array.append(structured_input[feature])
                        else:
                            # Feature not found, use 0 as default
                            feature_array.append(0)
                    structured_input = np.array(feature_array).reshape(1, -1)
                else:
                    # No feature order available, use values as-is
                    structured_input = np.array(list(structured_input.values())).reshape(1, -1)
            elif isinstance(structured_input, pd.DataFrame):
                # Handle DataFrame input
                if self.feature_order is not None:
                    # Align columns with expected feature order
                    aligned_df = pd.DataFrame(index=structured_input.index)
                    for feature in self.feature_order:
                        if feature in structured_input.columns:
                            aligned_df[feature] = structured_input[feature]
                        else:
                            # Feature not found, use 0 as default
                            aligned_df[feature] = 0
                    structured_input = aligned_df.values
                else:
                    structured_input = structured_input.values
            
            # Check if input shape matches expected shape
            expected_shape = self.scaler.mean_.shape[0]
            if structured_input.shape[1] != expected_shape:
                print(f"Warning: Input has {structured_input.shape[1]} features, but scaler expects {expected_shape}")
                
                # Adjust the size if needed
                if structured_input.shape[1] > expected_shape:
                    print("Truncating excess features")
                    structured_input = structured_input[:, :expected_shape]
                else:
                    print("Padding missing features with zeros")
                    padding = np.zeros((structured_input.shape[0], expected_shape - structured_input.shape[1]))
                    structured_input = np.hstack((structured_input, padding))
            
            # Scale the features
            scaled_input = self.scaler.transform(structured_input)
            
            # Apply PCA
            pca_input = self.pca.transform(scaled_input)
            
            return pca_input
            
        except Exception as e:
            print(f"Error preprocessing structured input: {e}")
            # Fallback method for handling preprocessing errors
            try:
                # Get expected feature count from scaler
                expected_feature_count = len(self.scaler.mean_)
                
                # Ensure input is correct shape
                if isinstance(structured_input, (list, np.ndarray)):
                    if isinstance(structured_input, list):
                        # Convert list to dictionary first for better feature alignment
                        feature_dict = map_input_to_feature_names(structured_input)
                        # Extract values in the right order if feature order is available
                        if self.feature_order is not None:
                            input_array = np.array([feature_dict.get(feature, 0) for feature in self.feature_order])
                        else:
                            input_array = np.array(list(feature_dict.values()))
                    else:
                        input_array = np.array(structured_input).flatten()
                elif isinstance(structured_input, dict):
                    # Extract values in the right order if feature order is available
                    if self.feature_order is not None:
                        input_array = np.array([structured_input.get(feature, 0) for feature in self.feature_order])
                    else:
                        input_array = np.array(list(structured_input.values())).flatten()
                elif isinstance(structured_input, pd.DataFrame):
                    input_array = structured_input.values.flatten()
                else:
                    raise ValueError(f"Unsupported input type: {type(structured_input)}")
                
                # Adjust array size if needed
                if len(input_array) > expected_feature_count:
                    input_array = input_array[:expected_feature_count]
                elif len(input_array) < expected_feature_count:
                    # Pad with zeros
                    padding = np.zeros(expected_feature_count - len(input_array))
                    input_array = np.concatenate([input_array, padding])
                
                # Reshape and manually scale
                input_array = input_array.reshape(1, -1)
                scaled_input = (input_array - self.scaler.mean_) / self.scaler.scale_
                
                # Apply PCA
                pca_input = self.pca.transform(scaled_input)
                return pca_input
                
            except Exception as fallback_error:
                print(f"Fallback preprocessing also failed: {fallback_error}")
                # Last resort: create a zero array of the right shape for the PCA
                pca_components = self.pca.n_components_
                return np.zeros((1, pca_components))
    
    def clean_alert_message(self, text):
        """
        Clean alert message by:
        - Removing IP addresses
        - Removing port numbers
        - Removing protocol numbers
        - Converting to lowercase
        - Removing extra spaces
        """
        # Convert to string if not already
        text = str(text)
        
        # Remove IP addresses
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
        
        # Remove port numbers
        text = re.sub(r'\b(?:on|to|from|at|using)\s+port\s+\d+\b', ' ', text)
        text = re.sub(r'\bport\s+\d+\b', '', text)
        text = re.sub(r':\d+', '', text)  # Remove port after colon
        
        # Remove protocol numbers
        text = re.sub(r'\bprotocol\s+\d+\b', 'protocol', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_alert_message(self, alert_message):
        """
        Preprocess alert message text using TF-IDF vectorization.
        
        Args:
            alert_message: Text string of the alert message
            
        Returns:
            Vectorized text ready for the logistic regression model
        """
        # Clean the alert message first
        cleaned_message = self.clean_alert_message(alert_message)
        
        # Vectorize the alert message
        vectorized_alert = self.vectorizer.transform([cleaned_message])
        
        return vectorized_alert
    
    def predict_model1(self, structured_input):
        """
        Get prediction from Model 1 (Structured Traffic Classifier).
        
        Args:
            structured_input: Raw feature array
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        # Preprocess the structured input
        preprocessed_input = self.preprocess_structured_input(structured_input)
        
        # Get probability predictions
        proba = self.rf_model.predict_proba(preprocessed_input)[0]
        
        # Get the class with highest probability
        class_idx = np.argmax(proba)
        predicted_class = self.class_labels[class_idx]
        
        # Get the confidence (max probability)
        confidence = np.max(proba)
        
        # Logical correction: If predicted BENIGN with confidence < 0.5,
        # it should be classified as an attack
        if predicted_class == "BENIGN" and confidence < 0.5:
            # Find the attack class with highest probability
            attack_indices = [i for i, label in enumerate(self.class_labels) if label != "BENIGN"]
            if attack_indices:
                attack_probs = [proba[i] for i in attack_indices]
                max_attack_idx = attack_indices[np.argmax(attack_probs)]
                predicted_class = self.class_labels[max_attack_idx]
                confidence = proba[max_attack_idx]
            else:
                # If no attack classes, use a generic "ATTACK" classification
                predicted_class = "ATTACK"
                confidence = 1.0 - confidence  # Use the inverse of BENIGN confidence
        
        return predicted_class, confidence
    
    def predict_model2(self, alert_message):
        """
        Get prediction from Model 2 (Alert Text Classifier).
        
        Args:
            alert_message: Text string of the alert message
            
        Returns:
            float: Probability of attack (0-1)
        """
        # Preprocess the alert message
        vectorized_alert = self.preprocess_alert_message(alert_message)
        
        # Get probability of attack (class 1)
        attack_prob = self.lr_model.predict_proba(vectorized_alert)[0][1]
        
        return attack_prob
    
    def get_recommended_action(self, threat_score):
        """
        Determine the recommended action based on the threat score.
        
        Args:
            threat_score: Combined threat score (0-1)
            
        Returns:
            str: Recommended action ('Ignore', 'Monitor', or 'Alert')
        """
        if threat_score >= THRESHOLDS['alert'][0]:
            return 'Alert'
        elif threat_score >= THRESHOLDS['monitor'][0]:
            return 'Monitor'
        else:
            return 'Ignore'
    
    def predict_threat(self, structured_input, alert_message):
        """
        Generate a combined threat prediction using both models.
        
        Args:
            structured_input: Raw feature array for Model 1
            alert_message: Text string for Model 2
            
        Returns:
            dict: Prediction results including predicted class, threat score, and recommended action
        """
        # Get Model 1 prediction
        predicted_class, model1_conf = self.predict_model1(structured_input)
        
        # Get Model 2 prediction
        model2_conf = self.predict_model2(alert_message)
        
        # Calculate combined threat score based on class prediction
        # Conservative logic to reduce false positives
        if predicted_class == "BENIGN":
               if model1_conf > 0.8:
                  model1_threat = 0.0  # Confident it's benign, no threat
               else:
                  model1_threat = 1.0 - model1_conf  # Low confidence in benign → moderate threat
        else:
           model1_threat = model1_conf  # Attack predicted → threat = confidence

        # Fusion score using both models
        threat_score = (MODEL1_WEIGHT * model1_threat + MODEL2_WEIGHT * model2_conf)*100
        
        # Get recommended action
        recommended_action = self.get_recommended_action(threat_score)
        
        # Format the output with more detail
        result = {
            "predicted_class": predicted_class,
            "threat_score": round(float(threat_score), 2),
            "recommended_action": recommended_action,
            "model1_confidence": round(float(model1_conf), 2),
            "model2_confidence": round(float(model2_conf), 2),
            "model1_weight": MODEL1_WEIGHT,
            "model2_weight": MODEL2_WEIGHT
        }
        
        return result


def generate_alert_message(structured_data):
    """
    Generate a synthetic alert message from structured network data.
    
    Args:
        structured_data: Dict or array of network flow features
        
    Returns:
        str: Generated alert message
    """
    # Convert to dictionary if array
    if isinstance(structured_data, (list, np.ndarray)):
        # Convert list to dictionary with feature names for better message generation
        if isinstance(structured_data, np.ndarray):
            structured_data = structured_data.tolist()
        
        # Map to feature dictionary
        structured_data = map_input_to_feature_names(structured_data)
        
        # Try to extract useful fields for alert generation
        dest_port = structured_data.get('Destination Port', 0)
        flow_packets = structured_data.get('Flow Packets/s', 0)
        
        if flow_packets > 1000:
            return f"High volume traffic detected on port {dest_port} with {flow_packets:.1f} packets/second"
        else:
            return f"Network traffic detected on port {dest_port} requiring analysis"
    
    # If already a dictionary, try to extract relevant fields
    if isinstance(structured_data, dict):
        # Example simple implementation
        source_ip = structured_data.get('Source IP', '192.168.1.1')
        dest_ip = structured_data.get('Destination IP', '10.0.0.1')
        dest_port = structured_data.get('Destination Port', 80)
        
        # Very simple template-based generation
        return f"Traffic detected from {source_ip} to {dest_ip} on port {dest_port}"
    
    # Default fallback
    return "Network traffic detected requiring security analysis"


def predict_threat(structured_input, alert_message=None):
    """
    Convenience function to make predictions using the fusion model.
    
    Args:
        structured_input: Raw feature array, dictionary, or DataFrame for the structured traffic classifier
                        Should contain the network traffic features
        alert_message: Optional text string for the alert text classifier. If None, generates one.
        
    Returns:
        dict: Prediction results
    """
    # Initialize the fusion model (this will load all required artifacts)
    model = FusionModel()
    
    # Generate an alert message if none is provided
    if alert_message is None:
        alert_message = generate_alert_message(structured_input)
    
    # Make prediction
    return model.predict_threat(structured_input, alert_message)


def save_fusion_model(model, filename):
    """
    Save the fusion model to a file in the same directory as the other models.
    
    Args:
        model: The FusionModel instance to save
        filename: Name of the file to save the model to
    """
    # Use the same MODEL_DIR defined for loading the models
    save_path = os.path.join(MODEL_DIR, filename)
    
    # Save the model using joblib
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("MULTIMODEL FUSION CYBERSECURITY DETECTION SYSTEM")
    print("="*50)
    
    # Example 1: DDoS attack
    # This example has high traffic values typical of a DDoS attack
    real_ddos_example = [
     80,306157,3,6,26,11607,20,0,8.666666667,10.26320288,5840,0,1934.5,2538.919278,37996.84476,29.39668209,38269.625,107926.2565,305373,3,649,324.5,432.0422433,630,19,306140,61228.0,136481.48,305373,3,0,0,72,132,9.798894031,19.59778806,0,5840,1163.3,2138.329153,4572451.567,0,0,0,1,0,0,0,0,2,1292.555556,8.666666667,1934.5,72,3,26,6,11607,8192,229,2,20,0.0,0.0,0,0,0.0,0.0,0,0
      ]
    
    # Define sample alert message for DDoS
    sample_alert_message_ddos = "Critical: High volume traffic flood detected targeting web server on port 80 from multiple source IPs"
    
    print("\nRunning prediction for DDoS attack example...")
    # Make prediction using the function
    result = predict_threat(real_ddos_example, sample_alert_message_ddos)
    
    # Print the result in a nicely formatted way
    print("\nPrediction Result:")
    print(f"  - Detected Class: {result['predicted_class']}")
    print(f"  - Threat Score: {result['threat_score']}")
    print(f"  - Recommended Action: {result['recommended_action']}")
    
    print("\nDetailed fusion analysis:")
    # Initialize the model for detailed analysis
    model = FusionModel()
    predicted_class, model1_conf = model.predict_model1(real_ddos_example)
    model2_conf = model.predict_model2(sample_alert_message_ddos)
    
    # Calculate model1 threat contribution based on class
    if predicted_class == "BENIGN":
        model1_threat = 1.0 - model1_conf
        model1_explanation = "1.0 - confidence (inverse for BENIGN)"
    else:
        model1_threat = model1_conf
        model1_explanation = "confidence (direct for ATTACK)"
    
    # Print detailed analysis
    print(f"  - Model 1 (Traffic Analysis): {predicted_class} with {model1_conf:.2f} confidence")
    print(f"  - Model 1 Threat Contribution: {model1_threat:.2f} ({model1_explanation})")
    print(f"  - Model 2 (Text Analysis): Attack probability {model2_conf:.2f}")
    
    combined_score = MODEL1_WEIGHT * model1_threat + MODEL2_WEIGHT * model2_conf
    print(f"  - Final Threat Score: {combined_score:.2f} = ({model1_threat:.2f} × {MODEL1_WEIGHT}) + ({model2_conf:.2f} × {MODEL2_WEIGHT})")
    print("="*50)
    
    # Example 2: Port scan
    print("\nComparison with simulated port scan example:")
    
    # Create a feature array for a port scan - exactly 70 features
    port_scan_features = [80,5185118,7,7,1022,2321,372,0,146.0,184.0787875,1047,0,331.5714286,439.6592837,644.7297824,2.700034985,398855.2308,1372180.71,4963956,4,221162,36860.33333,56141.02125,141434,4,5185004,864167.3333,2027593.314,5001548,879,0,0,232,232,1.350017492,1.350017492,0,1047,222.8666667,331.3239387,109775.5524,0,0,0,1,0,0,0,0,1,238.7857143,146.0,331.5714286,232,7,1022,7,2321,29200,252,3,32,0.0,0.0,0,0,0.0,0.0,0,0]  # Using 70 features to match model expectations

    sample_alert_message_scan = "ALERT: PortScan from 172.16.0.1 to 192.168.10.50 on port 80 using protocol TCP"
    
    # Run prediction for port scan
    result_scan = predict_threat(port_scan_features, sample_alert_message_scan)
    predicted_class_scan, model1_conf_scan = model.predict_model1(port_scan_features)
    model2_conf_scan = model.predict_model2(sample_alert_message_scan)
    
    # Calculate model1 threat contribution for port scan
    if predicted_class_scan == "BENIGN":
        model1_threat_scan = 1.0 - model1_conf_scan
        model1_explanation_scan = "1.0 - confidence (inverse for BENIGN)"
    else:
        model1_threat_scan = model1_conf_scan
        model1_explanation_scan = "confidence (direct for ATTACK)"
    
    combined_score_scan = (MODEL1_WEIGHT * model1_threat_scan + MODEL2_WEIGHT * model2_conf_scan)*100
    
    print("\nPort Scan Example Results:")
    print(f"  - Model 1 (Traffic Analysis): {predicted_class_scan} with {model1_conf_scan:.2f} confidence")
    print(f"  - Final Threat Score: {combined_score_scan:.2f}")
    print(f"  - Recommended Action: {result_scan['recommended_action']}")
    print("="*50)
    
    # Example 3: Benign traffic
    print("\nBenign traffic example:")
    
    # Create a feature array for benign traffic - exactly 70 features
    benign_features = [0.0] * 70  # Using 70 features to match model expectations
    # Set typical benign traffic indicators
    benign_features[0] = 443    # HTTPS port
    benign_features[2] = 10     # Normal packet count
    benign_features[3] = 8      # Normal response packet count
    benign_features[16] = 10    # Normal packet rate
    benign_features[48] = 5     # Normal ACK count
    
    sample_alert_message_benign = "ALERT: BENIGN from 192.168.10.15 to 104.25.162.101 on port 80 using protocol TCP"
    
    # Run prediction for benign traffic
    result_benign = predict_threat(benign_features, sample_alert_message_benign)
    
    print("\nBenign Traffic Example Results:")
    print(f"  - Detected Class: {result_benign['predicted_class']}")
    print(f"  - Threat Score: {result_benign['threat_score']}")
    print(f"  - Recommended Action: {result_benign['recommended_action']}")
    print("="*50)
    
    # Show feature mapping in action
    print("\nFeature Mapping Demonstration:")
    raw_features = [443, 1205, 25, 15, 4096, 2048, 1500, 128, 900, 350]
    feature_dict = map_input_to_feature_names(raw_features)
    print("Raw input:", raw_features[:5], "...")
    print("Mapped to features:")
    for i, (key, value) in enumerate(list(feature_dict.items())[:5]):
        print(f"  {i+1}. {key}: {value}")
    print("="*50)
    
    # Verify the feature count
    print("\nFeature Count Verification:")
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
    print(f"  - Standard feature names count: {len(standard_feature_names)}")
    print(f"  - Features expected by model: 70")
    print(f"  - Raw DDoS example length: {len(real_ddos_example)}")
    
    # Display feature mapping for DDoS example
    mapped_ddos = map_input_to_feature_names(real_ddos_example)
    print(f"  - After mapping, DDoS features: {len(mapped_ddos)}")
    
    # Save the fusion model
    print("\nSaving fusion model...")
    save_fusion_model(model, 'fusion_model.joblib')