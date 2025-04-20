
import re
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('models/logistic_regression_model.joblib')
vectorizer = joblib.load('models/logistic_regression_vectorizer.joblib')

def clean_alert_message(text):
    """
    Clean alert message by:
    - Removing IP addresses (completely)
    - Removing port numbers (completely)
    - Removing protocol numbers
    - Converting to lowercase
    - Removing extra spaces
    """
    # Convert to string if not already
    text = str(text)
    
    # Remove IP addresses (completely remove them)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
    
    # Remove port numbers (completely remove them)
    text = re.sub(r'\b(?:on|to|from|at|using)\s+port\s+\d+\b', ' ', text)
    text = re.sub(r'\bport\s+\d+\b', '', text)
    
    # Remove protocol numbers
    text = re.sub(r'\bprotocol\s+\d+\b', 'protocol', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_alert(alert_text):
    """
    Predict if an alert message indicates an attack
    """
    # Clean the message
    cleaned_text = clean_alert_message(alert_text)
    
    # Transform using the vectorizer
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    pred_prob = model.predict_proba(text_tfidf)[0, 1]
    pred_class = model.predict(text_tfidf)[0]
    label = "ATTACK" if pred_class == 1 else "BENIGN"
    
    # Enhanced explanation based on probability
    if pred_prob > 0.95:
        confidence = "Very high confidence"
    elif pred_prob > 0.8:
        confidence = "High confidence"
    elif pred_prob > 0.6:
        confidence = "Moderate confidence"
    else:
        confidence = "Low confidence"
    
    return {
        'prediction': label,
        'probability': pred_prob,
        'confidence': confidence,
        'class': int(pred_class)
    }

# Example usage
if __name__ == "__main__":
    # Test with example alerts
    print("\nTesting prediction function with example alerts:\n")
    
    test_alerts = [
        "ALERT: BENIGN from 192.168.1.1 to 10.0.0.1 using protocol 6 on port 80",
        "ALERT: Potential SQL injection attack detected from 45.62.118.34 on port 443",
        "ALERT: DoS attack detected, high traffic volume from 108.61.128.15",
        "ALERT: Suspicious login attempt detected from unknown source"
    ]
    
    for alert in test_alerts:
        result = predict_alert(alert)
        print(f"Alert: {alert}")
        print(f"Prediction: {result['prediction']} ({result['confidence']})")
        print(f"Probability: {result['probability']:.4f}")
        print("---")
