import joblib
import re

# Load the saved model and vectorizer
model = joblib.load('models/logistic_regression_model.joblib')
vectorizer = joblib.load('models/logistic_regression_vectorizer.joblib')

# Function to predict the alert
def predict_alert(alert_text):
    """
    Predict if an alert message indicates an attack or benign.
    Assumes the input text is already in the correct format.
    """
    # Transform using the vectorizer (no cleaning applied here)
    text_tfidf = vectorizer.transform([alert_text])
    
    # Make prediction
    pred_prob = model.predict_proba(text_tfidf)[0, 1]  # Probability of the alert being an attack
    pred_class = model.predict(text_tfidf)[0]  # Predicted class (0 = BENIGN, 1 = ATTACK)
    label = "ATTACK" if pred_class == 1 else "BENIGN"  # Convert class to label
    
    return {
        'prediction': label,
        'probability': pred_prob,
        'class': int(pred_class)
    }

# List of alert messages to test
alerts = [
    "ALERT: Web Attack  Brute Force from 172.16.0.1 to 192.168.10.50 on port 80.0 using protocol UNKNOWN",
    "ALERT: PortScan from 172.16.0.1 to 192.168.10.50 on port 80 using protocol TCP",
    "ALERT: BENIGN from 192.168.10.15 to 104.25.162.101 on port 80 using protocol TCP",
    "ALERT: BENIGN from 151.101.20.64 to 192.168.10.5 on port 50914.0 using protocol UNKNOWN",
    "ALERT: Web Attack  Sql Injection from 172.16.0.1 to 192.168.10.50 on port 80.0 using protocol UNKNOWN",
    "ALERT: Web Attack  XSS from 172.16.0.1 to 192.168.10.50 on port 80.0 using protocol UNKNOWN",
    "ALERT: SSH-Patator from 172.16.0.1 to 192.168.10.50 on port 22 using protocol TCP",
    "ALERT: SSH-Patator from 172.16.0.1 to 192.168.10.50 on port 22 using protocol TCP"
]

# Test each alert
for alert in alerts:
    result = predict_alert(alert)
    print(f"Alert: {alert}")
    print(f"Prediction: {result['prediction']} (Probability: {result['probability']:.4f})")
    print("---")
