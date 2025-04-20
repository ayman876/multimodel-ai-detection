import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import joblib
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load and clean the dataset
print("Loading and cleaning the dataset...")
start_time = time()

# Load dataset
df = pd.read_csv('./data/alert_dataset.csv')
print(f"Dataset shape: {df.shape}")
print("Sample data:")
print(df.head())

# Check class distribution
class_distribution = df['binary_label'].value_counts(normalize=True) * 100
print("\nClass Distribution:")
for label, percentage in class_distribution.items():
    label_name = "ATTACK" if label == 1 else "BENIGN"
    print(f"  {label_name}: {percentage:.2f}%")

# Clean the alert messages
def clean_alert_message(text):
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

# Apply cleaning
df['cleaned_message'] = df['alert_message'].apply(clean_alert_message)

# 2. Add column for message length
df['message_length'] = df['cleaned_message'].apply(len)

print("\nSample cleaned data:")
print(df[['alert_message', 'cleaned_message', 'message_length', 'binary_label']].head())
print(f"Data cleaning completed in {time() - start_time:.2f} seconds")

# 3. TF-IDF Vectorization
print("\nPerforming TF-IDF vectorization...")
start_time = time()

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words='english'  # Use built-in English stop words
)

# Prepare data for model training
X = df['cleaned_message']
y = df['binary_label']

# Fit the vectorizer on the entire dataset
X_tfidf = tfidf_vectorizer.fit_transform(X)

print(f"TF-IDF vectorization completed in {time() - start_time:.2f} seconds")
print(f"Number of features: {X_tfidf.shape[1]}")

# Get all features
all_features = tfidf_vectorizer.get_feature_names_out()
print("\nFeature extraction complete. Found", len(all_features), "features.")

# Save all features to CSV
feature_df = pd.DataFrame({'Feature': all_features})
feature_df.to_csv('output/logistic_regression_features.csv', index=False)

# 4. Cross-Validation with Logistic Regression
print("\nPerforming cross-validation with Logistic Regression...")
start_time = time()

# Define logistic regression model with class weight adjustment
# Class weight 'balanced' automatically adjusts weights inversely proportional to class frequencies
log_reg = LogisticRegression(
    C=1.0,                # Regularization strength (inverse)
    solver='liblinear',   # Solver that works well for both small and large datasets
    max_iter=1000,        # Increase max iterations to ensure convergence
    random_state=42,      
    class_weight='balanced', # Handle class imbalance
    verbose=0
)

# Define scorers for cross-validation
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# Stratified k-fold cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Perform cross-validation with multiple metrics
cv_results = cross_validate(
    log_reg, 
    X_tfidf, 
    y, 
    cv=skf, 
    scoring=scorers, 
    return_train_score=True,
    verbose=1
)

# Calculate and print cross-validation results
print("\nCross-Validation Results (Test):")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    test_scores = cv_results[f'test_{metric}']
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    print(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")

# Save cross-validation results
cv_results_df = pd.DataFrame({
    'Fold': list(range(1, n_folds + 1)) * 4,
    'Metric': ['Accuracy'] * n_folds + ['Precision'] * n_folds + ['Recall'] * n_folds + ['F1'] * n_folds,
    'Train Score': np.concatenate([
        cv_results['train_accuracy'],
        cv_results['train_precision'],
        cv_results['train_recall'],
        cv_results['train_f1']
    ]),
    'Test Score': np.concatenate([
        cv_results['test_accuracy'],
        cv_results['test_precision'],
        cv_results['test_recall'],
        cv_results['test_f1']
    ])
})

cv_results_df.to_csv('output/logistic_regression_cv_results.csv', index=False)

# Visualize cross-validation results
plt.figure(figsize=(12, 8))
sns.boxplot(x='Metric', y='Test Score', data=cv_results_df)
plt.title('Cross-Validation Performance Metrics (Logistic Regression)')
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('output/logistic_regression_cv_boxplot.png')
plt.close()

print(f"Cross-validation completed in {time() - start_time:.2f} seconds")

# 5. Train final model on the full dataset
print("\nTraining final logistic regression model...")
start_time = time()

# Split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Train the final model
final_model = LogisticRegression(
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    verbose=0
)

final_model.fit(X_train, y_train)

print(f"Final model training completed in {time() - start_time:.2f} seconds")

# 6. Evaluation on held-out test set
print("\nEvaluating final model on held-out test set...")

# Make predictions
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK'])
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Print evaluation results
print("\nFinal Model Evaluation Results (held-out test set):")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_rep)

# Create evaluation report
with open('output/logistic_regression_evaluation.txt', 'w') as f:
    f.write("CYBERSECURITY ALERT CLASSIFICATION WITH LOGISTIC REGRESSION\n")
    f.write("=====================================================\n\n")
    
    f.write("1. CROSS-VALIDATION RESULTS\n")
    f.write("-------------------------\n")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        test_scores = cv_results[f'test_{metric}']
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        f.write(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}\n")
    
    f.write("\nDetailed scores per fold:\n")
    for fold in range(n_folds):
        f.write(f"\nFold {fold+1}:\n")
        f.write(f"  Accuracy:  Train={cv_results['train_accuracy'][fold]:.4f}, Test={cv_results['test_accuracy'][fold]:.4f}\n")
        f.write(f"  Precision: Train={cv_results['train_precision'][fold]:.4f}, Test={cv_results['test_precision'][fold]:.4f}\n")
        f.write(f"  Recall:    Train={cv_results['train_recall'][fold]:.4f}, Test={cv_results['test_recall'][fold]:.4f}\n")
        f.write(f"  F1:        Train={cv_results['train_f1'][fold]:.4f}, Test={cv_results['test_f1'][fold]:.4f}\n")
    
    f.write("\n\n2. FINAL MODEL EVALUATION (held-out test set)\n")
    f.write("------------------------------------------\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_norm, 
    annot=conf_matrix,
    fmt='d', 
    cmap='Blues',
    xticklabels=['BENIGN', 'ATTACK'],
    yticklabels=['BENIGN', 'ATTACK']
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Logistic Regression)')
plt.savefig('output/logistic_regression_confusion_matrix.png')
plt.close()

# 7. Feature importance analysis
print("\nAnalyzing feature importance (coefficients)...")

# Get coefficients (feature importance for logistic regression)
coefficients = final_model.coef_[0]
feature_importance = np.abs(coefficients)  # Take absolute value for importance ranking

# Create sorted list of features by importance
# FIXED: Using all_features instead of feature_names
features_with_importance = [(all_features[i], coefficients[i], feature_importance[i]) 
                          for i in range(len(all_features))]
features_with_importance.sort(key=lambda x: x[2], reverse=True)  # Sort by absolute importance
top_features = features_with_importance[:20]

# Print top features
print("\nTop 20 Most Important Features (Logistic Regression):")
for i, (feature, coef, importance) in enumerate(top_features, 1):
    direction = "Indicates ATTACK" if coef > 0 else "Indicates BENIGN"
    print(f"{i}. {feature} (coefficient: {coef:.4f}, importance: {importance:.4f}) - {direction}")

# Save top features to report
with open('output/logistic_regression_evaluation.txt', 'a') as f:
    f.write("\n\n3. FEATURE IMPORTANCE (COEFFICIENTS)\n")
    f.write("----------------------------------\n")
    f.write("Top 20 Most Important Features:\n")
    for i, (feature, coef, importance) in enumerate(top_features, 1):
        direction = "Indicates ATTACK" if coef > 0 else "Indicates BENIGN"
        f.write(f"{i}. {feature} (coefficient: {coef:.4f}, importance: {importance:.4f}) - {direction}\n")

# Visualize feature importance with direction (positive vs negative coefficients)
plt.figure(figsize=(14, 10))
features, coefficients, _ = zip(*top_features)
colors = ['red' if c > 0 else 'blue' for c in coefficients]
plt.barh(range(len(features)), coefficients, color=colors)
plt.yticks(range(len(features)), features)
plt.xlabel('Coefficient Value')
plt.title('Top 20 Most Important Features (Red = Attack, Blue = Benign)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('output/logistic_regression_feature_importance.png')
plt.close()

# 8. Feature distribution analysis
print("\nPerforming additional feature analysis...")

# Analyze n-gram distribution
ngram_counts = {}
for feature in all_features:
    n = len(feature.split())
    ngram_counts[n] = ngram_counts.get(n, 0) + 1

print("\nN-gram Distribution:")
for n, count in sorted(ngram_counts.items()):
    print(f"{n}-gram: {count} features ({count/len(all_features)*100:.2f}%)")

# Analyze common words
all_words = []
for feature in all_features:
    words = feature.split()
    all_words.extend(words)

word_counts = {}
for word in all_words:
    word_counts[word] = word_counts.get(word, 0) + 1

print("\nTop 30 Most Common Words:")
for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"{word}: {count} occurrences")

# 9. Save model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(final_model, 'models/logistic_regression_model.joblib')
joblib.dump(tfidf_vectorizer, 'models/logistic_regression_vectorizer.joblib')
print("Model and vectorizer saved successfully.")

# Create a prediction function for future use
def predict_alert(alert_text, model, vectorizer):
    """
    Predict if an alert message indicates an attack with Logistic Regression
    """
    # Clean the message
    cleaned_text = clean_alert_message(alert_text)
    
    # Transform using the vectorizer
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    pred_prob = model.predict_proba(text_tfidf)[0, 1]
    pred_class = model.predict(text_tfidf)[0]
    label = "ATTACK" if pred_class == 1 else "BENIGN"
    
    return {
        'prediction': label,
        'probability': pred_prob,
        'class': int(pred_class)
    }

# Save the prediction function code
with open('models/logistic_regression_predict.py', 'w') as f:
    f.write("""
import re
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('models/logistic_regression_model.joblib')
vectorizer = joblib.load('models/logistic_regression_vectorizer.joblib')

def clean_alert_message(text):
    \"\"\"
    Clean alert message by:
    - Removing IP addresses (completely)
    - Removing port numbers (completely)
    - Removing protocol numbers
    - Converting to lowercase
    - Removing extra spaces
    \"\"\"
    # Convert to string if not already
    text = str(text)
    
    # Remove IP addresses (completely remove them)
    text = re.sub(r'\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b', '', text)
    
    # Remove port numbers (completely remove them)
    text = re.sub(r'\\b(?:on|to|from|at|using)\\s+port\\s+\\d+\\b', ' ', text)
    text = re.sub(r'\\bport\\s+\\d+\\b', '', text)
    
    # Remove protocol numbers
    text = re.sub(r'\\bprotocol\\s+\\d+\\b', 'protocol', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def predict_alert(alert_text):
    \"\"\"
    Predict if an alert message indicates an attack
    \"\"\"
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
    print("\\nTesting prediction function with example alerts:\\n")
    
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
""")

print("\nAll tasks completed successfully!")
print("Files generated:")
print("- models/logistic_regression_model.joblib")
print("- models/logistic_regression_vectorizer.joblib")
print("- models/logistic_regression_predict.py")
print("- output/logistic_regression_evaluation.txt")
print("- output/logistic_regression_confusion_matrix.png")
print("- output/logistic_regression_feature_importance.png")
print("- output/logistic_regression_features.csv")
print("- output/logistic_regression_cv_results.csv")
print("- output/logistic_regression_cv_boxplot.png")