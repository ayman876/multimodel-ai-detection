import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gc  # For garbage collection
import time
import warnings
from imblearn.over_sampling import SMOTE  # Added for SMOTE
warnings.filterwarnings('ignore')

# Set the display options for better output readability
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3, suppress=True)

# Step 1: Load and Combine CSVs
print("Step 1: Loading and combining CSV files...")
start_time = time.time()

# Define the filenames to load
filenames = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
]

base_path = "./data/MachineLearningCVE"

# Function to load and clean each CSV file
def load_csv(filepath):
    print(f"Loading {os.path.basename(filepath)}...")
    try:
        # Use low_memory=False to ensure proper column type inference
        df = pd.read_csv(filepath, low_memory=False, encoding='latin1')
        # Strip whitespace from column names immediately
        df.columns = df.columns.str.strip()
        print(f"  Loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Load and concatenate all CSV files
all_dfs = []
for filename in filenames:
    filepath = os.path.join(base_path, filename)
    df = load_csv(filepath)
    if df is not None:
        all_dfs.append(df)
    # Clear memory
    gc.collect()

# Concatenate all dataframes
df_combined = pd.concat(all_dfs, ignore_index=True)
print(f"Combined dataset shape: {df_combined.shape}")

# Clear memory
all_dfs = None
gc.collect()

print(f"Step 1 completed in {time.time() - start_time:.2f} seconds")

# Step 2: Clean and Preprocess Data
print("\nStep 2: Cleaning and preprocessing data...")
start_time = time.time()

# Check for duplicate rows
print(f"Checking for duplicates among {df_combined.shape[0]} rows...")
duplicates = df_combined.duplicated()
print(f"Found {duplicates.sum()} duplicate rows")
df_combined = df_combined.drop_duplicates()
print(f"After removing duplicates: {df_combined.shape}")

# Replace inf/-inf with NaN
print("Replacing infinite values with NaN...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaNs in 'Flow Bytes/s' and 'Flow Packets/s' with their median
for col in ['Flow Bytes/s', 'Flow Packets/s']:
    if col in df_combined.columns:
        median_value = df_combined[col].median()
        print(f"Filling NaN in '{col}' with median: {median_value}")
        df_combined[col].fillna(median_value, inplace=True)

# Get columns with only one unique value
single_value_cols = [col for col in df_combined.columns 
                    if df_combined[col].nunique() == 1]
print(f"Columns with only one unique value: {single_value_cols}")

# Get columns with zero standard deviation
zero_std_cols = [col for col in df_combined.columns 
                if col not in single_value_cols and  # Skip already identified cols
                df_combined[col].dtype.kind in 'bifc' and  # Only numeric columns
                df_combined[col].std() == 0]
print(f"Columns with zero standard deviation: {zero_std_cols}")

# Drop columns with one unique value or zero std
cols_to_drop = single_value_cols + zero_std_cols
print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
df_combined.drop(columns=cols_to_drop, inplace=True)

print(f"Step 2 completed in {time.time() - start_time:.2f} seconds")
print(f"Dataset shape after cleaning: {df_combined.shape}")

# Step 3: Process Labels for Multiclass
print("\nStep 3: Processing labels for multiclass classification...")
start_time = time.time()

# Define the mapping for attack types
attack_mapping = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack ï¿½ Brute Force': 'Web Attack',    # Added with encoding issues
    'Web Attack ï¿½ XSS': 'Web Attack',            # Added with encoding issues
    'Web Attack ï¿½ Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

# Check that 'Label' column exists
if 'Label' in df_combined.columns:
    # Create 'Attack Type' column based on mapping
    df_combined['Attack Type'] = df_combined['Label'].map(lambda x: attack_mapping.get(x, x))
    
    # Print value counts for both columns
    print("Original 'Label' distribution:")
    print(df_combined['Label'].value_counts())
    print("\n'Attack Type' distribution after mapping:")
    print(df_combined['Attack Type'].value_counts())
    
    # Drop the original 'Label' column
    df_combined.drop(columns=['Label'], inplace=True)
else:
    print("Warning: 'Label' column not found in dataset")

print(f"Step 3 completed in {time.time() - start_time:.2f} seconds")


X = df_combined.drop(columns=['Attack Type'])
y = df_combined['Attack Type']


# Save feature names before PCA
feature_names_before_pca = X.columns.tolist()
joblib.dump(feature_names_before_pca, 'models/features_before_pca.joblib')

# Print them
print("\nFeatures before PCA (after cleaning and scaling):")
for i, feature in enumerate(feature_names_before_pca):
    print(f"{i+1}. {feature}")

# After the preprocessing and before training and testing the model
df_combined.to_csv('cleaned_preprocessed_data.csv', index=False)
print("Cleaned and preprocessed dataset saved as 'cleaned_preprocessed_data.csv'.")


# Step 4: Optimize Memory
print("\nStep 4: Optimizing memory usage...")
start_time = time.time()

# Function to get memory usage of a dataframe
def get_memory_usage(df):
    return f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"

print(f"Memory usage before optimization: {get_memory_usage(df_combined)}")

# Downcast numeric columns
for col in df_combined.columns:
    if df_combined[col].dtype.kind in 'fc':  # Float columns
        df_combined[col] = pd.to_numeric(df_combined[col], downcast='float')
    elif df_combined[col].dtype.kind in 'i':  # Integer columns
        df_combined[col] = pd.to_numeric(df_combined[col], downcast='integer')

print(f"Memory usage after optimization: {get_memory_usage(df_combined)}")
print(f"Step 4 completed in {time.time() - start_time:.2f} seconds")

# Step 5: Standardize and Reduce Features
print("\nStep 5: Standardizing and reducing features...")
start_time = time.time()

# Separate features and target
X = df_combined.drop(columns=['Attack Type'])
y = df_combined['Attack Type']

# Keep track of column names for non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_cols:
    print(f"Non-numeric columns that will be excluded from scaling: {non_numeric_cols}")
    X = X.drop(columns=non_numeric_cols)

joblib.dump(X.columns.tolist(), 'models/model1_feature_order.joblib')


# Standardize numeric features
print("Applying StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Incremental PCA
n_components = int(X.shape[1] * 0.5)  # 50% of the number of features
print(f"Applying Incremental PCA with {n_components} components (50% of {X.shape[1]} features)...")
ipca = IncrementalPCA(n_components=n_components, batch_size=500)
X_pca = ipca.fit_transform(X_scaled)

# Calculate explained variance
explained_var = sum(ipca.explained_variance_ratio_) * 100
print(f"Explained variance by {n_components} components: {explained_var:.2f}%")

# Create a new dataset with PCA features + 'Attack Type'
pca_cols = [f'PC{i+1}' for i in range(n_components)]
X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
X_pca_df['Attack Type'] = y.values

print(f"PCA-transformed dataset shape: {X_pca_df.shape}")
print(f"Step 5 completed in {time.time() - start_time:.2f} seconds")

# NEW STEP: Filter, Downsample, and Balance Classes
print("\nStep 5b: Filtering, downsampling, and balancing classes...")
start_time = time.time()

# This is the PCA-transformed dataframe with 'Attack Type'
new_data = X_pca_df

# 1. Filter classes with enough samples (more than 1950)
print("Filtering classes with enough samples...")
class_counts = new_data['Attack Type'].value_counts()
print("Original class distribution:")
print(class_counts)

selected_classes = class_counts[class_counts > 1950]
class_names = selected_classes.index
print(f"\nSelected {len(class_names)} classes with more than 1950 samples:")
print(selected_classes)

# Filter dataset to keep only selected classes
selected = new_data[new_data['Attack Type'].isin(class_names)]
print(f"Filtered data shape: {selected.shape}")

# 2. Downsample high-frequency classes to 5000 max
print("\nDownsampling high-frequency classes...")
dfs = []
for name in class_names:
    df_class = selected[selected['Attack Type'] == name]
    if len(df_class) > 2500:
        print(f"Downsampling class '{name}' from {len(df_class)} to 5000 samples")
        df_class = df_class.sample(n=5000, random_state=0)
    else:
        print(f"Keeping all {len(df_class)} samples for class '{name}'")
    dfs.append(df_class)

# Combine all classes back into one DataFrame
df = pd.concat(dfs, ignore_index=True)
print(f"After downsampling, data shape: {df.shape}")
print("Class distribution after downsampling:")
print(df['Attack Type'].value_counts())

# 3. Balance classes using SMOTE to get exactly 5000 samples per class
print("\nBalancing classes with SMOTE...")
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# Define target count as 5000 for each class
sampling_strategy = {name: 5000 for name in class_names}

# Apply SMOTE
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create balanced DataFrame
blnc_data = pd.DataFrame(X_resampled, columns=X.columns)
blnc_data['Attack Type'] = y_resampled

# Shuffle the data
blnc_data = blnc_data.sample(frac=1, random_state=0)

print(f"After SMOTE, data shape: {blnc_data.shape}")
print("Final class distribution (should be 5000 each):")
print(blnc_data['Attack Type'].value_counts())

print(f"Step 5b completed in {time.time() - start_time:.2f} seconds")

# Step 6: Train and Evaluate Random Forest (MODIFIED)
print("\nStep 6: Training and evaluating Random Forest models with balanced data...")
start_time = time.time()

# Split data into X and y from the balanced dataset
features = blnc_data.drop('Attack Type', axis=1)
labels = blnc_data['Attack Type']

# Split into train and test sets (25% test size)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=0  # Changed to match the specified random_state
)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Train Model 1 with specified hyperparameters
print("Training Model 1 (10 trees, max_depth=6, no max_features)...")
model1 = RandomForestClassifier(
    n_estimators=10,
    max_depth=6,
    max_features=None,
    random_state=0
)
model1.fit(X_train, y_train)

# Train Model 2 with specified hyperparameters
print("Training Model 2 (15 trees, max_depth=8, max_features=20)...")
model2 = RandomForestClassifier(
    n_estimators=15,
    max_depth=8,
    max_features=20,
    random_state=0
)
model2.fit(X_train, y_train)

# Evaluate Model 1
print("\nEvaluating Model 1...")
y_pred1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"Accuracy: {accuracy1:.4f}")

# Cross-validation
cv_scores1 = cross_val_score(model1, X_train, y_train, cv=5)
print(f"5-fold Cross-validation scores: {cv_scores1}")
print(f"Mean CV score: {cv_scores1.mean():.4f}")

# Confusion matrix
cm1 = confusion_matrix(y_test, y_pred1)
print("Confusion Matrix:")
print(cm1)

# Classification report
cr1 = classification_report(y_test, y_pred1, output_dict=True)
print("Classification Report:")
print(classification_report(y_test, y_pred1))

# Evaluate Model 2
print("\nEvaluating Model 2...")
y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Accuracy: {accuracy2:.4f}")

# Cross-validation
cv_scores2 = cross_val_score(model2, X_train, y_train, cv=5)
print(f"5-fold Cross-validation scores: {cv_scores2}")
print(f"Mean CV score: {cv_scores2.mean():.4f}")

# Confusion matrix
cm2 = confusion_matrix(y_test, y_pred2)
print("Confusion Matrix:")
print(cm2)

# Classification report
cr2 = classification_report(y_test, y_pred2, output_dict=True)
print("Classification Report:")
print(classification_report(y_test, y_pred2))

print(f"Step 6 completed in {time.time() - start_time:.2f} seconds")

# Step 7: Visualize Performance
print("\nStep 7: Visualizing model performance...")
start_time = time.time()

# Set up the figure layout
plt.figure(figsize=(20, 15))

# Plot 1: Heatmap for Classification Report of Model 1
plt.subplot(2, 2, 1)
classes1 = list(cr1.keys())[:-3]  # Exclude 'accuracy', 'macro avg' and 'weighted avg'
metrics = ['precision', 'recall', 'f1-score']
cr1_df = pd.DataFrame({metric: [cr1[cls][metric] for cls in classes1] for metric in metrics}, index=classes1)
sns.heatmap(cr1_df, annot=True, cmap='Blues', fmt='.3f')
plt.title('Classification Report Metrics - Model 1 (10 trees, max_depth=6)')
plt.ylabel('Classes')
plt.tight_layout()

# Plot 2: Heatmap for Classification Report of Model 2
plt.subplot(2, 2, 2)
classes2 = list(cr2.keys())[:-3]  # Exclude 'accuracy', 'macro avg' and 'weighted avg'
cr2_df = pd.DataFrame({metric: [cr2[cls][metric] for cls in classes2] for metric in metrics}, index=classes2)
sns.heatmap(cr2_df, annot=True, cmap='Blues', fmt='.3f')
plt.title('Classification Report Metrics - Model 2 (15 trees, max_depth=8, max_features=20)')
plt.ylabel('Classes')
plt.tight_layout()

# Plot 3: Bar chart comparing accuracy
plt.subplot(2, 2, 3)
models = ['Model 1', 'Model 2']
accuracies = [accuracy1, accuracy2]
plt.bar(models, accuracies, color=['skyblue', 'navy'])
plt.ylim(min(accuracies) - 0.05, 1.0)
plt.axhline(y=max(accuracies), color='red', linestyle='--', alpha=0.7)
plt.text(0, max(accuracies) + 0.01, f'Best: {max(accuracies):.4f}', color='red')
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.tight_layout()

# Plot 4: Bar chart comparing cross-validation scores
plt.subplot(2, 2, 4)
cv_means = [cv_scores1.mean(), cv_scores2.mean()]
cv_stds = [cv_scores1.std(), cv_scores2.std()]
plt.bar(models, cv_means, yerr=cv_stds, capsize=10, color=['skyblue', 'navy'])
plt.ylim(min(cv_means) - 0.05, 1.0)
plt.axhline(y=max(cv_means), color='red', linestyle='--', alpha=0.7)
plt.text(0, max(cv_means) + 0.01, f'Best: {max(cv_means):.4f}', color='red')
plt.title('Cross-Validation Score Comparison')
plt.ylabel('Mean CV Score')
plt.tight_layout()

# Save the figure
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')

# Create separate confusion matrix plots (larger, more readable)
plt.figure(figsize=(20, 8))

# Confusion Matrix for Model 1
plt.subplot(1, 2, 1)
classes = sorted(labels.unique())
cm1_norm = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm1_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Normalized Confusion Matrix - Model 1')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Confusion Matrix for Model 2
plt.subplot(1, 2, 2)
cm2_norm = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm2_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Normalized Confusion Matrix - Model 2')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')

print(f"Visualization saved as 'model_performance_comparison.png' and 'confusion_matrices.png'")
print(f"Step 7 completed in {time.time() - start_time:.2f} seconds")

# Step 8: Choose the Best Model
print("\nStep 8: Choosing the best model...")
start_time = time.time()

# Calculate overall F1 score (weighted average)
f1_model1 = cr1['weighted avg']['f1-score']
f1_model2 = cr2['weighted avg']['f1-score']

# Calculate class balance metric (min/max F1 ratio)
class_f1_model1 = [cr1[cls]['f1-score'] for cls in classes1]
class_f1_model2 = [cr2[cls]['f1-score'] for cls in classes2]

f1_balance1 = min(class_f1_model1) / max(class_f1_model1)
f1_balance2 = min(class_f1_model2) / max(class_f1_model2)

# Create a comparison table
model_comparison = pd.DataFrame({
    'Model 1 (10 trees, max_depth=6)': [accuracy1, cv_scores1.mean(), f1_model1, f1_balance1],
    'Model 2 (15 trees, max_depth=8, max_features=20)': [accuracy2, cv_scores2.mean(), f1_model2, f1_balance2]
}, index=['Accuracy', 'CV Score', 'Weighted F1', 'F1 Balance (min/max)'])

print("Model Comparison:")
print(model_comparison)

# Determine the best model
best_model = 'Model 1 (10 trees, max_depth=6)' if (accuracy1 + cv_scores1.mean() + f1_model1 + f1_balance1) > (accuracy2 + cv_scores2.mean() + f1_model2 + f1_balance2) else 'Model 2 (15 trees, max_depth=8, max_features=20)'
print(f"\nBased on overall metrics, the best model is: {best_model}")

# Print the most important features for the best model
if best_model.startswith('Model 1'):
    selected_model = model1
else:
    selected_model = model2

# Get feature importances
importance = selected_model.feature_importances_
feature_names = features.columns

# Sort feature importances in descending order
sorted_indices = np.argsort(importance)[::-1]
sorted_importance = importance[sorted_indices]
sorted_features = feature_names[sorted_indices]

print(f"\nTop 10 most important features for {best_model}:")
for i in range(min(10, len(sorted_features))):
    print(f"{sorted_features[i]}: {sorted_importance[i]:.4f}")

print(f"Step 8 completed in {time.time() - start_time:.2f} seconds")
print("\nRandom Forest classification model training and evaluation completed!")

# Step 9: Export Model 2 using joblib
print("\nStep 9: Exporting Model 2 using joblib...")
start_time = time.time()

# Import joblib for model persistence
import joblib
import os

# Create models directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Export Model 2
model_filename = os.path.join(models_dir, 'rf_model2_cicids.joblib')
joblib.dump(model2, model_filename)
print(f"Model 2 exported to {model_filename}")

# For complete deployment, also export the scaler and PCA components
# This will allow proper preprocessing of new data for predictions
scaler_filename = os.path.join(models_dir, 'standard_scaler.joblib')
joblib.dump(scaler, scaler_filename)
print(f"StandardScaler exported to {scaler_filename}")

pca_filename = os.path.join(models_dir, 'incremental_pca.joblib')
joblib.dump(ipca, pca_filename)
print(f"IncrementalPCA exported to {pca_filename}")

print(f"Step 9 completed in {time.time() - start_time:.2f} seconds")
print("\nRandom Forest classification model training and evaluation completed!")

# Example code to load and use the model for prediction
print("\nExample of how to load and use the exported model:")
print("""
# Load the model, scaler, and PCA components
import joblib
import os

models_dir = 'models'
model = joblib.load(os.path.join(models_dir, 'rf_model2_cicids.joblib'))
scaler = joblib.load(os.path.join(models_dir, 'standard_scaler.joblib'))
ipca = joblib.load(os.path.join(models_dir, 'incremental_pca.joblib'))

# For new data preprocessing and prediction
def predict_attack_type(new_data):
    # Preprocess new data similarly to training data
    # 1. Scale the data
    new_data_scaled = scaler.transform(new_data)
    # 2. Apply PCA transformation
    new_data_pca = ipca.transform(new_data_scaled)
    # 3. Make predictions
    predictions = model.predict(new_data_pca)
    return predictions
""")
