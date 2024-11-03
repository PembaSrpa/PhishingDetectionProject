import pickle  # For saving the trained model
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
import warnings  # To handle warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # For splitting and evaluating models
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from xgboost import XGBClassifier  # XGBoost classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # For evaluating model performance
import time  # For measuring execution time
import os  # For directory operations

# Summary:
# This code implements a machine learning pipeline for classifying URLs as either phishing or legitimate. 
# It loads a dataset, preprocesses the data, splits it into training and test sets, and evaluates multiple models 
# (Random Forest, XGBoost, and Decision Tree) using hyperparameter tuning via GridSearchCV. 
# It calculates various performance metrics for each model and saves the best-performing model along with its confusion matrix visualizations.

# Suppress warnings
warnings.filterwarnings('ignore')

# Create a directory for saving diagrams if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

# Load dataset
d = pd.read_csv("dataset.csv")  # Read the CSV file
print("Columns in the dataset:", d.columns)  # Display the dataset columns
print("First few rows of the dataset:\n", d.head())  # Display the first few rows
print("\nChecking for NaN values in the dataset...")
print(d.isna().sum())  # Check for missing values

# Preprocess dataset
if not set(d['status'].unique()).issubset({0, 1}):  # Check if 'status' is binary
    print("Mapping 'status' column to binary values...")
    d['status'] = d['status'].map({'legitimate': 0, 'phishing': 1})  # Map values to binary
    d = d.dropna(subset=['status'])  # Drop rows with missing 'status'

# Separate features and target variable
x = d.drop(columns=['status', 'url'], errors='ignore')  # Features
y = d['status']  # Target variable
print("Feature DataFrame (x) shape:", x.shape)  # Print shape of features
print("Target Series (y) shape:", y.shape)  # Print shape of target

# Convert object columns to float or drop them
for col in x.columns:
    if x[col].dtype == 'object':
        try:
            x[col] = x[col].astype(float)  # Convert to float
        except ValueError:
            print(f"Non-numeric column detected and dropped: {col}")  # Drop non-numeric columns
            x = x.drop(columns=[col])

# Display features and target
print("Features:\n", x.head())  # Display feature data
print("\nTarget:\n", y.head())  # Display target data
print("Class distribution in the dataset:\n", y.value_counts())  # Print class distribution

# Split dataset into training and testing sets
if x.shape[0] > 0 and y.shape[0] > 0:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Split data
else:
    raise ValueError("Dataset is empty after preprocessing. Please check the data source and processing steps.")  # Check for empty dataset

def evaluate_model(model, x_train, y_train):
    # Evaluate model using cross-validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    return cv_scores.mean()  # Return mean cross-validation score

# Initialize models and parameter grids for hyperparameter tuning
models = {
    'Random Forest': RandomForestClassifier(random_state=0, class_weight='balanced'),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=0),
    'Decision Tree': DecisionTreeClassifier(random_state=0, class_weight='balanced'),
}

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
}

# Variables to track the best overall model
best_overall_model = None
best_overall_accuracy = 0

# Train and evaluate each model
for model_name in models:
    print(f"\nTraining {model_name}...")
    start_time = time.time()  # Start timer
    
    # Perform grid search to find the best parameters
    grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_  # Get the best model from grid search
    cv_score = evaluate_model(best_model, x_train, y_train)  # Evaluate using cross-validation

    # Make predictions on the test set
    y_pred = best_model.predict(x_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display model performance
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Cross-validation accuracy for {model_name}: {cv_score:.4f}")
    print(f"Test Accuracy for {model_name}: {accuracy:.4f}")
    print(f"Precision for {model_name}: {precision:.4f}")
    print(f"Recall for {model_name}: {recall:.4f}")
    print(f"F1 Score for {model_name}: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)  # Display confusion matrix

    # Save the trained model to a file
    model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(best_model, model_file)

    # Check if this model is the best overall
    if accuracy > best_overall_accuracy:
        best_overall_accuracy = accuracy
        best_overall_model = best_model

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Time taken to train {model_name}: {elapsed_time:.2f} seconds")

    # Save the confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")  # Create heatmap of confusion matrix
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f"diagrams/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")  # Save plot as image
    plt.close()  # Close the plot to free up memory

# Save the best overall model as model.pkl
if best_overall_model:
    with open("model.pkl", "wb") as best_model_file:
        pickle.dump(best_overall_model, best_model_file)
    print("Best model saved as model.pkl")  # Confirm saving the best model
