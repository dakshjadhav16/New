# Hand Gesture Recognition - Model Training and Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import pickle
import joblib
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Add the src directory to the path so we can import our modules
sys.path.append(r'D:\Sem6\EDAI\New')
from src.utils import load_dataset, visualize_landmarks

## 1. Load the Processed Dataset

print("Loading the processed dataset...")
try:
    # Load the processed and augmented dataset
    dataset_path = r'D:\Sem6\EDAI\New\data\landmarks_dataset_augmented.csv'
    if os.path.exists(dataset_path):
        # Read the CSV without headers since your data doesn't seem to have column headers
        df = pd.read_csv(dataset_path, header=None)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Extract the first column as labels and the rest as features
        labels = df.iloc[:, 0].values
        features = df.iloc[:, 1:].values
        
        # Create a new DataFrame with proper structure
        numeric_df = pd.DataFrame(features)
        numeric_df['label'] = labels
        df = numeric_df
        
        # Simplify the class labels to reduce the number of classes
        print("Simplifying class labels to reduce number of classes...")
        
        # Example mapping function
        def map_to_base_class(label):
            # Remove variations and keep the base letter/sign
            if isinstance(label, str):  # Make sure the label is a string
                base = label.split('_')[0]
                return base
            return label  # Return as is if not a string
        
        # Apply the mapping
        df['simplified_label'] = df['label'].apply(map_to_base_class)
        
        # Use simplified_label instead of label going forward
        df['label'] = df['simplified_label']
        df.drop('simplified_label', axis=1, inplace=True)
        
        # Print the new number of unique classes
        unique_classes = df['label'].nunique()
        print(f"Reduced classes to {unique_classes} basic sign language symbols")
        
    else:
        # If the processed dataset doesn't exist, load from raw and do basic preprocessing
        print("Processed dataset not found. Loading from raw...")
        df = pd.read_csv(r'D:\Sem6\EDAI\New\data\landmarks_dataset_augmented.csv')
        print(f"Raw dataset loaded with shape: {df.shape}")
        
        # Perform basic preprocessing
        print("Performing basic preprocessing...")
        # Assuming we have landmark columns and a 'gesture' column
        feature_cols = [col for col in df.columns if col not in ['gesture', 'subject_id', 'timestamp']]
        
        # Fill missing values if any
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale features
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Create dummy data for demonstration
    print("Creating dummy data for demonstration...")
    feature_cols = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
    df = pd.DataFrame(np.random.randn(1000, 63), columns=feature_cols)
    df['gesture'] = np.random.choice(['thumbs_up', 'peace', 'palm', 'fist', 'pointing'], size=1000)

## 2. Data Preparation

# Prepare features and target
print("\n--- Preparing features and target ---")

# Check if 'gesture' column exists
if 'label' in df.columns:
    # Encode the gesture labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    classes = label_encoder.classes_
    print(f"Classes: {classes}")
    print(f"Encoded classes: {np.unique(y)}")
    
    # Get feature columns (exclude label column)
    feature_cols = [col for col in df.columns if col != 'label']
else:
    # For demonstration without actual labels
    print("No 'label' column found. Creating dummy labels...")
    y = np.random.randint(0, 5, size=len(df))
    classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    feature_cols = df.columns.tolist()


X = df[feature_cols].values
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# new
# Count the occurrences of each class
value_counts = pd.Series(y).value_counts()
print(f"Number of classes: {len(value_counts)}")
print(f"Samples per class: min={value_counts.min()}, max={value_counts.max()}, mean={value_counts.mean():.2f}")

# Identify classes with too few samples
small_classes = value_counts[value_counts < 2].index
print(f"Found {len(small_classes)} classes with less than 2 samples")

# Filter out classes with too few samples
if len(small_classes) > 0:
    print("Removing classes with too few samples...")
    mask = ~np.isin(y, small_classes)
    X = X[mask]
    y = y[mask]
    print(f"Filtered dataset shape: X={X.shape}, y={len(y)}")
    
    # Re-encode the labels to ensure continuous indices
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    print(f"New number of classes: {len(classes)}")

# Now perform the train/test split

# Split the data into training, validation, and test sets
# Split the data into training, validation, and test sets
# Remove stratify=y if you still have too many rare classes
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

## 3. Model Training - Classic ML Models

print("\n--- Training Classical ML Models ---")

### 3.1 Random Forest

print("\n3.1 Random Forest Classifier")

# Modified Random Forest with parameters to handle large number of classes
# Random Forest with better params for fewer classes
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    verbose=1
)

# Consider using a subset of data for initial testing
# Uncomment if needed:
# from sklearn.utils import resample
# X_train_sample, y_train_sample = resample(X_train, y_train, 
#                                          n_samples=min(50000, len(X_train)),
#                                          random_state=42)
# rf_model.fit(X_train_sample, y_train_sample)

# Regular fit with full dataset
print("Starting Random Forest training - this may take time with 7,324 classes")
rf_model.fit(X_train, y_train)

# Evaluate on validation set
rf_val_pred = rf_model.predict(X_val)
rf_val_acc = accuracy_score(y_val, rf_val_pred)
print(f"Random Forest Validation Accuracy: {rf_val_acc:.4f}")

# Feature importance - only show top features to save memory
top_n = 20  # Only process top features
feature_indices = np.argsort(rf_model.feature_importances_)[-top_n:]
top_features = [feature_cols[i] for i in feature_indices]
top_importances = rf_model.feature_importances_[feature_indices]

feature_imp = pd.DataFrame({
    'Feature': top_features,
    'Importance': top_importances
}).sort_values('Importance', ascending=False)

print(f"Top {top_n} most important features:")
print(feature_imp)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Optional: Grid Search for hyperparameter tuning
print("\nPerforming Grid Search for Random Forest (commented out to save time)")
"""
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(5),
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use the best model
best_rf_model = grid_search.best_estimator_
rf_model = best_rf_model  # Update the model
"""
joblib.dump(rf_model, 'random_forest_model.joblib')
print("Model saved successfully as 'random_forest_model.joblib'")

# Method 2: Save using pickle
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Model also saved as 'random_forest_model.pkl'")

## 4. Model Training - Deep Learning Models

print("\n--- Training Deep Learning Models ---")

### 4.1 Simple Dense Neural Network

print("\n4.1 Dense Neural Network")

# Create a simple dense neural network
def create_dense_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

num_classes = len(np.unique(y))
dense_model = create_dense_model(X_train.shape[1], num_classes)
print(dense_model.summary())

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
]

# Train the model
print("\nTraining Dense Neural Network...")
dense_history = dense_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(dense_history.history['accuracy'], label='Training Accuracy')
plt.plot(dense_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(dense_history.history['loss'], label='Training Loss')
plt.plot(dense_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

### 4.2 LSTM Model for Sequential Data

print("\n4.2 LSTM Model (if data has sequential structure)")
print("Note: This section assumes the data has temporal/sequential structure.")
print("If your data doesn't have sequential structure, you can skip this section.")

# Reshape data for LSTM if it has sequential structure
# For hand landmarks, we'd typically reshape 63 features (21 landmarks x 3 coordinates) 
# into a sequence of 21 landmarks with 3 features each

try:
    # Try to reshape the data for LSTM
    # Assuming X_train contains 21 landmarks with x, y, z coordinates in sequence
    if X_train.shape[1] == 63:  # 21 landmarks * 3 coordinates
        # Reshape to [samples, timesteps, features]
        # Here, each landmark is a timestep, and x,y,z are features
        X_train_lstm = X_train.reshape(X_train.shape[0], 21, 3)
        X_val_lstm = X_val.reshape(X_val.shape[0], 21, 3)
        X_test_lstm = X_test.reshape(X_test.shape[0], 21, 3)
        
        print(f"Reshaped data for LSTM - Training set: {X_train_lstm.shape}")
        
        # Create LSTM model
        def create_lstm_model(input_shape, num_classes):
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        lstm_model = create_lstm_model((21, 3), num_classes)
        print(lstm_model.summary())
        
        # Train the LSTM model
        print("\nTraining LSTM Model...")
        lstm_history = lstm_model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(lstm_history.history['accuracy'], label='Training Accuracy')
        plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('LSTM Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(lstm_history.history['loss'], label='Training Loss')
        plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print("Data shape doesn't match expected dimensions for LSTM reshaping")
except Exception as e:
    print(f"Error in LSTM section: {e}")
    print("Skipping LSTM model due to error")

## 5. Model Evaluation

print("\n--- Model Evaluation on Test Set ---")

### 5.1 Random Forest Evaluation
rf_test_pred = rf_model.predict(X_test)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")

# Detailed metrics
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_test_pred, target_names=classes))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 5.2 Dense Neural Network Evaluation
dense_test_pred = dense_model.predict(X_test)
dense_test_pred_classes = np.argmax(dense_test_pred, axis=1)
dense_test_acc = accuracy_score(y_test, dense_test_pred_classes)
print(f"\nDense Neural Network Test Accuracy: {dense_test_acc:.4f}")

# Detailed metrics
print("\nDense Neural Network Classification Report:")
print(classification_report(y_test, dense_test_pred_classes, target_names=classes))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, dense_test_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Dense Neural Network Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### 5.3 LSTM Evaluation (if available)
try:
    if 'lstm_model' in locals():
        lstm_test_pred = lstm_model.predict(X_test_lstm)
        lstm_test_pred_classes = np.argmax(lstm_test_pred, axis=1)
        lstm_test_acc = accuracy_score(y_test, lstm_test_pred_classes)
        print(f"\nLSTM Model Test Accuracy: {lstm_test_acc:.4f}")
        
        # Detailed metrics
        print("\nLSTM Model Classification Report:")
        print(classification_report(y_test, lstm_test_pred_classes, target_names=classes))
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, lstm_test_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('LSTM Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
except:
    print("LSTM evaluation skipped")

### 5.4 Model Comparison

print("\n--- Model Comparison ---")
models = ['Random Forest']
accuracies = [rf_test_acc]

if 'dense_test_acc' in locals():
    models.append('Dense Neural Network')
    accuracies.append(dense_test_acc)

if 'lstm_test_acc' in locals():
    models.append('LSTM')
    accuracies.append(lstm_test_acc)

plt.figure(figsize=(10, 6))
sns