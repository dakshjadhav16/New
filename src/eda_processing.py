# Hand Gesture Recognition - Exploratory Data Analysis and Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add the src directory to the path so we can import our modules
sys.path.append(r'D:\Sem6\EDAI\New')
from src.utils import load_dataset, visualize_landmarks

## 1. Loading the Raw Datasets

print("Loading raw landmark datasets...")
# Load the main dataset
landmarks_df = pd.read_csv(r'D:\Sem6\EDAI\New\data\landmarks_dataset_augmented.csv')
print(f"Main dataset loaded with shape: {landmarks_df.shape}")

# Load the second dataset if it exists
second_dataset_path = r'D:\Sem6\EDAI\New\data\lkandmarks_dataset_augmented.csv'
if os.path.exists(second_dataset_path):
    lkandmarks_df = pd.read_csv(second_dataset_path)
    print(f"Second dataset loaded with shape: {lkandmarks_df.shape}")
else:
    lkandmarks_df = None
    print("Second dataset not found, proceeding with just the main dataset.")

## 2. Combine Datasets (Optional)
# If both datasets have the same structure, you can combine them
if lkandmarks_df is not None:
    combined_df = pd.concat([landmarks_df, lkandmarks_df], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
else:
    combined_df = landmarks_df
    print("Using only the main dataset.")

## 3. Basic Data Exploration

print("\n--- Basic Dataset Information ---")
print(combined_df.info())

print("\n--- First few rows of the dataset ---")
print(combined_df.head())

## 4. Exploring the Dataset Structure

# Let's see what gesture classes we have and their distribution
print("\n--- Gesture Class Distribution ---")
if 'gesture' in combined_df.columns:
    gesture_counts = combined_df['gesture'].value_counts()
    print(gesture_counts)
    
    # Visualize the distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=gesture_counts.index, y=gesture_counts.values)
    plt.title('Distribution of Gesture Classes')
    plt.xlabel('Gesture')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No 'gesture' column found. The dataset might not be labeled or uses a different column name.")

## 5. Exploring the Landmark Features

# Assuming columns follow the format 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', etc.
# First, identify the landmark columns
landmark_cols = [col for col in combined_df.columns if col.startswith(('x', 'y', 'z'))]

print(f"\nFound {len(landmark_cols)} landmark features")

# Basic statistics for the landmarks
print("\n--- Landmark Features Statistics ---")
landmark_stats = combined_df[landmark_cols].describe()
print(landmark_stats)

## 6. Visualizing Hand Landmarks

# Function to extract and reshape landmarks from a row for visualization
def get_landmarks_from_row(row, num_landmarks=21):
    """Extract landmarks from a row and reshape them for visualization."""
    landmarks = []
    
    for i in range(num_landmarks):
        x = row.get(f'x{i}', 0)
        y = row.get(f'y{i}', 0)
        z = row.get(f'z{i}', 0)
        landmarks.append([x, y, z])
        
    return np.array(landmarks)

# Visualize landmarks for a few samples from each gesture class
if 'gesture' in combined_df.columns:
    gestures = combined_df['gesture'].unique()
    
    for gesture in gestures[:3]:  # Limiting to first 3 gestures to avoid too many plots
        print(f"\nVisualizing samples for gesture: {gesture}")
        samples = combined_df[combined_df['gesture'] == gesture].head(3)
        
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3)
        
        for i, (_, sample) in enumerate(samples.iterrows()):
            ax = fig.add_subplot(gs[0, i], projection='3d')
            landmarks = get_landmarks_from_row(sample)
            
            # Plot landmarks
            ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='blue', s=20)
            
            # Connect landmarks with lines (simplified - just connecting sequential points)
            for j in range(len(landmarks) - 1):
                ax.plot([landmarks[j, 0], landmarks[j+1, 0]], 
                        [landmarks[j, 1], landmarks[j+1, 1]], 
                        [landmarks[j, 2], landmarks[j+1, 2]], 'k-')
            
            ax.set_title(f"Sample {i+1} - {gesture}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        plt.tight_layout()
        plt.show()
else:
    # If there's no gesture column, just show a few samples
    samples = combined_df.head(3)
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3)
    
    for i, (_, sample) in enumerate(samples.iterrows()):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        landmarks = get_landmarks_from_row(sample)
        
        # Plot landmarks
        ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='blue', s=20)
        
        # Connect landmarks with lines (simplified)
        for j in range(len(landmarks) - 1):
            ax.plot([landmarks[j, 0], landmarks[j+1, 0]], 
                    [landmarks[j, 1], landmarks[j+1, 1]], 
                    [landmarks[j, 2], landmarks[j+1, 2]], 'k-')
        
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    plt.tight_layout()
    plt.show()

## 7. Feature Analysis

# Let's perform some feature analysis on the landmarks

# First, let's check for missing values
print("\n--- Missing Values Check ---")
missing_values = combined_df[landmark_cols].isnull().sum()
print(f"Total missing values: {missing_values.sum()}")
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])

# Check the correlation between landmark features
print("\n--- Feature Correlation Analysis ---")
# Taking a sample of landmark features to visualize correlation (too many to show all)
sample_landmarks = landmark_cols[:10]  # First 10 landmark features
corr_matrix = combined_df[sample_landmarks].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Sample Landmark Features')
plt.tight_layout()
plt.show()

# Perform PCA to see if we can reduce dimensionality
print("\n--- PCA Analysis ---")
scaler = StandardScaler()
scaled_landmarks = scaler.fit_transform(combined_df[landmark_cols])

pca = PCA()
pca_result = pca.fit_transform(scaled_landmarks)

# Plot variance explained
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance by Components')
plt.legend()
plt.tight_layout()
plt.show()

# Determine how many components needed for 95% variance
components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for 95% variance: {components_95}")

## 8. Data Preprocessing

print("\n--- Preprocessing the Dataset ---")

# 8.1 Handling missing values (if any)
if missing_values.sum() > 0:
    print("Handling missing values...")
    # Simple imputation - median
    combined_df[landmark_cols] = combined_df[landmark_cols].fillna(combined_df[landmark_cols].median())

# 8.2 Normalization
print("Normalizing the landmark features...")
normalized_landmarks = scaler.transform(combined_df[landmark_cols])
normalized_df = pd.DataFrame(normalized_landmarks, columns=landmark_cols)

if 'gesture' in combined_df.columns:
    normalized_df['gesture'] = combined_df['gesture'].values

print("Shape after normalization:", normalized_df.shape)

# 8.3 Data Augmentation (using our custom module)
print("\nPerforming data augmentation...")
print("This would typically call the augmentation module from src.augmentation")
print("For this notebook, we'll just describe the process:")
print("- Random rotation: slightly rotating hand positions")
print("- Random scaling: small changes in hand size")
print("- Random translation: shifting hand position")
print("- Adding noise: small perturbations to landmark positions")
print("- Mirroring: flipping gestures (when appropriate)")

# 8.4 Save the processed dataset
output_path = '../data/processed/landmarks_dataset_augmented.csv'
print(f"\nSaving processed dataset to: {output_path}")
normalized_df.to_csv(output_path, index=False)
print("Processed dataset saved.")

## 9. Feature Engineering

print("\n--- Feature Engineering ---")
print("Potential features we could engineer from raw landmarks:")
print("- Hand centroid (average position)")
print("- Distances between fingers")
print("- Angles between finger joints")
print("- Finger curvature")
print("- Distance from palm to fingertips")
print("- Hand orientation (3D rotation)")

# Example of creating distance features (between thumb and other fingertips)
if len(landmark_cols) >= 63:  # Make sure we have enough landmarks
    print("\nCreating sample engineered features: distances from thumb to fingertips")
    
    # Assuming landmark indices: 
    # Thumb tip: 4
    # Index tip: 8
    # Middle tip: 12
    # Ring tip: 16
    # Pinky tip: 20
    
    # Create a function to calculate distance between two landmarks
    def calculate_distance(row, idx1, idx2):
        x1, y1, z1 = row[f'x{idx1}'], row[f'y{idx1}'], row[f'z{idx1}']
        x2, y2, z2 = row[f'x{idx2}'], row[f'y{idx2}'], row[f'z{idx2}']
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    # Calculate distances
    normalized_df['thumb_to_index'] = combined_df.apply(lambda row: calculate_distance(row, 4, 8), axis=1)
    normalized_df['thumb_to_middle'] = combined_df.apply(lambda row: calculate_distance(row, 4, 12), axis=1)
    normalized_df['thumb_to_ring'] = combined_df.apply(lambda row: calculate_distance(row, 4, 16), axis=1)
    normalized_df['thumb_to_pinky'] = combined_df.apply(lambda row: calculate_distance(row, 4, 20), axis=1)
    
    # Display the distribution of these new features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.histplot(normalized_df['thumb_to_index'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distance: Thumb to Index')
    
    sns.histplot(normalized_df['thumb_to_middle'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distance: Thumb to Middle')
    
    sns.histplot(normalized_df['thumb_to_ring'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distance: Thumb to Ring')
    
    sns.histplot(normalized_df['thumb_to_pinky'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distance: Thumb to Pinky')
    
    plt.tight_layout()
    plt.show()

## 10. Data Splitting

print("\n--- Data Splitting Strategy ---")
print("For the hand gesture recognition task, we recommend:")
print("- Training set: 70-80% of the data")
print("- Validation set: 10-15% of the data")
print("- Test set: 10-15% of the data")
print("\nWhen splitting the data, it's important to:")
print("- Maintain class distribution in all sets (stratified sampling)")
print("- Consider person-independence if data comes from multiple individuals")
print("- Ensure temporal consistency if gestures are part of a sequence")

print("\nThis would typically be done using sklearn's train_test_split or GroupKFold")
print("Example code:")
print("from sklearn.model_selection import train_test_split")
print("X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)")
print("X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)")

## 11. Conclusion and Next Steps

print("\n--- Conclusion and Next Steps ---")
print("In this notebook, we have:")
print("1. Loaded and explored the raw hand gesture landmark datasets")
print("2. Visualized hand landmark data")
print("3. Performed feature analysis and correlation study")
print("4. Applied preprocessing (normalization, handling missing values)")
print("5. Discussed data augmentation strategies")
print("6. Engineered additional features")
print("7. Outlined a data splitting strategy")

print("\nNext steps:")
print("1. Implement data augmentation using src.augmentation")
print("2. Finalize the feature engineering process")
print("3. Implement the data splitting strategy")
print("4. Move to model training (see Model_Training.ipynb)")
print("5. Consider collecting additional data if certain gestures are underrepresented")