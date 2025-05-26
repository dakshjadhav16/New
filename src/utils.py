import os
import sys
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("sign_language_utils")

# src/utils.py
import pandas as pd

def load_dataset(file_path):
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def visualize_landmarks(landmarks):
    """
    Visualize hand landmarks in 3D.
    
    Args:
        landmarks (np.array): Array of landmark coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2])
    plt.show()

def create_folders_if_not_exist(folder_list):
    """
    Create folders if they don't exist
    
    Args:
        folder_list (list): List of folder paths to create
    """
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"Created folder: {folder}")
        else:
            logger.info(f"Folder already exists: {folder}")

def get_project_root():
    """
    Get the project root directory
    
    Returns:
        Path: Project root path
    """
    return Path(__file__).parent.parent

def normalize_landmarks(landmarks, min_val=-1, max_val=1):
    """
    Normalize landmarks to a specific range
    
    Args:
        landmarks (numpy.ndarray): Input landmarks
        min_val (float): Minimum value after normalization
        max_val (float): Maximum value after normalization
        
    Returns:
        numpy.ndarray: Normalized landmarks
    """
    # Get min and max values
    landmarks_min = np.min(landmarks)
    landmarks_max = np.max(landmarks)
    
    # Normalize
    normalized = (landmarks - landmarks_min) / (landmarks_max - landmarks_min)
    
    # Scale to desired range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def calculate_angles(landmarks):
    """
    Calculate angles between finger joints from landmarks
    
    Args:
        landmarks (numpy.ndarray): Hand landmarks in format [x1, y1, z1, x2, y2, z2, ...]
        
    Returns:
        numpy.ndarray: Array of angles between joints
    """
    # Reshape landmarks to (21, 3) format - MediaPipe has 21 hand landmarks with x,y,z coordinates
    points = landmarks.reshape(-1, 3)
    
    # Calculate angles
    angles = []
    
    # Finger indices (MediaPipe hand landmark indices)
    # Thumb: 1,2,3,4
    # Index: 5,6,7,8
    # Middle: 9,10,11,12
    # Ring: 13,14,15,16
    # Pinky: 17,18,19,20
    fingers = [
        [1, 2, 3, 4],      # Thumb
        [5, 6, 7, 8],      # Index
        [9, 10, 11, 12],   # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20]   # Pinky
    ]
    
    # Calculate angle for each finger joint
    for finger in fingers:
        for i in range(len(finger) - 2):
            # Get three consecutive points
            p1 = points[finger[i]]
            p2 = points[finger[i + 1]]
            p3 = points[finger[i + 2]]
            
            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm == 0 or v2_norm == 0:
                angles.append(0)
                continue
            
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Calculate angle using dot product
            dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            angles.append(angle)
    
    return np.array(angles)

def visualize_confusion_matrix(cm, class_names, output_path=None):
    """
    Visualize confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        output_path (str, optional): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()