import numpy as np
import pandas as pd
import os
import random
import logging
import sys
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("landmark_augmenter")

def ensure_dir(directory):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory (str): Path to the directory to ensure exists
        
    Returns:
        str: The path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

class LandmarkAugmenter:
    """
    Class for augmenting hand landmark data to increase dataset size and variety.
    Supports various augmentation techniques like rotation, scaling, translation,
    adding noise, and flipping.
    """
    
    def __init__(self, landmark_csv_path: str, output_csv_path: Optional[str] = None):
        """
        Initialize the augmenter with the path to the landmark CSV file.
        
        Args:
            landmark_csv_path: Path to the CSV file containing landmark data
            output_csv_path: Path to save the augmented landmark data (default: None)
        """
        self.landmark_csv_path = landmark_csv_path
        self.output_csv_path = output_csv_path or self._generate_output_path(landmark_csv_path)
        self.data = pd.read_csv(landmark_csv_path)
        
        # Determine the number of landmarks in the dataset
        # Assuming format: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
        self.num_landmarks = (len(self.data.columns) - 1) // 3
        print(f"Detected {self.num_landmarks} landmarks in the dataset")
    
    def _generate_output_path(self, input_path: str) -> str:
        """
        Generate a unique output path for the augmented dataset.
        
        Args:
            input_path: Path to the input CSV file
            
        Returns:
            str: Unique output path for the augmented dataset
        """
        base_dir = os.path.dirname(input_path)
        base_name = os.path.basename(input_path).replace('.csv', '')
        output_name = f"{base_name}_augmented.csv"
        output_path = os.path.join(base_dir, output_name)
        
        # Ensure the output filename is unique
        counter = 1
        while os.path.exists(output_path):
            output_name = f"{base_name}_augmented_{counter}.csv"
            output_path = os.path.join(base_dir, output_name)
            counter += 1
        
        return output_path
    
    def _get_landmarks_array(self, row) -> np.ndarray:
        """
        Extract landmarks from a row and convert to a numpy array.
        
        Args:
            row: DataFrame row containing landmark data
            
        Returns:
            np.ndarray: Landmark coordinates as a numpy array shaped (num_landmarks, 3)
        """
        landmarks = []
        for i in range(self.num_landmarks):
            x = row[f'x{i}']
            y = row[f'y{i}']
            z = row[f'z{i}']
            landmarks.append([x, y, z])
        
        return np.array(landmarks)
    
    def _update_row_with_landmarks(self, row, landmarks: np.ndarray) -> pd.Series:
        """
        Update a row with new landmark coordinates.
        
        Args:
            row: Original DataFrame row
            landmarks: Modified landmark coordinates
            
        Returns:
            pd.Series: Updated row with new landmark coordinates
        """
        new_row = row.copy()
        for i in range(self.num_landmarks):
            new_row[f'x{i}'] = landmarks[i, 0]
            new_row[f'y{i}'] = landmarks[i, 1]
            new_row[f'z{i}'] = landmarks[i, 2]
        
        return new_row
    
    def rotate(self, angle_range: Tuple[float, float] = (-30, 30)) -> 'LandmarkAugmenter':
        """
        Rotate landmarks around the center point.
        
        Args:
            angle_range: Range of rotation angles in degrees (default: -30 to 30)
            
        Returns:
            self: For method chaining
        """
        augmented_rows = []
        
        for _, row in self.data.iterrows():
            angle = random.uniform(angle_range[0], angle_range[1])
            angle_rad = np.radians(angle)
            
            landmarks = self._get_landmarks_array(row)
            
            # Calculate center point
            center = landmarks.mean(axis=0)
            
            # Create rotation matrix (for x-y plane)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            
            # Apply rotation
            centered_landmarks = landmarks - center
            rotated_landmarks = np.dot(centered_landmarks, rotation_matrix.T)
            new_landmarks = rotated_landmarks + center
            
            # Create new sample
            new_row = self._update_row_with_landmarks(row, new_landmarks)
            new_row['label'] = f"{row['label']}_rot{angle:.1f}"
            
            augmented_rows.append(new_row)
        
        # Add augmented data
        self.data = pd.concat([self.data, pd.DataFrame(augmented_rows)], ignore_index=True)
        return self
    
    def scale(self, scale_range: Tuple[float, float] = (0.8, 1.2)) -> 'LandmarkAugmenter':
        """
        Scale landmarks around the center point.
        
        Args:
            scale_range: Range of scaling factors (default: 0.8 to 1.2)
            
        Returns:
            self: For method chaining
        """
        augmented_rows = []
        
        for _, row in self.data.iterrows():
            scale_factor = random.uniform(scale_range[0], scale_range[1])
            
            landmarks = self._get_landmarks_array(row)
            
            # Calculate center point
            center = landmarks.mean(axis=0)
            
            # Apply scaling
            centered_landmarks = landmarks - center
            scaled_landmarks = centered_landmarks * scale_factor
            new_landmarks = scaled_landmarks + center
            
            # Create new sample
            new_row = self._update_row_with_landmarks(row, new_landmarks)
            new_row['label'] = f"{row['label']}_scale{scale_factor:.2f}"
            
            augmented_rows.append(new_row)
        
        # Add augmented data
        self.data = pd.concat([self.data, pd.DataFrame(augmented_rows)], ignore_index=True)
        return self
    
    def translate(self, translation_range: Tuple[float, float] = (-0.1, 0.1)) -> 'LandmarkAugmenter':
        """
        Translate landmarks by a random offset.
        
        Args:
            translation_range: Range of translation values (default: -0.1 to 0.1)
            
        Returns:
            self: For method chaining
        """
        augmented_rows = []
        
        for _, row in self.data.iterrows():
            tx = random.uniform(translation_range[0], translation_range[1])
            ty = random.uniform(translation_range[0], translation_range[1])
            
            landmarks = self._get_landmarks_array(row)
            
            # Apply translation (x and y only)
            translation = np.array([tx, ty, 0])
            new_landmarks = landmarks + translation
            
            # Create new sample
            new_row = self._update_row_with_landmarks(row, new_landmarks)
            new_row['label'] = f"{row['label']}_trans{tx:.2f}_{ty:.2f}"
            
            augmented_rows.append(new_row)
        
        # Add augmented data
        self.data = pd.concat([self.data, pd.DataFrame(augmented_rows)], ignore_index=True)
        return self
    
    def add_noise(self, noise_level: float = 0.01) -> 'LandmarkAugmenter':
        """
        Add random noise to landmarks.
        
        Args:
            noise_level: Standard deviation of noise (default: 0.01)
            
        Returns:
            self: For method chaining
        """
        augmented_rows = []
        
        for _, row in self.data.iterrows():
            landmarks = self._get_landmarks_array(row)
            
            # Add random noise
            noise = np.random.normal(0, noise_level, landmarks.shape)
            new_landmarks = landmarks + noise
            
            # Create new sample
            new_row = self._update_row_with_landmarks(row, new_landmarks)
            new_row['label'] = f"{row['label']}_noise{noise_level}"
            
            augmented_rows.append(new_row)
        
        # Add augmented data
        self.data = pd.concat([self.data, pd.DataFrame(augmented_rows)], ignore_index=True)
        return self
    
    def flip_horizontal(self) -> 'LandmarkAugmenter':
        """
        Flip landmarks horizontally (around y-axis).
        
        Returns:
            self: For method chaining
        """
        augmented_rows = []
        
        for _, row in self.data.iterrows():
            landmarks = self._get_landmarks_array(row)
            
            # Flip horizontally (negate x coordinates)
            flipped_landmarks = landmarks.copy()
            flipped_landmarks[:, 0] = -flipped_landmarks[:, 0]
            
            # Create new sample
            new_row = self._update_row_with_landmarks(row, flipped_landmarks)
            new_row['label'] = f"{row['label']}_flipped"
            
            augmented_rows.append(new_row)
        
        # Add augmented data
        self.data = pd.concat([self.data, pd.DataFrame(augmented_rows)], ignore_index=True)
        return self
    
    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the augmented dataset to a CSV file.
        
        Args:
            output_path: Path to save the augmented dataset (default: None, uses self.output_csv_path)
            
        Returns:
            str: Path to the saved CSV file
        """
        output_path = output_path or self.output_csv_path
        ensure_dir(os.path.dirname(output_path))
        self.data.to_csv(output_path, index=False)
        print(f"Augmented dataset saved to {output_path} with {len(self.data)} samples")
        return output_path

def augment_dataset(
    input_csv: str,
    output_csv: Optional[str] = None,
    rotation: bool = True,
    scaling: bool = True,
    translation: bool = True,
    add_noise: bool = True,
    flip: bool = True
) -> str:
    """
    Augment a landmark dataset with various transformations.
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Path to save the augmented dataset (default: None, auto-generates a unique name)
        rotation: Whether to apply rotation augmentation (default: True)
        scaling: Whether to apply scaling augmentation (default: True)
        translation: Whether to apply translation augmentation (default: True)
        add_noise: Whether to add noise augmentation (default: True)
        flip: Whether to apply horizontal flipping (default: True)
        
    Returns:
        str: Path to the augmented dataset
    """
    augmenter = LandmarkAugmenter(input_csv, output_csv)
    
    if rotation:
        augmenter.rotate()
    
    if scaling:
        augmenter.scale()
    
    if translation:
        augmenter.translate()
    
    if add_noise:
        augmenter.add_noise()
    
    if flip:
        augmenter.flip_horizontal()
    
    return augmenter.save()

# Example usage
if __name__ == "__main__":
    # Attach paths to the two input CSV files
    input_csv_1 = r"D:\Sem6\EDAI\New\dataset\landmarks_dataset.csv"
    input_csv_2 = r"D:\Sem6\EDAI\New\dataset\lkandmarks_dataset.csv"
    
    # Augment the first dataset
    augment_dataset(input_csv_1)
    
    # Augment the second dataset
    augment_dataset(input_csv_2)