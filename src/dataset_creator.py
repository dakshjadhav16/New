import os
import cv2
import csv
import time
import numpy as np
import mediapipe as mp
from utils import create_folders_if_not_exist

class SignLanguageDatasetCreator:
    def __init__(self, output_dir="dataset", csv_filename="lkandmarks_dataset.csv"):
        """
        Initialize the dataset creator
        
        Args:
            output_dir (str): Directory to save the dataset
            csv_filename (str): Name of the CSV file to store landmarks
        """
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "raw_images")
        self.csv_path = os.path.join(output_dir, csv_filename)
        
        # Create necessary directories
        create_folders_if_not_exist([output_dir, self.image_dir])
        
        # Initialize MediaPipe hands model
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create CSV file with headers
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Create headers for the CSV file
                headers = ['label']
                for i in range(21):  # MediaPipe has 21 hand landmarks
                    headers.extend([f'x{i}', f'y{i}', f'z{i}'])
                
                writer.writerow(headers)
    
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from an image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: Landmarks as a list, or None if no hand detected
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                
            return landmarks, hand_landmarks
        
        return None, None
    
    def save_landmarks_to_csv(self, label, landmarks):
        """
        Save landmarks to the CSV file
        
        Args:
            label (str): Class label (e.g., 'A', 'B', etc.)
            landmarks (list): Extracted landmarks
        """
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Prepare row: label followed by landmarks
            row = [label] + landmarks
            
            # Write to CSV
            writer.writerow(row)
    
    def capture_dataset(self):
        """
        Capture dataset using webcam
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Check if webcam is opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Initialize variables
        current_label = None
        collect_data = False
        frames_collected = 0
        total_samples = 0
        
        print("\n=== Sign Language Dataset Capture ===")
        print("Press a letter (A-Z) to set the current class")
        print("Press SPACE to start/stop collecting data")
        print("Press ESC to exit")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks_data, hand_landmarks = self.extract_landmarks(frame)
            
            # Create a copy for visualization
            display_frame = frame.copy()
            
            # Draw landmarks if detected
            if hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    display_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # Display current status
            status_text = f"Class: {current_label if current_label else 'Not set'}"
            cv2.putText(display_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if collect_data:
                # Collecting status
                cv2.putText(display_frame, "Collecting: YES", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Samples: {frames_collected}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Collect data every few frames
                if landmarks_data and frames_collected % 3 == 0:  # Collect every 3rd frame to vary poses
                    # Save landmarks to CSV
                    self.save_landmarks_to_csv(current_label, landmarks_data)
                    
                    # Save image
                    label_dir = os.path.join(self.image_dir, current_label)
                    create_folders_if_not_exist([label_dir])
                    
                    img_path = os.path.join(label_dir, f"{current_label}_{total_samples}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                    total_samples += 1
                
                frames_collected += 1
            else:
                # Not collecting status
                cv2.putText(display_frame, "Collecting: NO", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show total collected samples
            cv2.putText(display_frame, f"Total samples: {total_samples}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            
            # Display the frame
            cv2.imshow('Sign Language Dataset Capture', display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Exit if ESC is pressed
            if key == 27:  # ESC key
                break
            
            # Toggle data collection with SPACE
            elif key == 32:  # SPACE key
                if current_label:
                    collect_data = not collect_data
                    frames_collected = 0
                    print(f"{'Started' if collect_data else 'Stopped'} collecting data for class {current_label}")
                else:
                    print("Please set a class label first (A-Z)")
            
            # Set current label with letter keys (A-Z)
            elif 97 <= key <= 122:  # ASCII for lowercase a-z
                current_label = chr(key - 32)  # Convert to uppercase
                collect_data = False
                frames_collected = 0
                print(f"Current class set to: {current_label}")
                
                # Create directory for this label
                label_dir = os.path.join(self.image_dir, current_label)
                create_folders_if_not_exist([label_dir])
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nDataset collection complete! {total_samples} samples collected.")
        print(f"CSV file saved to: {self.csv_path}")
        print(f"Images saved to: {self.image_dir}")


# Example usage
if __name__ == "__main__":
    creator = SignLanguageDatasetCreator()
    creator.capture_dataset()