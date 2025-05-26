import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
import joblib
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import threading
import queue
import os
from datetime import datetime  # Added missing import
import serial
import time
from arduino_integration import ArduinoIntegration, integrate_arduino

class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition System")
        self.root.configure(bg="#2c3e50")
        
        # Set window size and position
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create a queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Initialize variables
        self.current_sign = None
        self.predicted_sign = None
        self.confidence = 0
        self.is_correct = False
        self.current_text = ""
        self.all_sentences = []
        self.active_model = "both"
        self.running = True
        
        # Load models
        self.load_models()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Define class labels
        self.class_labels_a_to_l = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        self.class_labels_m_to_z = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.all_class_labels = self.class_labels_a_to_l + self.class_labels_m_to_z
        
        # Initialize data structures to store test results
        self.results_data = {
            'actual': [],
            'predicted': [],
            'correct': [],
            'timestamp': [],
            'model_used': []
        }
        self.sign_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Create GUI components
        self.create_widgets()
        
        # Start video stream in a separate thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Check queue regularly
        self.root.after(10, self.check_queue)
    
    def load_models(self):
        try:
            self.rf_model_a_to_l = joblib.load("D:/Sem6/EDAI/New/random_forest_model.joblib")
            self.rf_model_m_to_z = joblib.load("D:/Sem6/EDAI/New/random_forest_model_m_to_z.joblib")
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.rf_model_a_to_l = None
            self.rf_model_m_to_z = None
    
    def create_widgets(self):
        # Create main frames
        self.top_frame = tk.Frame(self.root, bg="#34495e", pady=10)
        self.top_frame.pack(fill=tk.X)
        
        self.middle_frame = tk.Frame(self.root, bg="#2c3e50")
        self.middle_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.bottom_frame = tk.Frame(self.root, bg="#34495e", pady=10)
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Title label
        title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.title_label = tk.Label(self.top_frame, text="Sign Language Recognition", 
                                   font=title_font, bg="#34495e", fg="white")
        self.title_label.pack(pady=10)
        
        # Create left and right panels in middle frame
        self.left_panel = tk.Frame(self.middle_frame, bg="#2c3e50", width=600)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_panel.pack_propagate(False)
        
        self.right_panel = tk.Frame(self.middle_frame, bg="#34495e", width=550)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Video display (left panel)
        self.video_label = tk.Label(self.left_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results display (right panel)
        # Create a styled frame for predicted character
        self.pred_char_frame = tk.Frame(self.right_panel, bg="#1abc9c", bd=2, relief=tk.RAISED)
        self.pred_char_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        pred_label_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.pred_char_title = tk.Label(self.pred_char_frame, text="Predicted Character", 
                                       font=pred_label_font, bg="#1abc9c", fg="white")
        self.pred_char_title.pack(pady=(10, 0))
        
        pred_char_font = font.Font(family="Helvetica", size=72, weight="bold")
        self.pred_char_label = tk.Label(self.pred_char_frame, text="?", font=pred_char_font, 
                                      bg="#1abc9c", fg="white", height=1, width=2)
        self.pred_char_label.pack(pady=(0, 10))
        
        # Create a styled frame for confidence
        self.confidence_frame = tk.Frame(self.right_panel, bg="#3498db", bd=2, relief=tk.RAISED)
        self.confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        confidence_label_font = font.Font(family="Helvetica", size=14)
        self.confidence_title = tk.Label(self.confidence_frame, text="Confidence", 
                                       font=confidence_label_font, bg="#3498db", fg="white")
        self.confidence_title.pack(pady=(10, 0))
        
        confidence_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.confidence_label = tk.Label(self.confidence_frame, text="0%", 
                                       font=confidence_font, bg="#3498db", fg="white")
        self.confidence_label.pack(pady=(0, 10))
        
        # Create a styled frame for formed word
        self.word_frame = tk.Frame(self.right_panel, bg="#9b59b6", bd=2, relief=tk.RAISED)
        self.word_frame.pack(fill=tk.X, padx=20, pady=10)
        
        word_title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.word_title = tk.Label(self.word_frame, text="Formed Word/Sentence", 
                                 font=word_title_font, bg="#9b59b6", fg="white")
        self.word_title.pack(pady=(10, 0))
        
        word_font = font.Font(family="Helvetica", size=20)
        self.word_label = tk.Label(self.word_frame, text="", font=word_font, 
                                 bg="#9b59b6", fg="white", wraplength=450)
        self.word_label.pack(pady=(5, 10), fill=tk.X)
        
        # Model selection
        self.model_frame = tk.Frame(self.right_panel, bg="#34495e")
        self.model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.model_label = tk.Label(self.model_frame, text="Active Model:", 
                                   font=("Helvetica", 12), bg="#34495e", fg="white")
        self.model_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_var = tk.StringVar(value="both")
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, 
                                         values=["model1 (A-L)", "model2 (M-Z)", "both (A-Z)"],
                                         width=15, state="readonly")
        self.model_combobox.pack(side=tk.LEFT)
        self.model_combobox.bind("<<ComboboxSelected>>", self.toggle_model)
        
        # Current sign being tested
        self.test_frame = tk.Frame(self.right_panel, bg="#34495e")
        self.test_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.test_label = tk.Label(self.test_frame, text="Testing Sign:", 
                                 font=("Helvetica", 12), bg="#34495e", fg="white")
        self.test_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_var = tk.StringVar(value="None")
        self.test_combobox = ttk.Combobox(self.test_frame, textvariable=self.test_var, 
                                        values=self.all_class_labels, width=5, state="readonly")
        self.test_combobox.pack(side=tk.LEFT)
        self.test_combobox.bind("<<ComboboxSelected>>", self.set_current_sign)
        
        # Control buttons
        self.button_frame = tk.Frame(self.bottom_frame, bg="#34495e")
        self.button_frame.pack(pady=10)
        
        self.add_btn = tk.Button(self.button_frame, text="Add Letter (Space)", 
                              command=self.add_letter, bg="#2ecc71", fg="white",
                              font=("Helvetica", 12), padx=10)
        self.add_btn.grid(row=0, column=0, padx=10)
        
        self.space_btn = tk.Button(self.button_frame, text="Add Space (Enter)", 
                                command=self.add_space, bg="#3498db", fg="white",
                                font=("Helvetica", 12), padx=10)
        self.space_btn.grid(row=0, column=1, padx=10)
        
        self.backspace_btn = tk.Button(self.button_frame, text="Delete (Backspace)", 
                                    command=self.delete_last, bg="#e74c3c", fg="white",
                                    font=("Helvetica", 12), padx=10)
        self.backspace_btn.grid(row=0, column=2, padx=10)
        
        self.clear_btn = tk.Button(self.button_frame, text="Clear All", 
                                command=self.clear_text, bg="#f39c12", fg="white",
                                font=("Helvetica", 12), padx=10)
        self.clear_btn.grid(row=0, column=3, padx=10)
        
        self.report_btn = tk.Button(self.button_frame, text="Generate Report", 
                                 command=self.generate_report, bg="#9b59b6", fg="white",
                                 font=("Helvetica", 12), padx=10)
        self.report_btn.grid(row=0, column=4, padx=10)

        self.save_btn = tk.Button(self.button_frame, text="Save Text", 
                               command=self.save_text, bg="#16a085", fg="white",
                               font=("Helvetica", 12), padx=10)
        self.save_btn.grid(row=0, column=5, padx=10)
        
        # Bind keyboard shortcuts
        self.root.bind("<space>", lambda event: self.add_letter())
        self.root.bind("<Return>", lambda event: self.add_space())
        self.root.bind("<BackSpace>", lambda event: self.delete_last())
        self.root.bind("s", lambda event: self.save_text() if event.state & 0x4 else None)  # Ctrl+S
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#34495e", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_model(self, event=None):
        selection = self.model_var.get()
        if "model1" in selection:
            self.active_model = "model1"
        elif "model2" in selection:
            self.active_model = "model2"
        else:
            self.active_model = "both"
        self.status_var.set(f"Active model set to: {selection}")
    
    def set_current_sign(self, event=None):
        self.current_sign = self.test_var.get()
        self.status_var.set(f"Now testing sign: {self.current_sign}")
    
    def add_letter(self):
        if self.predicted_sign:
            self.current_text += self.predicted_sign
            self.word_label.config(text=self.current_text)
            self.status_var.set(f"Added letter: {self.predicted_sign}")
    
    def add_space(self):
        if self.current_text:
            self.current_text += " "
            self.word_label.config(text=self.current_text)
            self.status_var.set("Added space")
    
    def delete_last(self):
        if self.current_text:
            self.current_text = self.current_text[:-1]
            self.word_label.config(text=self.current_text)
            self.status_var.set("Deleted last character")
    
    def clear_text(self):
        if self.current_text:
            self.all_sentences.append(self.current_text)
        self.current_text = ""
        self.word_label.config(text="")
        self.status_var.set("Text cleared and saved to sentences")
    
    def save_text(self):
        """Save all text including current text to a file"""
        # First add current text to sentences if it exists
        if self.current_text:
            self.all_sentences.append(self.current_text)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sign_language_text_{timestamp}.txt"
        
        # Write all sentences and current text to file
        try:
            with open(filename, 'w') as f:
                for sentence in self.all_sentences:
                    f.write(sentence + '\n')
                if self.current_text:
                    f.write(self.current_text)
            
            self.status_var.set(f"Text saved to {filename}")
        except Exception as e:
            self.status_var.set(f"Error saving text: {e}")
    
    def extract_features(self, hand_landmarks):
        # Convert landmarks to feature vector
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Ensure we have exactly 63 features (as in your training data)
        if len(features) > 63:
            features = features[:63]
        elif len(features) < 63:
            features = np.pad(features, (0, 63 - len(features)), 'constant')
        
        return np.array(features)
    
    def predict_sign(self, features):
        # Check if models are loaded
        if self.rf_model_a_to_l is None or self.rf_model_m_to_z is None:
            return "Models not loaded", 0, None, None, 0
            
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Variables to store predictions
        predicted_sign1 = None
        predicted_sign2 = None
        confidence1 = 0
        confidence2 = 0
        
        # Get prediction and confidence from first model (A-L)
        if self.active_model in ["model1", "both"]:
            prediction1 = self.rf_model_a_to_l.predict(features)[0]
            probabilities1 = self.rf_model_a_to_l.predict_proba(features)[0]
            confidence1 = np.max(probabilities1) * 100
            if 0 <= prediction1 < len(self.class_labels_a_to_l):
                predicted_sign1 = self.class_labels_a_to_l[prediction1]
        
        # Get prediction and confidence from second model (M-Z)
        if self.active_model in ["model2", "both"]:
            prediction2 = self.rf_model_m_to_z.predict(features)[0]
            probabilities2 = self.rf_model_m_to_z.predict_proba(features)[0]
            confidence2 = np.max(probabilities2) * 100
            if 0 <= prediction2 < len(self.class_labels_m_to_z):
                predicted_sign2 = self.class_labels_m_to_z[prediction2]
        
        # If using both models, choose the one with higher confidence
        if self.active_model == "both" and predicted_sign1 and predicted_sign2:
            if confidence1 >= confidence2:
                return predicted_sign1, confidence1, "model1", predicted_sign2, confidence2
            else:
                return predicted_sign2, confidence2, "model2", predicted_sign1, confidence1
        elif predicted_sign1:
            return predicted_sign1, confidence1, "model1", None, 0
        elif predicted_sign2:
            return predicted_sign2, confidence2, "model2", None, 0
        else:
            return "Unknown", 0, None, None, 0
    
    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for more intuitive view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Add basic info to frame
            cv2.putText(frame, "Sign Language Recognition", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            hand_detected = False
            
            # Draw hand landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract features and make prediction
                    features = self.extract_features(hand_landmarks)
                    
                    # Get prediction using appropriate model(s)
                    self.predicted_sign, self.confidence, model_used, alt_sign, alt_confidence = self.predict_sign(features)
                    
                    # Record results if a sign is currently being tested
                    if self.current_sign:
                        self.is_correct = (self.predicted_sign == self.current_sign)
                        self.results_data['actual'].append(self.current_sign)
                        self.results_data['predicted'].append(self.predicted_sign)
                        self.results_data['correct'].append(self.is_correct)
                        self.results_data['timestamp'].append(time.time())
                        self.results_data['model_used'].append(model_used)
                        
                        # Update sign performance data
                        self.sign_performance[self.current_sign]['total'] += 1
                        if self.is_correct:
                            self.sign_performance[self.current_sign]['correct'] += 1
                        
                        # Display whether prediction was correct
                        result_text = "CORRECT" if self.is_correct else "INCORRECT"
                        result_color = (0, 255, 0) if self.is_correct else (0, 0, 255)
                        cv2.putText(frame, result_text, (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                    
                    # Put the results in the queue for the main thread to process
                    self.queue.put((self.predicted_sign, self.confidence))
            
            if not hand_detected:
                cv2.putText(frame, "No hand detected", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Convert frame to PhotoImage for display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (580, 440))
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Put the image in the queue
            self.queue.put(("IMAGE", imgtk))
    
    def check_queue(self):
        try:
            while not self.queue.empty():
                item = self.queue.get(block=False)
                
                if item[0] == "IMAGE":
                    # Update video display
                    self.video_label.configure(image=item[1])
                    self.video_label.image = item[1]  # Keep a reference
                else:
                    # Update prediction display
                    pred_sign, confidence = item
                    
                    # Update predicted character
                    if pred_sign and pred_sign != "Unknown":
                        self.pred_char_label.config(text=pred_sign)
                        
                        # Change color based on confidence
                        if confidence >= 80:
                            self.pred_char_frame.config(bg="#2ecc71")  # Green for high confidence
                            self.pred_char_title.config(bg="#2ecc71")
                            self.pred_char_label.config(bg="#2ecc71")
                        elif confidence >= 50:
                            self.pred_char_frame.config(bg="#f39c12")  # Orange for medium confidence
                            self.pred_char_title.config(bg="#f39c12")
                            self.pred_char_label.config(bg="#f39c12")
                        else:
                            self.pred_char_frame.config(bg="#e74c3c")  # Red for low confidence
                            self.pred_char_title.config(bg="#e74c3c")
                            self.pred_char_label.config(bg="#e74c3c")
                    else:
                        self.pred_char_label.config(text="?")
                        self.pred_char_frame.config(bg="#1abc9c")
                        self.pred_char_title.config(bg="#1abc9c")
                        self.pred_char_label.config(bg="#1abc9c")
                    
                    # Update confidence label
                    self.confidence_label.config(text=f"{confidence:.1f}%")
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(10, self.check_queue)
    
    def generate_report(self):
        if not self.results_data['actual']:
            self.status_var.set("No data collected. Test signs first.")
            return
        
        # Create DataFrame from collected data
        results_df = pd.DataFrame(self.results_data)
        
        # Overall accuracy
        total_tests = len(results_df)
        correct_tests = sum(results_df['correct'])
        overall_accuracy = correct_tests / total_tests * 100 if total_tests > 0 else 0
        
        # Create visualization
        plt.figure(figsize=(20, 10))
        
        signs_tested = []
        accuracies = []
        
        for sign in self.all_class_labels:
            if self.sign_performance[sign]['total'] > 0:
                correct = self.sign_performance[sign]['correct']
                total = self.sign_performance[sign]['total']
                accuracy = correct / total * 100 if total > 0 else 0
                accuracies.append(accuracy)
                signs_tested.append(sign)
        
        if signs_tested:
            # Color bars by model range (A-L vs M-Z)
            colors = []
            for sign in signs_tested:
                if sign in self.class_labels_a_to_l:
                    colors.append('skyblue')
                else:
                    colors.append('lightgreen')
            
            bars = plt.bar(signs_tested, accuracies, color=colors)
            plt.xlabel('Sign')
            plt.ylabel('Recognition Accuracy (%)')
            plt.title('Sign Language Recognition Accuracy by Sign')
            plt.ylim(0, 105)  # Leave room for text
            plt.axhline(y=overall_accuracy, color='r', linestyle='-', label=f'Overall: {overall_accuracy:.2f}%')
            
            # Add legend for models
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='skyblue', label='Model 1 (A-L)'),
                Patch(facecolor='lightgreen', label='Model 2 (M-Z)'),
                Patch(facecolor='red', label=f'Overall: {overall_accuracy:.2f}%')
            ]
            plt.legend(handles=legend_elements)
            
            # Add values on top of bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{acc:.1f}%', ha='center', va='bottom')
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save chart
            plt.savefig(f'sign_language_results_{timestamp}.png')
            
            # Save detailed results to CSV
            results_df.to_csv(f'sign_language_results_{timestamp}.csv', index=False)
            
            # Always save the current text content
            self.save_text()
            
            self.status_var.set(f"Report generated: results saved with timestamp {timestamp}")
    
    def on_closing(self):
        """Handle cleanup when window is closed"""
        self.running = False
        
        # Save any unsaved text before closing
        self.save_text()
        
        # Release resources
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 