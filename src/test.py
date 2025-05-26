import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
import joblib
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. Load both trained models
rf_model_a_to_l = joblib.load('random_forest_model.joblib')
rf_model_m_to_z = joblib.load('random_forest_model_m_to_z.joblib')

# 2. Initialize MediaPipe for hand detection and landmark extraction
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 3. Define class labels for both models
class_labels_a_to_l = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
class_labels_m_to_z = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
all_class_labels = class_labels_a_to_l + class_labels_m_to_z

# 4. Create data structures to store test results
results_data = {
    'actual': [],
    'predicted': [],
    'correct': [],
    'timestamp': [],
    'model_used': []
}
sign_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
current_sign = None

# 5. Add text building variables
current_text = ""
all_sentences = []
display_text = ""
   # Variables for results
predicted_sign = None
confidence = 0
is_correct = False

# 6. Add model selection variable
active_model = "both"  # Options: "model1" (A-L), "model2" (M-Z), "both"

# 7. Function to extract features from hand landmarks
def extract_features(hand_landmarks):
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

# 8. Function to toggle between models
def toggle_model():
    global active_model
    if active_model == "model1":
        active_model = "model2"
        print("Switched to Model 2 (M-Z)")
    elif active_model == "model2":
        active_model = "both"
        print("Using both models")
    else:
        active_model = "model1"
        print("Switched to Model 1 (A-L)")

# 9. Function to make prediction using appropriate model
def predict_sign(features):
    # Reshape for prediction
    features = features.reshape(1, -1)
    
    # Variables to store predictions
    predicted_sign1 = None
    predicted_sign2 = None
    confidence1 = 0
    confidence2 = 0
    
    # Get prediction and confidence from first model (A-L)
    if active_model in ["model1", "both"]:
        prediction1 = rf_model_a_to_l.predict(features)[0]
        probabilities1 = rf_model_a_to_l.predict_proba(features)[0]
        confidence1 = np.max(probabilities1) * 100
        if 0 <= prediction1 < len(class_labels_a_to_l):
            predicted_sign1 = class_labels_a_to_l[prediction1]
    
    # Get prediction and confidence from second model (M-Z)
    if active_model in ["model2", "both"]:
        prediction2 = rf_model_m_to_z.predict(features)[0]
        probabilities2 = rf_model_m_to_z.predict_proba(features)[0]
        confidence2 = np.max(probabilities2) * 100
        if 0 <= prediction2 < len(class_labels_m_to_z):
            predicted_sign2 = class_labels_m_to_z[prediction2]
    
    # If using both models, choose the one with higher confidence
    if active_model == "both" and predicted_sign1 and predicted_sign2:
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

# 10. Function to generate report
def generate_report():
    if not results_data['actual']:
        print("No data collected. Test signs first.")
        return
    
    # Create DataFrame from collected data
    results_df = pd.DataFrame(results_data)
    
    # Overall accuracy
    total_tests = len(results_df)
    correct_tests = sum(results_df['correct'])
    overall_accuracy = correct_tests / total_tests * 100 if total_tests > 0 else 0
    
    # Print report
    print("\n===== Sign Language Recognition Test Report =====")
    print(f"Total tests conducted: {total_tests}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print("\nPerformance by sign:")
    print("-" * 50)
    print(f"{'Sign':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Model':<10}")
    print("-" * 50)
    
    accuracies = []
    signs_tested = []
    
    for sign in all_class_labels:
        if sign_performance[sign]['total'] > 0:
            correct = sign_performance[sign]['correct']
            total = sign_performance[sign]['total']
            accuracy = correct / total * 100 if total > 0 else 0
            accuracies.append(accuracy)
            signs_tested.append(sign)
            
            # Determine which model was used the most for this sign
            sign_results = results_df[results_df['actual'] == sign]
            model_counts = sign_results['model_used'].value_counts()
            most_used_model = model_counts.idxmax() if not model_counts.empty else "Unknown"
            
            print(f"{sign:<10} {correct:<10} {total:<10} {accuracy:.2f}%    {most_used_model}")
    
    # Create visualization
    if signs_tested:
        plt.figure(figsize=(20, 10))
        
        # Color bars by model range (A-L vs M-Z)
        colors = []
        for sign in signs_tested:
            if sign in class_labels_a_to_l:
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
        plt.savefig('sign_language_test_results.png')
        print("\nResults chart saved as 'sign_language_test_results.png'")
        plt.show()
    
    # Save detailed results to CSV
    results_df.to_csv('sign_language_test_results.csv', index=False)
    print("Detailed results saved as 'sign_language_test_results.csv'")

    # Save created sentences to a text file
    if all_sentences:
        with open('sign_language_sentences.txt', 'w') as f:
            for sentence in all_sentences:
                f.write(sentence + '\n')
        print("Created sentences saved as 'sign_language_sentences.txt'")

# 11. Initialize webcam
cap = cv2.VideoCapture(0)

print("=== Sign Language Recognition Testing ===")
print("Instructions:")
print("1. Press a letter key (A-Z) to set the current sign you're testing")
print("2. Show that sign to the camera")
print("3. Press SPACE to add the currently predicted letter to the text")
print("4. Press ENTER to complete the current word/sentence")
print("5. Press BACKSPACE to delete the last letter")
print("6. Press 'M' to toggle between models (A-L, M-Z, or both)")
print("7. Press 'R' to generate a report")
print("8. Press 'Q' to quit")

while cap.isOpened():
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for more intuitive view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe
    results = hands.process(rgb_frame)
    
    # Add instruction text
    cv2.putText(frame, "A-Z: Set sign, M: Toggle model, SPACE: Add letter, ENTER: New sentence", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display current sign being tested
    if current_sign:
        cv2.putText(frame, f"Current sign: {current_sign}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Display active model
    model_text = ""
    if active_model == "model1":
        model_text = "Model: A-L"
    elif active_model == "model2":
        model_text = "Model: M-Z"
    else:
        model_text = "Model: A-Z (both)"
    
    cv2.putText(frame, model_text, (frame.shape[1] - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    
    # Draw hand landmarks and make predictions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract features and make prediction
            features = extract_features(hand_landmarks)
            
            # Get prediction using appropriate model(s)
            predicted_sign, confidence, model_used, alt_sign, alt_confidence = predict_sign(features)
            
            # Display prediction and confidence
            cv2.putText(frame, f"Predicted: {predicted_sign} ({confidence:.1f}%)", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # If using both models, display alternative prediction
            if alt_sign:
                cv2.putText(frame, f"Alt Pred: {alt_sign} ({alt_confidence:.1f}%)", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                y_pos = 150
            else:
                y_pos = 120
            
            # Record results if a sign is currently being tested
            if current_sign:
                is_correct = (predicted_sign == current_sign)
                results_data['actual'].append(current_sign)
                results_data['predicted'].append(predicted_sign)
                results_data['correct'].append(is_correct)
                results_data['timestamp'].append(time.time())
                results_data['model_used'].append(model_used)
                
                # Update sign performance data
                sign_performance[current_sign]['total'] += 1
                if is_correct:
                    sign_performance[current_sign]['correct'] += 1
                
                # Display whether prediction was correct
                result_text = "CORRECT" if is_correct else "INCORRECT"
                result_color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(frame, result_text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                y_pos += 30
    else:
        cv2.putText(frame, "No hand detected", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display current text and sentences
    text_y_pos = frame.shape[0] - 60  # Position at bottom of frame
    cv2.putText(frame, f"Current word: {current_text}", (10, text_y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the last 2 sentences (or as many as available)
    if all_sentences:
        display_sentences = all_sentences[-2:] if len(all_sentences) > 1 else all_sentences
        y_offset = text_y_pos + 30
        for sentence in display_sentences:
            # Truncate long sentences to fit screen
            max_length = 50
            disp_sentence = sentence if len(sentence) <= max_length else sentence[:max_length] + "..."
            cv2.putText(frame, disp_sentence, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_offset += 25
    
    # Display the frame
    cv2.imshow('Sign Language Recognition Testing', frame)
    
    # Check for key press
# Check for key press
    key = cv2.waitKey(1) & 0xFF

# Process key press
    if key == ord('q'):
        break
    elif key == ord('r'):
        generate_report()
    elif key == ord('m'):
        toggle_model()
    elif 97 <= key <= 122 or 65 <= key <= 90:  # a-z or A-Z
        # Set current sign based on key press
        current_sign = chr(key).upper()
        print(f"Now testing sign: {current_sign}")
    elif key == 32:  # Space - add predicted letter to text if available
        if results.multi_hand_landmarks and predicted_sign:
            current_text += predicted_sign
            print(f"Added letter: {predicted_sign}, Current word: {current_text}")
    elif key == 13:  # Enter - complete current sentence
        if current_text:
            # Instead of adding to all_sentences, just add a space to continue the sentence
            current_text += " "
            print(f"Added space. Current sentence: {current_text}")
    elif key == 8:  # Backspace - delete last character
        if current_text:
            current_text = current_text[:-1]
            print(f"Deleted last character. Current word: {current_text}")
# These lines should be OUTSIDE the main loop (notice the indentation)
# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Generate final report if there's data
if results_data['actual']:
    generate_report()

print("Program ended")