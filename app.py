import cv2
import numpy as np
import mediapipe as mp
import keras
from skimage.transform import resize

# Path to your trained model
model_path = "C:\\Users\\LIKHITH\\OneDrive\\Desktop\\asl project\\asl-model.h5"  # Replace with the actual path to your model

# Load your trained model
model = keras.models.load_model(model_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ASL Labels (adjust according to your model's output)
asl_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def classify(image):
    image = resize(image, (64, 64, 3))  # Resize image to match your model's input size
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    idx = np.argmax(proba)
    return asl_labels[idx]

def process_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    height, width, _ = img.shape
    top, right, bottom, left = 75, 350, 300, 590

    # Extract hand ROI
    roi = img[top:bottom, right:left]
    
    # Classify the gesture on the original ROI
    alpha = classify(roi)

    # Process the landmarks on the ROI for display
    roi_with_landmarks = process_landmarks(roi.copy())

    # Place the processed ROI back into the frame
    img[top:bottom, right:left] = roi_with_landmarks

    # Draw ROI and put classified text
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, alpha, (0, 130), font, 1, (0, 0, 255), 2)

    # Show images
    cv2.imshow('ASL Detection', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
