'''
Step 1: Import Additional Libraries
You'll need Tkinter for the GUI and PIL (Python Imaging Library) for handling the image within the Tkinter window.
'''

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from skimage.transform import resize
'''Step 2: Create a Function to Open and Display the Image
This function will be used to open an image file and display it on the GUI.
'''
top, bottom, left, right = 0, 0, 0, 0
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((500, 500))  # Resize for display
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(window, image=img)
        panel.image = img
        panel.pack()

        process_and_predict(file_path)
'''Step 3: Process and Predict Function
Modify the existing process_landmarks and classify functions to work with a static image.
'''
def classify(image):
    try:
        image_resized = resize(image, (64, 64, 3))  # Resize image to match your model's input size
        image_expanded = np.expand_dims(image_resized, axis=0)
        proba = model.predict(image_expanded)
        idx = np.argmax(proba)
        return asl_labels[idx]
    except Exception as e:
        print("Error during classification:", e)
        return "Unknown"

def process_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame
def process_and_predict(file_path):
    img = cv2.imread(file_path)
    img = cv2.flip(img, 1)
    roi = img[top:bottom, right:left]

    alpha = classify(roi)
    roi_with_landmarks = process_landmarks(roi.copy())

    img[top:bottom, right:left] = roi_with_landmarks
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(img, alpha, (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the processed image to a format that Tkinter can display
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img.thumbnail((500, 500))  # Resize for display
    img = ImageTk.PhotoImage(img)

    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack()
'''Step 4: Setting Up the GUI
Now, create the main window and add a button to open the image.
'''
window = tk.Tk()
window.title("ASL Alphabet Detection")

open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack()

window.mainloop()
