# Import necessary libraries
import numpy as np  # For numerical operations and handling arrays
import cv2  # For image and video processing
from tensorflow.keras.models import load_model  # For loading the pre-trained model
import os  # For file operations

# Define the labels corresponding to the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect emotions in a given frame (image)
def detect_emotions_in_image(image_path, face_cascade, model):
    # Read the image from the specified path
    frame = cv2.imread(image_path)
    # Convert the frame to grayscale as face detection usually works on gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image using the face cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the face
        face = gray[y:y+h, x:x+w]
        # Resize the face ROI to 48x48 pixels as required by the model
        face = cv2.resize(face, (48, 48))
        # Normalize the pixel values to the range [0, 1]
        face = face / 255.0
        # Reshape the face array to add batch and channel dimensions
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict the emotion using the pre-trained model
        prediction = model.predict(face)
        # Get the emotion label with the highest probability
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Put the emotion label above the rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detected faces and emotion labels
    cv2.imshow('Emotion Detector - Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to detect emotions in a given video
def detect_emotions_in_video(video_path, face_cascade, model):
    # Capture video from the specified file
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        # If the frame was not captured successfully, break the loop
        if not ret:
            break

        # Convert the frame to grayscale as face detection usually works on gray images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image using the face cascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) corresponding to the face
            face = gray[y:y+h, x:x+w]
            # Resize the face ROI to 48x48 pixels as required by the model
            face = cv2.resize(face, (48, 48))
            # Normalize the pixel values to the range [0, 1]
            face = face / 255.0
            # Reshape the face array to add batch and channel dimensions
            face = np.reshape(face, (1, 48, 48, 1))

            # Predict the emotion using the pre-trained model
            prediction = model.predict(face)
            # Get the emotion label with the highest probability
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Put the emotion label above the rectangle
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame with detected faces and emotion labels
        cv2.imshow('Emotion Detector - Video', frame)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Main execution block
if __name__ == "__main__":
    # Load the pre-trained emotion detection model
    model = load_model('emotion_detection_model.h5')
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Choose the file type (image or video) for emotion detection
    file_type = input("Enter 'image' to upload an image or 'video' to upload a video: ").strip().lower()
    
    if file_type == 'image':
        # Get the image file path from the user
        image_path = input("Enter the path to the image file: ").strip()
        if os.path.exists(image_path):
            detect_emotions_in_image(image_path, face_cascade, model)
        else:
            print("Image file not found.")
    
    elif file_type == 'video':
        # Get the video file path from the user
        video_path = input("Enter the path to the video file: ").strip()
        if os.path.exists(video_path):
            detect_emotions_in_video(video_path, face_cascade, model)
        else:
            print("Video file not found.")
    
    else:
        print("Invalid input. Please enter 'image' or 'video'.")
