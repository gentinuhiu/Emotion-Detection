# data_preprocessing.py

import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    pixels = data['pixels'].tolist()
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(48, 48)
        faces.append(face)
    
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)  # Add channel dimension
    faces = faces / 255.0  # Normalize pixel values
    
    emotions = pd.get_dummies(data['emotion']).values
    
    return faces, emotions

if __name__ == "__main__":
    faces, emotions = load_and_preprocess_data('fer2013.csv')
    np.save('faces.npy', faces)
    np.save('emotions.npy', emotions)
