# train_model.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 classes for 7 emotions

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    faces = np.load('faces.npy')
    emotions = np.load('emotions.npy')

    X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

    model = create_model()
    history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

    model.save('emotion_detection_model.h5')
