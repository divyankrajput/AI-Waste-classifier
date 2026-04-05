import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_path):
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Build and train the model
    model = build_model()
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('waste_classifier.h5', save_best_only=True)
        ]
    )

    # Save the model
    model.save('waste_classifier.h5')
    print("Model trained and saved as 'waste_classifier.h5'")

if __name__ == '__main__':
    # Replace 'path/to/your/dataset' with the actual path to your dataset
    # The dataset should have subdirectories: 'Dry Waste', 'Wet Waste', 'Recyclable Waste'
    dataset_path = 'dataset'  # Update this path
    if os.path.exists(dataset_path):
        train_model(dataset_path)
    else:
        print(f"Dataset path '{dataset_path}' not found. Please update the path in train.py")
