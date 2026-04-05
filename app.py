import os
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import random

app = Flask(__name__)

# Define categories and resource mappings
CATEGORIES = ['Dry Waste', 'Wet Waste', 'Recyclable Waste', 'Mixed Waste']
RESOURCE_MAPPING = {
    'Dry Waste': 'Bricks',
    'Wet Waste': 'Electricity',
    'Recyclable Waste': 'Recyclable',
    'Mixed Waste': 'Mixed'
}

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
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image):
    image = image.resize((224, 224))
    if NUMPY_AVAILABLE:
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
    else:
        # Dummy preprocessing for testing
        image = [[[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(224)]]  # Dummy 4D array
    return image

def load_model():
    if os.path.exists('waste_classifier.h5'):
        return tf.keras.models.load_model('waste_classifier.h5')
    else:
        # If model doesn't exist, return None and train later
        return None

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_waste():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        if model is None:
            # Dummy prediction for testing
            import random
            predicted_class = random.randint(0, 3)
            confidence = random.uniform(50, 95)
        else:
            predictions = model.predict(processed_image)[0]
            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class]) * 100

        category = CATEGORIES[predicted_class]
        resource = RESOURCE_MAPPING[category]

        return jsonify({
            'waste_category': category,
            'confidence_percentage': round(confidence, 2),
            'resource_usage': resource
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    # This is a placeholder for training. In a real scenario, you'd load the dataset here.
    # For now, we'll create a dummy model
    global model
    model = build_model()

    # Dummy training data (replace with actual dataset loading)
    # Assuming you have a dataset with subdirectories for each category
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Placeholder paths - replace with actual dataset paths
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',  # Replace with actual path
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'dataset/train',  # Replace with actual path
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save('waste_classifier.h5')

    return jsonify({'message': 'Model trained and saved successfully'})

if __name__ == '__main__':
    app.run(debug=True)
