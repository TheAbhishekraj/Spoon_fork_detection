import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

class SpoonForkDetector:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None

    def create_model(self):
        """Create a CNN model for spoon/fork classification"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # 2 classes: spoon, fork
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def prepare_data(self, data_dir, validation_split=0.2, batch_size=32):
        """Prepare training and validation data"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )

        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_generator, validation_generator

    def train(self, train_generator, validation_generator, epochs=50):
        """Train the model"""
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
        ]

        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, test_generator):
        """Evaluate the model"""
        loss, accuracy = self.model.evaluate(test_generator, verbose=0)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")

        # Generate predictions for detailed metrics
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        # Classification report
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        print("Classification Report:")
        print(report)

        return accuracy, loss, report

    def predict_image(self, img_path):
        """Predict single image"""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img)
        class_names = ['fork', 'spoon']  # Alphabetical order
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        return predicted_class, confidence

    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Spoon Fork Detection Project")
    print("=" * 30)

    # Initialize detector
    detector = SpoonForkDetector(img_height=224, img_width=224)

    # Create model
    model = detector.create_model()
    print(f"Model created with {model.count_params():,} parameters")

    # Prepare data
    print("Preparing data...")
    data_dir = "data"
    train_gen, val_gen = detector.prepare_data(data_dir, validation_split=0.2, batch_size=32)

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")

    # Train model
    print("Starting training...")
    history = detector.train(train_gen, val_gen, epochs=50)

    # Evaluate model
    print("Evaluating model...")
    accuracy, loss, report = detector.evaluate(val_gen)

    # Save model if accuracy > 95%
    if accuracy > 0.95:
        detector.save_model("models/spoon_fork_detector.h5")
        print(f"🎉 Success! Model achieved {accuracy:.2%} accuracy (>95% target)")
    else:
        print(f"⚠️  Model achieved {accuracy:.2%} accuracy (below 95% target)")
        detector.save_model("models/spoon_fork_detector.h5")

    # Plot training history
    detector.plot_training_history()

    # Save training results
    results = {
        'final_accuracy': float(accuracy),
        'final_loss': float(loss),
        'epochs_trained': len(history.history['accuracy']),
        'classification_report': report
    }

    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Training completed! Check results/ folder for detailed metrics.")

if __name__ == "__main__":
    main()
