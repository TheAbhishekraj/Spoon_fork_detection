import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import json

# REMARK: Always create output folders first.
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

class SpoonForkDetector:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None

    def create_model(self):
        """
        REMARK: Use MobileNetV2 pretrained on ImageNet for strong feature extraction, freeze for initial training.
        """
        # Load MobileNetV2 base
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', 
                                                       include_top=False, 
                                                       input_shape=(self.img_height, self.img_width, 3))
        base_model.trainable = False  # Freeze base layers

        # Add custom classifier on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # REMARK: For 2 classes: spoon, fork
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def prepare_data(self, data_dir, validation_split=0.2, batch_size=32):
        """
        REMARK: Use strong augmentation for robust training and generalization.
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            validation_split=validation_split
        )
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        return train_generator, validation_generator

    def train(self, train_generator, validation_generator, epochs=30):
        """
        REMARK: EarlyStopping prevents overfitting, ReduceLROnPlateau improves convergence.
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
        ]
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        # REMARK: Optionally, fine-tune base model for a few epochs after initial training
        self.model.layers[0].trainable = True  # Unfreeze base
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # Low LR for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.history_ft = self.model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        return self.history

    def evaluate(self, test_generator):
        """
        REMARK: Use predictions and classification report for analysis.
        """
        loss, accuracy = self.model.evaluate(test_generator, verbose=0)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        print("Classification Report:")
        print(report)
        return accuracy, loss, report

    def plot_training_history(self):
        """
        REMARK: Visualize training and validation accuracy/loss for better tuning.
        """
        if self.history is None:
            print("No training history available")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
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

    def save_model(self, filepath):
        """
        REMARK: Save using recommended Keras format.
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def main():
    print("Spoon Fork Detection Project")
    print("=" * 30)

    # REMARK: Always ensure output folders exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    detector = SpoonForkDetector(img_height=224, img_width=224)
    model = detector.create_model()
    print(f"Model created with {model.count_params():,} parameters")

    print("Preparing data...")
    data_dir = "data"  # REMARK: Points to parent folder of 'spoon/' and 'fork/' folders
    train_gen, val_gen = detector.prepare_data(data_dir, validation_split=0.2, batch_size=32)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")

    print("Starting training...")
    history = detector.train(train_gen, val_gen, epochs=30)

    print("Evaluating model...")
    accuracy, loss, report = detector.evaluate(val_gen)

    detector.save_model("models/spoon_fork_detector.keras")
    if accuracy > 0.8:
        print(f"🎉 Success! Model achieved {accuracy:.2%} accuracy (>=80% target)")
    else:
        print(f"⚠️  Model achieved {accuracy:.2%} accuracy (<80% target) - see training curves and data.")

    detector.plot_training_history()

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
