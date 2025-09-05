import tensorflow as tf
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

class SpoonForkPredictor:
    def __init__(self, model_path, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['fork', 'spoon']  # Alphabetical order

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image_path):
        """Make prediction on a single image"""
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img, verbose=0)

        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]

        return predicted_class, confidence

    def predict_batch(self, image_folder):
        """Make predictions on all images in a folder"""
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_folder, filename)
                try:
                    predicted_class, confidence = self.predict(image_path)
                    results.append({
                        'filename': filename,
                        'predicted_class': predicted_class,
                        'confidence': float(confidence)
                    })
                    print(f"{filename}: {predicted_class} (confidence: {confidence:.2%})")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        return results

def main():
    parser = argparse.ArgumentParser(description='Predict spoon or fork from images')
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--image', help='Path to a single image')
    parser.add_argument('--folder', help='Path to folder containing images')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    predictor = SpoonForkPredictor(args.model)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return

        predicted_class, confidence = predictor.predict(args.image)
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")

    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return

        results = predictor.predict_batch(args.folder)
        print(f"\nProcessed {len(results)} images")

    else:
        print("Please provide either --image or --folder argument")

if __name__ == "__main__":
    main()
