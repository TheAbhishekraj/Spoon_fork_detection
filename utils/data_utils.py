import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random

class DataPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        img = cv2.resize(img, self.target_size)

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        return img

    def augment_data(self, data_dir, output_dir, augmentation_factor=5):
        """Augment existing images to increase dataset size"""
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.target_size)
                    img = np.expand_dims(img, axis=0)

                    # Generate augmented images
                    count = 0
                    for batch in datagen.flow(img, batch_size=1):
                        if count >= augmentation_factor:
                            break

                        aug_img = batch[0].astype(np.uint8)
                        output_path = os.path.join(
                            output_class_dir, 
                            f"{os.path.splitext(img_name)[0]}_aug_{count}.jpg"
                        )
                        cv2.imwrite(output_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                        count += 1

    def visualize_samples(self, data_dir, samples_per_class=5):
        """Visualize sample images from each class"""
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(15, 5*len(classes)))
        if len(classes) == 1:
            axes = [axes]

        for i, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sample_images = random.sample(images, min(samples_per_class, len(images)))

            for j, img_name in enumerate(sample_images):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                axes[i][j].imshow(img)
                axes[i][j].set_title(f"{class_name}: {img_name}")
                axes[i][j].axis('off')

        plt.tight_layout()
        plt.savefig('results/sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_sample_dataset():
    """Create sample images for demonstration"""
    print("Creating sample dataset...")

    # This is a placeholder function
    # In a real project, you would collect actual images
    print("Note: Add real spoon and fork images to data/spoon/ and data/fork/ directories")
    print("Each directory should contain at least 20-30 images for proper training")

    # Create placeholder files to show structure
    sample_files = [
        'data/spoon/spoon_001.jpg',
        'data/spoon/spoon_002.jpg', 
        'data/spoon/spoon_003.jpg',
        'data/fork/fork_001.jpg',
        'data/fork/fork_002.jpg',
        'data/fork/fork_003.jpg'
    ]

    for file_path in sample_files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create empty placeholder files
        with open(file_path, 'w') as f:
            f.write("# Placeholder for actual image file")

if __name__ == "__main__":
    create_sample_dataset()
