#!/usr/bin/env python3
"""
Demo script for Spoon Fork Detection Project
This script demonstrates the basic usage of the project.
"""

import os
import sys

def demo():
    print("🥄🍴 Spoon Fork Detection Project Demo")
    print("=" * 40)

    print("\n1. Project Structure:")
    print("   📁 data/spoon/     - Place spoon images here")
    print("   📁 data/fork/      - Place fork images here")
    print("   📁 src/            - Python training & prediction scripts")
    print("   📁 notebooks/      - Jupyter notebook for interactive training")
    print("   📁 models/         - Saved trained models")
    print("   📁 results/        - Training results and plots")

    print("\n2. Quick Start Commands:")
    print("   # Install dependencies")
    print("   pip install -r requirements.txt")
    print("")
    print("   # Train model (command line)")
    print("   cd src && python train_model.py")
    print("")
    print("   # Train model (interactive)")
    print("   jupyter notebook notebooks/train_model.ipynb")
    print("")
    print("   # Make predictions")
    print("   cd src && python predict.py --model ../models/spoon_fork_detector.h5 --image test_image.jpg")

    print("\n3. Expected Performance:")
    print("   🎯 Target Accuracy: >95%")
    print("   ⚡ Training Time: ~10-20 minutes (depends on data size)")
    print("   🔮 Inference Time: <100ms per image")

    print("\n4. Model Features:")
    print("   • Deep CNN with 4 convolutional blocks")
    print("   • Batch normalization for stable training")
    print("   • Dropout layers to prevent overfitting")
    print("   • Data augmentation for better generalization")
    print("   • Early stopping to prevent overtraining")

    print("\n5. Data Requirements:")
    print("   • Minimum: 20-30 images per class")
    print("   • Recommended: 50+ images per class")
    print("   • Format: JPG, JPEG, PNG")
    print("   • Resolution: 224x224 or higher")

    print("\n🚀 Ready to start? Add your images and run the training!")

if __name__ == "__main__":
    demo()
