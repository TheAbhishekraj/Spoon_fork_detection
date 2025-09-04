#!/usr/bin/env python3
"""
Setup script for Spoon Fork Detection Project
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = [
        "data/spoon",
        "data/fork", 
        "models",
        "results"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ Directories created successfully!")

def check_tensorflow_gpu():
    """Check if TensorFlow can access GPU"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        if tf.config.list_physical_devices('GPU'):
            print("✅ GPU is available for TensorFlow")
        else:
            print("⚠️  GPU not available, will use CPU")

    except ImportError:
        print("❌ TensorFlow not installed")
        return False

    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Spoon Fork Detection Project")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

    # Create directories
    create_directories()

    # Install requirements
    if not install_requirements():
        sys.exit(1)

    # Check TensorFlow
    if not check_tensorflow_gpu():
        sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add spoon images to data/spoon/ directory")
    print("2. Add fork images to data/fork/ directory") 
    print("3. Run: python src/train_model.py")
    print("   or use: jupyter notebook notebooks/train_model.ipynb")

if __name__ == "__main__":
    main()
