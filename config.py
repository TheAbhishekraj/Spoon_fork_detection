# Configuration file for Spoon Fork Detection Project

import os

class Config:
    # Data Configuration
    DATA_DIR = "data"
    SPOON_DIR = os.path.join(DATA_DIR, "spoon")
    FORK_DIR = os.path.join(DATA_DIR, "fork")

    # Model Configuration  
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2

    # Training Configuration
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.2

    # Model Saving
    MODEL_DIR = "models"
    MODEL_NAME = "spoon_fork_detector.h5"
    RESULTS_DIR = "results"

    # Augmentation Configuration
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True

    # Target Performance
    TARGET_ACCURACY = 0.95

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR, cls.SPOON_DIR, cls.FORK_DIR,
            cls.MODEL_DIR, cls.RESULTS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_model_path(cls):
        """Get full path to model file"""
        return os.path.join(cls.MODEL_DIR, cls.MODEL_NAME)
