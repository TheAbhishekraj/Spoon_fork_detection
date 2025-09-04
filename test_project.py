import unittest
import os
import sys
import tempfile
import numpy as np

# Add src directory to path
sys.path.append('src')
sys.path.append('utils')

from train_model import SpoonForkDetector
from data_utils import DataPreprocessor

class TestSpoonForkDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SpoonForkDetector(img_height=224, img_width=224)

    def test_model_creation(self):
        """Test if model can be created successfully"""
        model = self.detector.create_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 2))

    def test_model_compilation(self):
        """Test if model is properly compiled"""
        model = self.detector.create_model()
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam')
        self.assertEqual(model.loss, 'categorical_crossentropy')

    def test_data_preprocessor(self):
        """Test data preprocessor initialization"""
        preprocessor = DataPreprocessor(target_size=(224, 224))
        self.assertEqual(preprocessor.target_size, (224, 224))

class TestProjectStructure(unittest.TestCase):

    def test_directory_structure(self):
        """Test if all required directories exist"""
        required_dirs = [
            'data/spoon',
            'data/fork', 
            'models',
            'src',
            'utils',
            'notebooks',
            'results'
        ]

        for directory in required_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory {directory} does not exist")

    def test_required_files(self):
        """Test if all required files exist"""
        required_files = [
            'src/train_model.py',
            'src/predict.py',
            'utils/data_utils.py',
            'notebooks/train_model.ipynb',
            'README.md',
            'requirements.txt',
            'config.py'
        ]

        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")

if __name__ == '__main__':
    # Change to project directory
    if os.path.basename(os.getcwd()) != 'spoon_fork_detection':
        os.chdir('spoon_fork_detection')

    unittest.main()
