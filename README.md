# Spoon Fork Detection Project

A deep learning project for classifying spoons and forks with >95% accuracy using TensorFlow and Keras.

## Project Structure

```
spoon_fork_detection/
├── data/
│   ├── spoon/          # Spoon images
│   └── fork/           # Fork images
├── src/
│   ├── train_model.py  # Main training script
│   └── predict.py      # Prediction script
├── utils/
│   └── data_utils.py   # Data preprocessing utilities
├── notebooks/
│   └── train_model.ipynb # Interactive training notebook
├── models/             # Saved trained models
├── results/            # Training results and visualizations
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Features

- **High Accuracy**: Designed to achieve >95% classification accuracy
- **Data Augmentation**: Built-in image augmentation for better generalization  
- **Robust Architecture**: CNN with batch normalization and dropout for stability
- **Easy to Use**: Simple command-line interface for training and prediction
- **Interactive Notebook**: Jupyter notebook for step-by-step training
- **Comprehensive Logging**: Detailed training metrics and visualizations

## Quick Start

### 1. Setup Environment

```bash
# Clone or download this project
# Navigate to project directory
cd spoon_fork_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

- Add spoon images to `data/spoon/` directory
- Add fork images to `data/fork/` directory
- Recommended: At least 30-50 images per class for good results
- Supported formats: JPG, JPEG, PNG

### 3. Train Model

#### Option A: Command Line
```bash
cd src
python train_model.py
```

#### Option B: Interactive Notebook
```bash
jupyter notebook notebooks/train_model.ipynb
```

### 4. Make Predictions

#### Single Image
```bash
cd src
python predict.py --model ../models/spoon_fork_detector.h5 --image path/to/image.jpg
```

#### Batch Prediction
```bash
cd src
python predict.py --model ../models/spoon_fork_detector.h5 --folder path/to/images/
```

## Model Architecture

- **Input**: 224x224 RGB images
- **Architecture**: Custom CNN with 4 convolutional blocks
- **Features**: 
  - Batch normalization for stable training
  - Dropout layers to prevent overfitting
  - Data augmentation for better generalization
- **Output**: Binary classification (spoon vs fork)

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32 (adjustable)
- **Image Size**: 224x224 pixels
- **Validation Split**: 20%
- **Early Stopping**: Enabled with patience=10
- **Learning Rate Reduction**: Enabled on plateau

## Expected Results

With sufficient training data (30+ images per class), the model should achieve:
- **Training Accuracy**: >98%
- **Validation Accuracy**: >95%
- **Inference Time**: <100ms per image

## License

This project is open source and available under the MIT License.
