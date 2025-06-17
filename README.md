# Deeply Visual Microphone: Sound Recovery from Video using Deep Learning

## Abstract

This repository presents a novel approach to visual microphone technology that combines traditional steerable pyramid-based motion detection with deep learning architectures for enhanced sound recovery from video. The project implements a comprehensive pipeline for extracting audio signals from subtle visual motion patterns captured in video recordings, utilizing advanced neural network architectures including Siamese networks, ResNet-based feature extractors, and Feature Pyramid Networks (FPN).

**Keywords:** Visual Microphone, Deep Learning, Computer Vision, Signal Processing, Audio Recovery, Steerable Pyramids

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [References](#references)

## Introduction

Visual microphones represent a revolutionary approach to sound capture, enabling audio recovery from video recordings by analyzing minute visual vibrations caused by sound waves. This research extends traditional visual microphone techniques by incorporating deep learning methodologies to improve signal quality and robustness.

### Research Objectives

1. **Primary Objective**: Develop a hybrid approach combining steerable pyramid motion detection with deep learning for enhanced sound recovery
2. **Secondary Objectives**: 
   - Implement multiple neural network architectures for comparative analysis
   - Develop robust noise reduction techniques using spectral subtraction
   - Create a comprehensive evaluation framework for visual microphone performance

## Methodology

### Core Algorithm

The visual microphone implementation is based on a multi-stage pipeline:

1. **Motion Detection**: Utilizes steerable pyramid decomposition to detect subtle motion across multiple scales and orientations
2. **Phase Analysis**: Computes phase differences between reference and current frames to extract motion signals
3. **Signal Alignment**: Aligns motion signals from different pyramid bands using cross-correlation
4. **Noise Reduction**: Applies spectral subtraction for signal enhancement
5. **Deep Learning Enhancement**: Employs neural networks for further signal refinement

### Steerable Pyramid Implementation

The core visual microphone algorithm (`thonVM/sound_from_video.py`) implements:

- **Multi-scale Analysis**: Decomposes video frames into multiple scales for comprehensive motion detection
- **Orientation Selectivity**: Analyzes motion across different orientations to capture diverse vibration patterns
- **Phase-based Motion Extraction**: Utilizes complex steerable pyramid coefficients for precise motion quantification
- **Temporal Alignment**: Synchronizes signals from different frequency bands using FFT-based cross-correlation

### Signal Processing Pipeline

```
Video Input → Downsampling → Grayscale Conversion → Steerable Pyramid → 
Phase Analysis → Motion Signal Extraction → Band Alignment → 
High-pass Filtering → Spectral Subtraction → Audio Output
```

## Architecture

### Neural Network Models

The project implements three distinct deep learning architectures:

#### 1. Siamese Convolutional Network
- **Purpose**: Learn similarity metrics between adjacent video frames
- **Architecture**: Dual-branch CNN with shared weights
- **Input**: Concatenated frame pairs (64×64 pixels)
- **Output**: Continuous audio amplitude values

#### 2. ResNet-50 Based Feature Extractor
- **Purpose**: Leverage pre-trained ImageNet features for visual-audio mapping
- **Architecture**: Modified ResNet-50 with custom fully connected layers
- **Input**: Frame pairs (224×224 pixels)
- **Features**: 4096-dimensional concatenated feature vectors

#### 3. Feature Pyramid Network (FPN)
- **Purpose**: Multi-scale feature extraction with top-down pathway
- **Architecture**: Bottom-up and top-down processing with lateral connections
- **Advantages**: Captures both fine-grained and semantic features

### Loss Function and Optimization

- **Loss Function**: Mean Squared Error (MSE) for regression tasks
- **Optimizer**: Adam optimizer with learning rate scheduling
- **Evaluation Metrics**: Root Mean Square Error (RMSE), Pearson correlation

## Dataset

### Data Structure

The project utilizes a custom dataset structure:

```
Audio/
├── SampledData.csv          # Frame-audio correspondence data
├── metadata.csv             # Video metadata (frame rates, durations)
├── Video1.wav              # Original audio tracks
├── Video1_subsampled_-1To1.wav  # Processed audio samples
└── ...
```

### Dataset Statistics

- **Videos**: 4 experimental videos
- **Frame Rate**: 2200 Hz sampling rate
- **Total Frames**: ~140,000 frames across all videos
- **Audio Format**: WAV files with spectral subtraction variants

### Data Preprocessing

The `VMDataset.py` implements:
- **Dynamic range organization**: Frames organized in numbered ranges
- **Label extraction**: Frame-to-audio amplitude mapping
- **Transform pipeline**: PyTorch-compatible data loading
- **Batch processing**: Efficient data loading for training

## Experimental Setup

### Training Configuration

```python
# Model Parameters
batch_size = 32
learning_rate = 0.001
epochs = 100
input_size = (3, 32, 64)  # Siamese Network
input_size = (3, 224, 448)  # ResNet/FPN

# Optimization
optimizer = Adam
loss_function = MSELoss
scheduler = StepLR(step_size=30, gamma=0.1)
```

### Evaluation Protocol

1. **Training/Validation Split**: 80/20 split with temporal consistency
2. **Cross-Validation**: K-fold validation across different video segments
3. **Metrics**: RMSE, Mean Absolute Error, Pearson correlation coefficient
4. **Visualization**: Spectrogram comparison between recovered and ground truth audio

## Results

### Performance Metrics

| Model | RMSE | MAE | Correlation | Training Time |
|-------|------|-----|-------------|---------------|
| Siamese CNN | - | - | - | ~2 hours |
| ResNet-50 | - | - | - | ~4 hours |
| FPN | - | - | - | ~3 hours |

*Note: Detailed experimental results to be updated upon completion of full evaluation*

### Qualitative Analysis

- **Spectrogram Visualization**: Generated spectrograms demonstrate successful audio recovery
- **Noise Reduction**: Spectral subtraction effectively reduces background noise
- **Signal Quality**: Recovered audio maintains fundamental frequency characteristics

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for large dataset processing

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Deeply_Visual_Mic
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Basic Sound Recovery

```bash
# Extract sound from video using traditional visual microphone
python videoToSound.py path/to/video.mp4 -o recovered_audio.wav -s 30

# Arguments:
# - video_path: Input video file
# - -o, --output: Output audio file (default: recoveredsound.wav)
# - -s, --sampling-rate: Video frame rate for audio sampling
```

### Deep Learning Model Training

```bash
# Train neural network models
python DeepVM/eval.py --model siamese --epochs 100 --batch-size 32

# Available models: siamese, resnet, fpn
# Additional parameters: --learning-rate, --data-path, --save-model
```

### Advanced Processing

```bash
# Apply spectral subtraction for noise reduction
python thonVM/spec_sub.py input_audio.wav output_clean.wav

# Generate dataset from video
python DeepVM/preprocessdata.py --video-dir /path/to/videos --output-dir /path/to/dataset
```

## Project Structure

```
Deeply_Visual_Mic/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── videoToSound.py             # Main visual microphone script
├── test.wav                    # Sample audio output
├── test_specsub.wav           # Spectral subtraction result
│
├── Audio/                      # Dataset and audio files
│   ├── SampledData.csv        # Frame-audio correspondence
│   ├── metadata.csv           # Video metadata
│   └── *.wav                  # Audio samples
│
├── DeepVM/                    # Deep learning components
│   ├── VMDataset.py          # PyTorch dataset implementation
│   ├── eval.py               # Model training and evaluation
│   ├── generate_audio.py     # Audio generation utilities
│   ├── preprocessdata.py     # Data preprocessing
│   ├── readWav.py           # Audio file utilities
│   └── spectrogram.png      # Visualization output
│
└── thonVM/                   # Core visual microphone algorithms
    ├── sound_from_video.py  # Main VM implementation
    └── spec_sub.py          # Spectral subtraction
```

## Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision and video processing
- **NumPy/SciPy**: Numerical computing and signal processing
- **pyrtools**: Steerable pyramid implementation
- **scikit-image**: Image processing utilities
- **matplotlib**: Visualization and plotting
- **pandas**: Data manipulation and analysis

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (GTX 1060+ recommended)
- **Memory**: 16GB+ system RAM
- **Storage**: 50GB+ free space for datasets and models

## Future Work

### Planned Enhancements

1. **Real-time Processing**: Develop streaming visual microphone capability
2. **Attention Mechanisms**: Incorporate transformer architectures
3. **Synthetic Data Generation**: Create larger training datasets using simulation

### Research Directions

- **Cross-domain Adaptation**: Generalization across different recording conditions
- **Adversarial Training**: Robust performance against noise and artifacts
- **Multimodal Fusion**: Combining visual and traditional audio cues
- **Edge Computing**: Efficient implementation for IoT devices

## References

1. Davis, A., Rubinstein, M., Wadhwa, N., Mysore, G. J., Durand, F., & Freeman, W. T. (2014). The visual microphone: Recovering sound from video. *ACM Transactions on Graphics*, 33(4), 1-10.

2. Simoncelli, E. P., & Freeman, W. T. (1995). The steerable pyramid: A flexible architecture for multi-scale derivative computation. *Proceedings of IEEE International Conference on Image Processing*.

3. Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

---

## Contributors

- Primary Researcher: Nischal B K
- Institution: Indiana University Bloomington