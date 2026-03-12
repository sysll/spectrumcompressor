# SpectrumCompress - Multimodal Deep Learning Model for Medical Prediction

A multimodal deep learning framework based on spectral compression and weighted Top-K pooling for clinical outcome prediction and recurrence prediction.

## Project Overview

This project proposes a novel multimodal deep learning framework that efficiently fuses text, serum, and demographic features through spectral compression techniques for medical prediction tasks.

### Core Innovations

1. **Spectral Compression**: Uses FFT to perform frequency domain compression on text features extracted by BERT, significantly reducing computational complexity
2. **Learnable Gating**: Enhances important feature representations through learnable spectral gating weights
3. **Weighted Top-K Pooling**: Improves Top-K pooling effectiveness through learnable weights
4. **Multimodal Fusion**: Uses Transformer to fuse features from three modalities

## Project Structure

```
Spectrumcompress/
├── Clinical outcome prediction/    # Clinical outcome prediction task
│   ├── 传统的Bert的对比模型/        # Traditional machine learning model comparison
│   ├── 我们的Bert对比模型/          # BERT-based model comparison
│   ├── 我们的加一般编码器1/          # Main model implementation
│   ├── 消融实验/                    # Ablation experiments
│   └── README.md
└── recurrence prediction/            # Recurrence prediction task
    ├── 传统的Bert的对比模型/        # Traditional machine learning model comparison
    ├── 我们的Bert对比模型/          # BERT-based model comparison
    ├── 我们的加一般编码器1/          # Main model implementation
    ├── 消融实验/                    # Ablation experiments
    └── README.md
```

## Model Architecture

### Text Encoder (SpectralCompressor)

- Uses BERT (sentence-transformers/all-MiniLM-L6-v2) to extract multi-layer features
- Extracts features from first layer, middle layer, and last layer and concatenates them
- Uses FFT for spectral compression, compressing sequence length to specified size (default 20)
- Uses learnable gating mechanism to weight the spectrum
- Uses linear interpolation for spectral downsampling

### Serum Feature Encoder (SerumMLPEncoder)

- Uses MLP to map serum features (33 dimensions) to high-dimensional space (384 dimensions)
- Uses learnable sequence parameters to expand to specified length (37)
- Hidden layer dimension: 128

### Demographic Feature Encoder (DemographicEncoder)

- Uses MLP to map demographic features (10 dimensions) to high-dimensional space (384 dimensions)
- Uses learnable sequence parameters to expand to specified length (10)
- Hidden layer dimension: 64

### Fusion Model (FusionTransformer3Modal)

- Uses Transformer encoder to fuse three-modality features
- Uses weighted Top-K pooling for feature aggregation (Top-K=3)
- Uses learnable weights to perform weighted average on Top-K values
- Final prediction through linear classifier

## Environment Setup

### Dependencies Installation

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy scikit-learn
pip install matplotlib
pip install catboost xgboost
```

### Hardware Requirements

- GPU acceleration is recommended for training
- At least 8GB memory

## Quick Start

### Clinical Outcome Prediction

```bash
cd "Clinical outcome prediction/我们的加一般编码器1"
python ENCODER+TOPK.py
```

### Recurrence Prediction

```bash
cd "recurrence prediction/我们的加一般编码器1"
python ENCODER+TOPK.py
```

## Data Format

All models use the same dataset `patients_all_data_encoded.xlsx`:

- Last three columns: Text features (clinical records)
- Other columns: Numerical features (serum features + demographic features)
- Target column: `survival_status` (clinical outcome) or `recurrence` (recurrence)
- Data split: 6:2:2 (training set:validation set:test set)

## Ablation Experiments

Ablation experiments are used to evaluate the contribution of each component to model performance:

1. Base model: Multi-layer concatenation + FFT compression + gating + weighted Top-K pooling
2. Last layer only: Removes multi-layer concatenation, uses only BERT's last layer
3. No FFT compression: Removes FFT compression, uses learnable parameters
4. No gating mechanism: Removes spectral gating mechanism
5. Gating without interpolation: Removes spectral interpolation, directly takes first S2 frequency components
6. Low frequency only: Uses only low frequency components, removes both gating and interpolation
7. Different S2 value: Uses S2=10 or S2=30 (base model uses S2=20)
8. Average pooling: Uses average pooling instead of weighted Top-K pooling
9. Max pooling: Uses max pooling instead of weighted Top-K pooling
10. CLS pooling: Uses CLS pooling instead of weighted Top-K pooling
11. No serum features: Removes serum features, uses only text and demographic features
12. No demographic features: Removes demographic features, uses only text and serum features
13. Text features only: Uses only text features
14. Serum features only: Uses only serum features
15. Demographic features only: Uses only demographic features

## Evaluation Metrics

All models are evaluated using the following metrics:

- **Accuracy**: Overall accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-score**: F1 score
- **AUC**: Area under the ROC curve
- **AUPRC**: Area under the Precision-Recall curve

## Comparison Models

### Traditional Machine Learning Models

- **CatBoost**: Gradient boosting decision tree model
- **XGBoost**: Extreme gradient boosting model
- **AdaBoost**: Adaptive boosting model
- **Neural Network**: Simple feedforward neural network

### BERT-based Comparison Models

- Uses BERT to extract text features
- Uses the same numerical feature encoders
- Uses the same fusion model
- Difference lies in text feature processing approach

## Experimental Settings

- Random seed: 32 (clinical outcome prediction) / 22 (recurrence prediction)
- Training epochs: 100
- Batch size: 32
- Learning rate: 1e-3
- Optimizer: Adam
- Loss function: Cross-entropy loss
- Early stopping: Stops training when validation set performance does not improve for 10 consecutive epochs

## Result Visualization

Use the `绘图.py` script to visualize model prediction results and feature distributions, including:

- ROC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance analysis

## Notes

1. Training time is long, GPU acceleration is recommended
2. All experiments use fixed random seeds to ensure reproducibility
3. Best model is selected based on validation set performance and evaluated only once on test set
4. Dataset needs to be prepared in advance and placed in the corresponding directory

## Citation

If you use the code or methods from this project, please cite the relevant paper.

## Contact

For questions, please contact the project lead.

## License

This project code is for academic research use only.
