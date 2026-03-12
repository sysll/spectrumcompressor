# Clinical Outcome Prediction Model Project

This project contains deep learning models for clinical outcome prediction, along with related comparative experiments and evaluations.

## Project Structure

```
临床结局预测/
├── 传统的Bert的对比模型/       # Traditional BERT model comparative experiments
│   ├── 1.CatBoost.py         # CatBoost classifier
│   ├── 2.XGBoost.py          # XGBoost classifier
│   ├── 3.AdaBoost.py         # AdaBoost classifier
│   ├── 4.神经网络.py         # Neural network classifier
│   ├── 神经网络.py           # Neural network classifier (backup)
│   └── patients_all_data_encoded.xlsx  # Encoded patient data
├── 我们的Bert对比模型/         # Our BERT model comparative experiments
│   ├── 1.Catboost.py         # CatBoost classifier
│   ├── 2.XGBoost.py          # XGBoost classifier
│   ├── 3.AdaBoost.py         # AdaBoost classifier
│   ├── 4.神经网络.py         # Neural network classifier
│   └── patients_all_data_encoded.xlsx  # Encoded patient data
├── 我们的加一般编码器1/         # Main model implementation
│   ├── 传统模型/             # Traditional model results
│   ├── 我们模型的结果/         # Our model results
│   ├── ENCODER+TOPK.py       # Main model implementation
│   ├── 传统的.py            # Traditional model implementation
│   ├── 绘图.py              # Result visualization script
│   └── patients_all_data_encoded.xlsx  # Encoded patient data
├── 消融实验/                # Ablation experiments
│   ├── run_ablations.py      # Main ablation experiment script
│   ├── base_model.py         # Base model copy
│   ├── README.md            # Ablation experiment instructions
│   └── patients_all_data_encoded.xlsx  # Encoded patient data
└── README.md                # Project documentation
```

## Project Overview

This project aims to develop and evaluate multimodal deep learning models for clinical outcome prediction. Key features include:

1. **Multimodal fusion**: Combining text features, serological features, and demographic features
2. **Spectral compression**: Using FFT to compress text features and improve model efficiency
3. **Gating mechanism**: Enhancing important feature representations through learnable gating weights
4. **Weighted Top-K pooling**: Improving feature aggregation through learnable weight-based Top-K pooling
5. **Comparative experiments**: Comparing with traditional machine learning models and other deep learning models

## Environment Setup

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy scikit-learn
pip install matplotlib
pip install catboost xgboost
```

### Hardware Requirements

- GPU acceleration is recommended for training
- At least 8GB of memory

## Data Preparation

All models use the same dataset `patients_all_data_encoded.xlsx` with the following format requirements:

- Last three columns are text features
- Other columns are numerical features (serological and demographic features)
- Target column is `survival_status` (binary classification)

## Running Instructions

### 1. Running the Main Models

```bash
# Navigate to the main model directory
cd 我们的加一般编码器1

# Run our model
python ENCODER+TOPK.py

# Run traditional model
python 传统的.py
```

### 2. Running Comparative Experiments

```bash
# Navigate to traditional BERT comparison model directory
cd 传统的Bert的对比模型

# Run each comparison model
python 1.CatBoost.py
python 2.XGBoost.py
python 3.AdaBoost.py
python 4.神经网络.py

# Navigate to our BERT comparison model directory
cd ../我们的Bert对比模型

# Run each comparison model
python 1.Catboost.py
python 2.XGBoost.py
python 3.AdaBoost.py
python 4.神经网络.py
```

## Model Description

### 1. Main Model (ENCODER+TOPK.py)

- **Text encoder**: Uses BERT to extract multi-layer features, enhanced through FFT compression and gating mechanism
- **Serum encoder**: Uses MLP to map serological features to high-dimensional space
- **Demographic encoder**: Uses MLP to map demographic features to high-dimensional space
- **Fusion model**: Uses Transformer to fuse multimodal features, aggregates features through weighted Top-K pooling

### 2. Traditional Model (传统的.py)

- **Text encoder**: Uses BERT to extract features, aggregated through average pooling
- **Numerical encoder**: Uses MLP to map numerical features to high-dimensional space
- **Fusion model**: Uses Transformer to fuse multimodal features, aggregates features through average pooling

### 3. Comparison Models

- **CatBoost**: Gradient boosting decision tree model
- **XGBoost**: Extreme gradient boosting model
- **AdaBoost**: Adaptive boosting model
- **Neural network**: Simple feedforward neural network

### 4. Ablation Experiment Models

Contains 14 model variants to evaluate the contribution of different components:
- Base model: Multi-layer concatenation + FFT compression + gating + weighted Top-K pooling
- Last layer only: Removes multi-layer concatenation, only uses BERT's last layer output
- No FFT compression: Removes FFT compression, uses learnable parameters
- No gating mechanism: Removes spectral gating mechanism
- Gating without interpolation: Removes spectral interpolation, directly takes first S2 frequency components
- Low frequency only: Uses only low frequency components, removes both gating and interpolation
- Different S2 value: Uses S2=10 (base model uses S2=20)
- Different S2 value: Uses S2=30 (base model uses S2=20)
- Average pooling: Uses average pooling instead of weighted Top-K pooling
- Max pooling: Uses max pooling instead of weighted Top-K pooling
- CLS pooling: Uses CLS pooling instead of weighted Top-K pooling
- No serum features: Removes serum features, only uses text and demographic features
- No demographic features: Removes demographic features, only uses text and serum features
- Only text features: Uses only text features
- Only serum features: Uses only serum features
- Only demographic features: Uses only demographic features

## Result Evaluation

All models are evaluated using the following metrics:

- **Accuracy**: Overall accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-score**: F1 score
- **AUC**: Area under the ROC curve
- **AUPRC**: Area under the Precision-Recall curve

## Visualization

Use the `绘图.py` script to visualize model prediction results and feature distributions.

## Notes

1. Training time may be long, GPU acceleration is recommended
2. All models use 6:2:2 train/validation/test split
3. Best model is selected based on validation set performance, evaluated only once on test set
4. All experiments use fixed random seeds to ensure reproducibility

## Contact

For questions, please contact the project负责人 (project lead).
