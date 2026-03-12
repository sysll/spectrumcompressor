# Ablation Experiment Instructions

This folder contains ablation experiment code for our proposed model, used to evaluate the contribution of different components to model performance.

## Experiment Content

We conducted the following ablation experiments:

1. **Base model**: Multi-layer concatenation + FFT compression + gating + weighted Top-K pooling
2. **Last layer only**: Removes multi-layer concatenation, only uses BERT's last layer output
3. **FFT compression without gating**: Removes spectral gating mechanism
4. **Gating without interpolation**: Removes spectral interpolation, directly takes first S2 frequency components
5. **Low frequency only**: Uses only low frequency components, removes both gating and interpolation
6. **Different S2 value**: Uses S2=10 (base model uses S2=20)
7. **Average pooling**: Uses average pooling instead of weighted Top-K pooling
8. **Max pooling**: Uses max pooling instead of weighted Top-K pooling
9. **CLS pooling**: Uses CLS pooling instead of weighted Top-K pooling
10. **Top-K pooling without learnable weights**: Uses regular Top-K pooling, removes learnable weights

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

1. Ensure `patients_all_data_encoded.xlsx` file is in the current directory
2. Data format requirements:
   - Last three columns are text features
   - Other columns are numerical features (serological and demographic features)
   - Target column is `recurrence` (binary classification)

## Running Experiments

### Running All Ablation Experiments

```bash
python run_ablations.py
```

This command will:
1. Load data and preprocess
2. Train 9 different model variants
3. Select the best epoch based on validation set performance
4. Evaluate final performance on test set
5. Save results to `ablation_results.json` file

### Running Individual Models

If you only need to run a specific model variant, you can modify the `run_ablations.py` file and comment out the training code for other models.

## Result Explanation

After running, the following files will be generated:

1. `ablation_results.json`: Contains performance metrics for all model variants
2. Console output: Detailed training and testing results

### Performance Metrics

- **Accuracy**: Overall accuracy
- **AUPRC**: Area under the Precision-Recall curve

## Code Structure

- `run_ablations.py`: Main experiment script, contains all model variants and training logic
- `base_model.py`: Copy of the base model for reference

## Notes

1. Training time may be long, GPU acceleration is recommended
2. You can adjust the number of training epochs by modifying the `num_epochs` parameter
3. All experiments use the same random seed to ensure reproducibility

## Contact

For questions, please contact the project负责人 (project lead).
