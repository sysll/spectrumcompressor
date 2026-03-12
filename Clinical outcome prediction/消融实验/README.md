# Ablation Experiment Instructions

This folder contains ablation experiment code for our proposed model, used to evaluate the contribution of different components to model performance.

## Experiment Content

We conducted the following ablation experiments:

1. **Base model**: Multi-layer concatenation + FFT compression + gating + weighted Top-K pooling
2. **Last layer only**: Removes multi-layer concatenation, only uses BERT's last layer output
3. **No FFT compression**: Removes FFT compression, uses learnable parameters
4. **No gating mechanism**: Removes spectral gating mechanism
5. **Gating without interpolation**: Removes spectral interpolation, directly takes first S2 frequency components
6. **Low frequency only**: Uses only low frequency components, removes both gating and interpolation
7. **Different S2 value**: Uses S2=10 (base model uses S2=20)
8. **Different S2 value**: Uses S2=30 (base model uses S2=20)
9. **Average pooling**: Uses average pooling instead of weighted Top-K pooling
10. **Max pooling**: Uses max pooling instead of weighted Top-K pooling
11. **CLS pooling**: Uses CLS pooling instead of weighted Top-K pooling
12. **No serum features**: Removes serum features, only uses text and demographic features
13. **No demographic features**: Removes demographic features, only uses text and serum features
14. **Text features only**: Uses only text features
15. **Serum features only**: Uses only serum features
16. **Demographic features only**: Uses only demographic features

## Environment Setup

### Dependencies Installation

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy scikit-learn
pip install matplotlib
```

### Hardware Requirements

- GPU acceleration is recommended for training
- At least 8GB memory

## Data Preparation

1. Ensure `patients_all_data_encoded.xlsx` file is in current directory
2. Data format requirements:
   - Last three columns are text features
   - Other columns are numerical features (serological features and demographic features)
   - Target column is `survival_status` (binary classification)

## Running Experiments

### Running All Ablation Experiments

```bash
python run_ablations.py
```

This command will:
1. Load data and preprocess
2. Train 16 different model variants
3. Select best epoch based on validation set performance
4. Evaluate final performance on test set
5. Save results to `ablation_results.csv` file

### Running Individual Models

If you only need to run a specific model variant, you can modify `run_ablations.py` file and comment out the training code for other models.

## Result Explanation

After running, the following files will be generated:

1. `ablation_results.csv`: Contains performance metrics for all model variants
2. Console output: Detailed training and testing results
3. `ablation_auc_comparison.png`: Bar chart comparing AUC scores
4. `ablation_auprc_comparison.png`: Bar chart comparing AUPRC scores

### Performance Metrics

- **Accuracy**: Overall accuracy
- **AUC**: Area under the ROC curve
- **AUPRC**: Area under the Precision-Recall curve

## Code Structure

- `run_ablations.py`: Main experiment script, contains all model variants and training logic
- `base_model.py`: Copy of base model for reference

## Notes

1. Training time may be long, GPU acceleration is recommended
2. You can adjust the number of training epochs by modifying the `num_epochs` parameter
3. All experiments use the same random seed to ensure reproducibility

## Experimental Settings

- Random seed: 32
- Training epochs: 100
- Batch size: 32
- Learning rate: 1e-3
- Optimizer: Adam
- Loss function: Cross-entropy loss

## Contact

For questions, please contact the project lead.
