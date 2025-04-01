# Dataset0 Experiment - WST Baseline Model

This is a baseline experiment using the Wavelet Scattering Transform model on the original 4-class dataset from `/home/brus/Projects/wavelet/datasets/HPL_images/4_classes`.

## Experiment Details

- **Date**: April 1, 2025
- **Dataset**: HPL_images/4_classes
- **Classes**: vegetation 1, vegetation 2, vegetation 3, water
- **Model**: Wavelet Scattering Transform + CNN
- **Run**: 0_run (baseline)

## Directory Structure

- `classification_result/`: Contains classification results for sample images
- `dataset_info/`: Contains dataset statistics and information
- `evaluation/`: Contains evaluation metrics and confusion matrices
- `models/`: Contains the trained model checkpoints (best, final, and intermediates)
- `model_output/`: Contains training metrics and progress data
- `visualization/`: Contains generated visualizations of samples and metrics

## Model Performance

The model achieved excellent performance on the 4-class vegetation dataset, with high accuracy across all classes. See the evaluation directory for detailed metrics and confusion matrices.

## Comparing Different Models/Datasets

### Different Models on Same Dataset

To test a different model on the same dataset:

```bash
# Run training with a different model architecture or parameters
python script/core/train.py --dataset /home/brus/Projects/wavelet/datasets/HPL_images/4_classes \
  --output-base experiments/dataset0 --experiment-name model2_run \
  --model-type mlp  # Different model type

# Evaluate the new model
python script/core/evaluate.py --model-path experiments/dataset0/models/model2_run/best_model.pth \
  --dataset /home/brus/Projects/wavelet/datasets/HPL_images/4_classes \
  --output-base experiments/dataset0 --experiment-name model2_run
```

### Same Model on Different Dataset

To test the same model on a different dataset:

```bash
# Run training on a new dataset
python script/core/train.py --dataset /path/to/new_dataset \
  --output-base experiments/dataset1 --experiment-name wst_run

# Evaluate the model
python script/core/evaluate.py --model-path experiments/dataset1/models/wst_run/best_model.pth \
  --dataset /path/to/new_dataset \
  --output-base experiments/dataset1 --experiment-name wst_run
```

### Comparing Results

Generate comparison visualizations between experiments:

```bash
# Compare model outputs across datasets
python script/core/visualize.py metrics \
  --model-dir experiments/dataset0/model_output/0_run \
  --compare-with experiments/dataset1/model_output/wst_run \
  --output-base experiments/comparisons --experiment-name dataset0_vs_dataset1
```