# Wavelet Model Experiments

This directory contains all experiments with the Wavelet Scattering Transform library, organized to enable comprehensive comparison between different models and datasets.

## Directory Structure

### Dataset-based Experiments

- `dataset0/`: Experiments on the original 4-class vegetation dataset
- `dataset1/`: Reserved for experiments on future datasets
- `dataset2/`: Reserved for experiments on future datasets
- `dataset3/`: Reserved for experiments on future datasets

Each dataset directory contains multiple experiments with different models or configurations, organized by experiment name.

### Cross-experiment Resources

- `comparisons/`: Cross-dataset and cross-model comparisons
- `models/`: Reference models and shared model architectures
- `results/`: Consolidated results across multiple experiments

## Running Experiments

To run a new experiment, use the format:

```bash
python script/core/train.py --dataset /path/to/dataset \
  --output-base experiments/dataset_name --experiment-name model_name_run
```

## Comparing Experiments

After running multiple experiments, use the visualization tools to generate comparisons:

```bash
python script/core/visualize.py metrics \
  --model-dir experiments/dataset0/model_output/exp1 \
  --compare-with experiments/dataset1/model_output/exp1 \
  --output-base experiments/comparisons
```

## Experimental Matrix

| Dataset | WST+CNN | WST+MLP | Linear | Notes |
|---------|---------|---------|--------|-------|
| dataset0 |  | - | - | Baseline |
| dataset1 | - | - | - | Future |
| dataset2 | - | - | - | Future |

 = Completed, Ë = In Progress, - = Not Started