"""
Utility generali per la manipolazione di dati e modelli.
Questo package fornisce funzioni di utilità per l'analisi di dataset,
la gestione di modelli, e altre operazioni comuni.
"""

from .data_utils import (
    analyze_dataset,
    visualize_dataset_samples,
    plot_class_distribution,
    check_dataset_balance,
    find_corrupted_images,
    compute_dataset_stats
)

from .model_utils import (
    list_model_checkpoints,
    load_latest_checkpoint,
    visualize_model_performance,
    plot_confusion_matrix,
    plot_training_history,
    analyze_model_predictions,
    export_model_to_onnx
)

__all__ = [
    # Data utils
    'analyze_dataset',
    'visualize_dataset_samples',
    'plot_class_distribution',
    'check_dataset_balance',
    'find_corrupted_images',
    'compute_dataset_stats',
    
    # Model utils
    'list_model_checkpoints',
    'load_latest_checkpoint',
    'visualize_model_performance',
    'plot_confusion_matrix',
    'plot_training_history',
    'analyze_model_predictions',
    'export_model_to_onnx'
]
