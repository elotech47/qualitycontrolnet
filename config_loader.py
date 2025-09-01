import json
import os
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from a JSON file
    
    Args:
        config_path: Path to the config JSON file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set defaults for missing values
    defaults = {
        # Model settings
        "model_architecture": "QualityControlNet",
        "pretrained": True,
        "backbone": "resnet18",
        "feature_dim": 512,
        "dropout_rate": 0.5,
        "similarity_head_layers": [512, 256, 1],
        "classification_head_layers": [512, 256, 2],
        "use_bilinear": True,
        "bilinear_out_features": 128,
        # Training settings
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        "num_epochs": 50,
        "batch_size": 32,
        "random_seed": 42,
        
        # Optimizer settings
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "beta1": 0.9,
        "beta2": 0.999,
        
        # Scheduler settings
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "min",
        "scheduler_factor": 0.1,
        "scheduler_patience": 5,
        "scheduler_min_lr": 1e-6,
        
        # Loss settings
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss",
        "classification_weight": 1.0,
        "similarity_weight": 1.0,
        
        # Data settings
        "data_root": "../data",
        "test_size": 0.2,
        "stratify_split": True,
        "img_size": 224,
        
        # Augmentation settings
        "use_augmentation": True,
        "aug_rotate90_prob": 0.5,
        "aug_flip_prob": 0.5,
        "aug_shift_scale_rotate_prob": 0.5,
        "aug_noise_prob": 0.2,
        "aug_blur_prob": 0.2,
        "aug_color_prob": 0.3,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        
        # Logging settings
        "experiment_name": "quality_control",
        "checkpoint_interval": 5,
        "log_batch_interval": 10,
        "save_best_only": False,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
    }
    
    # Fill in missing values with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    return config


def save_config(config, save_path):
    """
    Save configuration to a JSON file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)