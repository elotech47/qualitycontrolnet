#!/usr/bin/env python3
"""
Model Configurations for Comparison Experiments

This script provides predefined configurations for different model architectures
to ensure fair comparison in the research paper.
"""

import json
from pathlib import Path

# Base configuration that will be shared across all models
BASE_CONFIG = {
    "# General Settings": "Settings that control the overall experiment",
    "experiment_name": "model_comparison",
    "random_seed": 42,
    "device": "cuda",
    
    "# Data Settings": "Settings for data loading and preprocessing",
    "data_root": "../Mohan",
    "img_size": 224,
    "batch_size": 32,
    "test_size": 0.2,
    "stratify_split": True,
    "num_workers": 4,
    "pin_memory": True,
    
    "# Training Settings": "Settings for the training process",
    "num_epochs": 50,
    "classification_weight": 1.0,
    "similarity_weight": 1.0,
    
    "# Optimizer Settings": "Settings for the optimizer",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "beta1": 0.9,
    "beta2": 0.999,
    
    "# Scheduler Settings": "Settings for the learning rate scheduler",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_mode": "min",
    "scheduler_factor": 0.1,
    "scheduler_patience": 10,
    "scheduler_min_lr": 1e-6,
    
    "# Early Stopping": "Settings for early stopping",
    "early_stopping": True,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    
    "# Augmentation Settings": "Settings for data augmentation",
    "use_augmentation": True,
    "aug_rotate90_prob": 0.5,
    "aug_flip_prob": 0.5,
    "aug_shift_scale_rotate_prob": 0.5,
    "aug_noise_prob": 0.2,
    "aug_blur_prob": 0.2,
    "aug_color_prob": 0.3,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    
    "# Logging Settings": "Settings for logging and checkpoints",
    "checkpoint_interval": 5,
    "log_batch_interval": 10,
    "save_best_only": True
}

# Model-specific configurations
MODEL_CONFIGS = {
    "QualityControlNet": {
        "model_architecture": "QualityControlNet",
        "backbone": "mobilenet_v2",
        "pretrained": True,
        "dropout_rate": 0.5,
        "similarity_head_layers": [512, 256, 1],
        "classification_head_layers": [512, 256, 2],
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    },
    
    "VanillaCNN": {
        "model_architecture": "VanillaCNN",
        "dropout_rate": 0.5,
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    },
    
    "SiameseNetwork": {
        "model_architecture": "SiameseNetwork",
        "pretrained": True,
        "dropout_rate": 0.5,
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    },
    
    "DifferenceNetwork": {
        "model_architecture": "DifferenceNetwork",
        "backbone": "resnet18",
        "pretrained": True,
        "dropout_rate": 0.5,
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    },
    
    "EarlyFusionCNN": {
        "model_architecture": "EarlyFusionCNN",
        "dropout_rate": 0.5,
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    },
    
    "MobileNetModel": {
        "model_architecture": "MobileNetModel",
        "pretrained": True,
        "dropout_rate": 0.5,
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    },
    
    "AttentionModel": {
        "model_architecture": "AttentionModel",
        "backbone": "resnet18",
        "pretrained": True,
        "dropout_rate": 0.5,
        "classification_loss": "CrossEntropyLoss",
        "similarity_loss": "MSELoss"
    }
}


def generate_model_configs(output_dir="model_configs"):
    """
    Generate configuration files for each model architecture
    
    Args:
        output_dir: Directory to save configuration files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate config for each model
    for model_name, model_config in MODEL_CONFIGS.items():
        # Combine base config with model-specific config
        config = {**BASE_CONFIG}
        config.update(model_config)
        config["experiment_name"] = f"{model_name}_experiment"
        
        # Save to file
        config_path = output_dir / f"{model_name.lower()}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Created configuration for {model_name}: {config_path}")
    
    # Also save the common baseline configuration
    with open(output_dir / "base_config.json", 'w') as f:
        json.dump(BASE_CONFIG, f, indent=4)
    
    print(f"Created base configuration: {output_dir / 'base_config.json'}")


def get_model_config(model_name):
    """
    Get configuration for a specific model
    
    Args:
        model_name: Name of the model architecture
        
    Returns:
        dict: Configuration dictionary for the specified model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_CONFIGS.keys())}")
    
    # Combine base config with model-specific config
    config = {**BASE_CONFIG}
    config.update(MODEL_CONFIGS[model_name])
    config["experiment_name"] = f"{model_name}_experiment"
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate configurations for model comparison')
    parser.add_argument('--output_dir', type=str, default='model_configs',
                        help='Directory to save configuration files')
    
    args = parser.parse_args()
    generate_model_configs(args.output_dir)