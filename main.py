import argparse
import os
import torch
import random
import numpy as np
from pathlib import Path

# Local imports
from config_loader import load_config, save_config
from data_utils import prepare_multi_shape_data
from model import get_model
from training import train_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D Print Quality Control Training')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for data (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (overrides config)')

        


    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'config':
            config[arg_name] = arg_value
    
    # Set random seed for reproducibility
    set_seed(config.get('random_seed', 42))
    
    # Prepare data
    train_loader, val_loader = prepare_multi_shape_data(config)
    
    if train_loader is None or val_loader is None:
        print("Error: Failed to prepare data. Exiting.")
        return
    
    # Create model
    model = get_model(config)
    
    # Train model
    trained_model, logger = train_model(model, train_loader, val_loader, config)
    
    print(f"Training completed. Results saved to {logger.exp_dir}")
    
    # Save the final model
    final_model_path = Path(logger.exp_dir) / 'final_model.pt'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()