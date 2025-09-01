import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


import matplotlib 
matplotlib.use('Agg')

class ExperimentLogger:
    def __init__(self, config):
        """
        Initialize the experiment logger
        
        Args:
            config: Configuration dictionary
        """
        # Extract config parameters
        exp_name = config.get('experiment_name', 'experiment')
        
        # Create experiment directory
        self.exp_dir = Path(f'experiments/{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.exp_dir / 'tensorboard'))
        
        # Setup logging
        self.setup_logging()
        
        # Store config
        self.config = config
        
        # Initialize metrics storage
        self.metrics = {
            'train_losses': [],
            'train_class_losses': [],
            'train_sim_losses': [],
            'val_losses': [],
            'val_class_losses': [],
            'val_sim_losses': [],
            'accuracies': [],
            'learning_rates': [],
            'batch_losses': [],
            'batch_class_losses': [],
            'batch_sim_losses': [],
            'best_epoch': -1,
            'best_val_loss': float('inf'),
            'best_val_accuracy': 0
        }
    
    def setup_logging(self):
        """Setup file and console logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.exp_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to a file and TensorBoard"""
        # Save config to a file
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        # Log to console
        self.logger.info(f"Hyperparameters: {config}")
        
        # Log to TensorBoard
        # Convert config to string-only dict
        string_config = {k: str(v) for k, v in config.items()}
        self.writer.add_text('hyperparameters', str(string_config))
    
    def log_metrics(self, metrics, step, phase):
        """
        Log metrics to tensorboard and store them if they are validation metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Training step or epoch
            phase: 'train', 'validation', or 'test'
        """
        for name, value in metrics.items():
            metric_key = f'{phase}/{name}'
            self.writer.add_scalar(metric_key, value, step)
            
            # Store the metrics for plotting
            if phase == 'validation':
                if name == 'loss':
                    self.metrics['val_losses'].append(value)
                elif name == 'classification_loss':
                    self.metrics['val_class_losses'].append(value)
                elif name == 'similarity_loss':
                    self.metrics['val_sim_losses'].append(value)
                elif name == 'accuracy':
                    self.metrics['accuracies'].append(value)
                
            elif phase == 'train':
                if name == 'loss':
                    self.metrics['train_losses'].append(value)
                elif name == 'classification_loss':
                    self.metrics['train_class_losses'].append(value)
                elif name == 'similarity_loss':
                    self.metrics['train_sim_losses'].append(value)
                elif name == 'lr':
                    self.metrics['learning_rates'].append(value)
    
    # def plot_metrics(self):
    #     """Plot training curves if we have collected metrics"""
    #     if not self.metrics['train_losses'] or not self.metrics['val_losses']:
    #         self.logger.warning("No metrics to plot yet")
    #         return
            
    #     plt.figure(figsize=(15, 10))
        
    #     # Plot training and validation loss
    #     plt.subplot(2, 2, 1)
    #     epochs = range(1, len(self.metrics['train_losses']) + 1)
    #     plt.plot(epochs, self.metrics['train_losses'], 'b-', label='Train Loss')
    #     plt.plot(epochs, self.metrics['val_losses'], 'r-', label='Val Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.title('Training and Validation Loss')
        
    #     # Plot classification and similarity losses
    #     plt.subplot(2, 2, 2)
    #     plt.plot(epochs, self.metrics['train_class_losses'], 'b--', label='Train Classification Loss')
    #     plt.plot(epochs, self.metrics['val_class_losses'], 'r--', label='Val Classification Loss')
    #     plt.plot(epochs, self.metrics['train_sim_losses'], 'g--', label='Train Similarity Loss')
    #     plt.plot(epochs, self.metrics['val_sim_losses'], 'y--', label='Val Similarity Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.title('Component Losses')
        
    #     # Plot accuracy
    #     plt.subplot(2, 2, 3)
    #     plt.plot(epochs, self.metrics['accuracies'], 'g-', label='Validation Accuracy')
    #     plt.axhline(y=max(self.metrics['accuracies']), color='r', linestyle='--', 
    #                 label=f'Best: {max(self.metrics["accuracies"]):.2f}%')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy (%)')
    #     plt.legend()
    #     plt.title('Validation Accuracy')
        
    #     # Plot learning rate
    #     if self.metrics['learning_rates']:
    #         plt.subplot(2, 2, 4)
    #         plt.plot(epochs, self.metrics['learning_rates'], 'b-', label='Learning Rate')
    #         plt.xlabel('Epoch')
    #         plt.ylabel('Learning Rate')
    #         plt.yscale('log')
    #         plt.title('Learning Rate Schedule')
        
    #     plt.tight_layout()
    #     plt.savefig(self.exp_dir / 'training_curves.png')
    #     plt.close()
    
    def log_confusion_matrix(self, y_true, y_pred, epoch):
        """Log confusion matrix as image and to TensorBoard"""
        # Convert to numpy if tensors
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
            
        cm = confusion_matrix(y_true, y_pred)
        
        # Create and save figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.savefig(self.exp_dir / f'confusion_matrix_epoch_{epoch}.png')
        plt.close()
        
        # Log to tensorboard
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        self.writer.add_figure('Confusion Matrix', fig, epoch)
        plt.close()

    def plot_metrics(self):
        import matplotlib.ticker as mtick
        """Plot training curves with publication-quality settings for academic papers"""
        if not self.metrics['train_losses'] or not self.metrics['val_losses']:
            self.logger.warning("No metrics to plot yet")
            return
        
        # Set publication-quality plot parameters
        plt.rcParams.update({
            # General parameters
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
            'font.size': 14,
            'font.weight': 'normal',
            
            # Figure parameters
            'figure.figsize': (15, 12),
            'figure.dpi': 600,
            'figure.titlesize': 22,
            'figure.titleweight': 'bold',
            
            # Axes parameters
            'axes.titlesize': 20,
            'axes.titleweight': 'bold',
            'axes.labelsize': 18,
            'axes.labelweight': 'bold',
            'axes.linewidth': 1.5,
            'axes.edgecolor': '#333333',
            
            # Tick parameters
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            
            # Legend parameters
            'legend.fontsize': 16,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '#333333',
            
            # Grid parameters
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            
            # Saving parameters
            'savefig.dpi': 600,
            'savefig.format': 'png',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })
        
        # Create figure with suptitle
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('Training Dynamics', fontsize=24, fontweight='bold', y=0.98)
        
        # Define professional color palette
        colors = {
            'train_loss': '#1F77B4',        # Blue
            'val_loss': '#FF7F0E',          # Orange
            'train_class_loss': '#2CA02C',  # Green
            'val_class_loss': '#D62728',    # Red
            'train_sim_loss': '#9467BD',    # Purple
            'val_sim_loss': '#8C564B',      # Brown
            'accuracy': '#17BECF',          # Cyan
            'best_acc': '#E377C2',          # Pink
            'learning_rate': '#7F7F7F'      # Gray
        }
        
        # Calculate epochs
        epochs = range(1, len(self.metrics['train_losses']) + 1)
        
        # Plot training and validation loss
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(epochs, self.metrics['train_losses'], '-', color=colors['train_loss'], linewidth=2.5, label='Train Loss')
        ax1.plot(epochs, self.metrics['val_losses'], '-', color=colors['val_loss'], linewidth=2.5, label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=18, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.set_title('Training and Validation Loss', fontsize=20, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Add annotations for min values
        min_train_loss = min(self.metrics['train_losses'])
        min_train_epoch = self.metrics['train_losses'].index(min_train_loss) + 1
        min_val_loss = min(self.metrics['val_losses'])
        min_val_epoch = self.metrics['val_losses'].index(min_val_loss) + 1
        
        ax1.annotate(f'Min: {min_train_loss:.4f}',
                    xy=(min_train_epoch, min_train_loss),
                    xytext=(min_train_epoch+2, min_train_loss+0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=14, fontweight='bold')
        
        ax1.annotate(f'Min: {min_val_loss:.4f}',
                    xy=(min_val_epoch, min_val_loss),
                    xytext=(min_val_epoch+2, min_val_loss+0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=14, fontweight='bold')
        
        # Plot classification and similarity losses
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(epochs, self.metrics['train_class_losses'], '-', color=colors['train_class_loss'], linewidth=2.5, label='Train Classification')
        ax2.plot(epochs, self.metrics['val_class_losses'], '-', color=colors['val_class_loss'], linewidth=2.5, label='Val Classification')
        ax2.plot(epochs, self.metrics['train_sim_losses'], '--', color=colors['train_sim_loss'], linewidth=2.5, label='Train Similarity')
        ax2.plot(epochs, self.metrics['val_sim_losses'], '--', color=colors['val_sim_loss'], linewidth=2.5, label='Val Similarity')
        ax2.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=18, fontweight='bold')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.set_title('Component Losses', fontsize=20, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # # Plot accuracy
        # ax3 = plt.subplot(2, 2, 3)
        # ax3.plot(epochs, self.metrics['accuracies'], '-', color=colors['accuracy'], linewidth=3.0, label='Validation Accuracy')
        
        # # Find best accuracy and its epoch
        # best_accuracy = max(self.metrics['accuracies'])
        # best_epoch = self.metrics['accuracies'].index(best_accuracy) + 1
        
        # # Add horizontal line for best accuracy
        # ax3.axhline(y=best_accuracy, color=colors['best_acc'], linestyle='--', linewidth=2.0, 
        #             label=f'Best: {best_accuracy:.2f}%')
        
        # # Add annotation for best accuracy
        # ax3.annotate(f'Best: {best_accuracy:.2f}% (Epoch {best_epoch})',
        #             xy=(best_epoch, best_accuracy),
        #             xytext=(best_epoch-5, best_accuracy-8),
        #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        #             fontsize=14, fontweight='bold')
        
        # ax3.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        # ax3.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
        # ax3.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        # ax3.set_title('Validation Accuracy', fontsize=20, fontweight='bold')
        # ax3.grid(True, linestyle='--', alpha=0.3)
        
        # # Format y-axis as percentage
       
        # ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # # Plot learning rate if available
        # if self.metrics['learning_rates']:
        #     ax4 = plt.subplot(2, 2, 4)
        #     ax4.plot(epochs, self.metrics['learning_rates'], '-', color=colors['learning_rate'], linewidth=2.5, label='Learning Rate')
        #     ax4.set_xlabel('Epoch', fontsize=18, fontweight='bold')
        #     ax4.set_ylabel('Learning Rate', fontsize=18, fontweight='bold')
        #     ax4.set_yscale('log')
        #     ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        #     ax4.set_title('Learning Rate Schedule', fontsize=20, fontweight='bold')
        #     ax4.grid(True, linestyle='--', alpha=0.3)
            
        #     # Format y-axis in scientific notation
        #     ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save figure with timestamp and high resolution
        output_path = self.exp_dir / 'training_curves_publication.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        # Save a vector version for publication
        vector_path = self.exp_dir / 'training_curves_publication.pdf'
        plt.savefig(vector_path, format='pdf', bbox_inches='tight')
        
        self.logger.info(f"Publication-quality training curves saved to {output_path} and {vector_path}")
        plt.close()
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss, val_accuracy):
        """Save model checkpoint"""
        # Update best metrics
        is_best = val_loss < self.metrics['best_val_loss']
        
        if is_best:
            self.metrics['best_val_loss'] = val_loss
            self.metrics['best_val_accuracy'] = val_accuracy
            self.metrics['best_epoch'] = epoch
            
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'config': self.config
        }
        
        # Save based on configuration
        save_best_only = self.config.get('save_best_only', False)
        
        if save_best_only:
            if is_best:
                torch.save(checkpoint, self.exp_dir / 'best_model.pt')
                self.logger.info(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%")
        else:
            # Save latest checkpoint
            torch.save(checkpoint, self.exp_dir / 'latest_model.pt')
            
            # Save interval checkpoints
            checkpoint_interval = self.config.get('checkpoint_interval', 5)
            if epoch % checkpoint_interval == 0 or is_best:
                torch.save(checkpoint, self.exp_dir / f'checkpoint_epoch_{epoch}.pt')
                self.logger.info(f"Saved checkpoint for epoch {epoch}")
                
            # Always save best model
            if is_best:
                torch.save(checkpoint, self.exp_dir / 'best_model.pt')
                self.logger.info(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%")
    
    def close(self):
        """Close the logger and generate final plots"""
        self.writer.close()
        self.plot_metrics()
        
        # Log best performance
        if self.metrics['best_epoch'] >= 0:
            self.logger.info(f"Best performance at epoch {self.metrics['best_epoch']}")
            self.logger.info(f"Best validation loss: {self.metrics['best_val_loss']:.4f}")
            self.logger.info(f"Best validation accuracy: {self.metrics['best_val_accuracy']:.2f}%")


def get_loss_function(loss_name, **kwargs):
    """
    Factory function to create a loss function based on configuration
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        torch.nn.Module: Loss function
    """
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'BCELoss':
        return nn.BCELoss(**kwargs)
    elif loss_name == 'MSELoss':
        return nn.MSELoss(**kwargs)
    elif loss_name == 'L1Loss':
        return nn.L1Loss(**kwargs)
    elif loss_name == 'SmoothL1Loss':
        return nn.SmoothL1Loss(**kwargs)
    elif loss_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def get_optimizer(optimizer_name, model_parameters, config):
    """
    Factory function to create an optimizer based on configuration
    
    Args:
        optimizer_name: Name of the optimizer
        model_parameters: Model parameters to optimize
        config: Configuration dictionary
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'Adam':
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        return optim.Adam(
            model_parameters, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
    elif optimizer_name == 'SGD':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(
            model_parameters, 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'AdamW':
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        return optim.AdamW(
            model_parameters, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
    elif optimizer_name == 'RMSprop':
        alpha = config.get('alpha', 0.99)
        return optim.RMSprop(
            model_parameters, 
            lr=lr, 
            alpha=alpha,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, config):
    """
    Factory function to create a learning rate scheduler based on configuration
    
    Args:
        scheduler_name: Name of the scheduler
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    if scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('scheduler_mode', 'min'),
            factor=config.get('scheduler_factor', 0.1),
            patience=config.get('scheduler_patience', 5),

            min_lr=config.get('scheduler_min_lr', 1e-6)
        )
    elif scheduler_name == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('scheduler_step_size', 10),
            gamma=config.get('scheduler_gamma', 0.1)
        )
    elif scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('scheduler_T_max', 10),
            eta_min=config.get('scheduler_min_lr', 0)
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('scheduler_T_0', 10),
            T_mult=config.get('scheduler_T_mult', 1),
            eta_min=config.get('scheduler_min_lr', 0)
        )
    else:
        return None


def train_model(model, train_loader, val_loader, config):
    """
    Train a model with the given configuration
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        
    Returns:
        model: Trained model
        logger: Experiment logger
    """
    # Initialize experiment logger
    logger = ExperimentLogger(config)
    
    # Log experiment configuration
    logger.log_hyperparameters(config)
    
    # Set device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    # Get number of epochs
    num_epochs = config.get('num_epochs', 50)
    
    # Get loss functions
    classification_loss_name = config.get('classification_loss', 'CrossEntropyLoss')
    similarity_loss_name = config.get('similarity_loss', 'MSELoss')
    classification_weight = config.get('classification_weight', 1.0)
    similarity_weight = config.get('similarity_weight', 1.0)
    
    classification_criterion = get_loss_function(classification_loss_name)
    similarity_criterion = get_loss_function(similarity_loss_name)
    
    # Get optimizer
    optimizer_name = config.get('optimizer', 'Adam')
    optimizer = get_optimizer(optimizer_name, model.parameters(), config)
    
    # Get scheduler
    scheduler_name = config.get('scheduler', 'ReduceLROnPlateau')
    scheduler = get_scheduler(scheduler_name, optimizer, config)
    
    # Early stopping settings
    early_stopping = config.get('early_stopping', False)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    # For logging
    log_batch_interval = config.get('log_batch_interval', 10)
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training", total=num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_class_loss = 0.0
        total_sim_loss = 0.0
        
        for batch_idx, (prints, references, labels) in enumerate(train_loader):
            prints, references = prints.to(device), references.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            classifications, similarities = model(prints, references)
            
            # Calculate losses
            class_loss = classification_criterion(classifications, labels)
            # For similarity, we expect good prints to be more similar to reference
            similarity_target = (labels == 1).float().unsqueeze(1)
            sim_loss = similarity_criterion(similarities, similarity_target)
            
            # Combined loss
            loss = classification_weight * class_loss + similarity_weight * sim_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_class_loss += class_loss.item()
            total_sim_loss += sim_loss.item()
            
            # Log batch metrics
            if batch_idx % log_batch_interval == 0:
                logger.log_metrics({
                    'batch_loss': loss.item(),
                    'batch_classification_loss': class_loss.item(),
                    'batch_similarity_loss': sim_loss.item()
                }, epoch * len(train_loader) + batch_idx, 'train')
        
        # Calculate average training metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)   
        avg_sim_loss = total_sim_loss / len(train_loader)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log training metrics
        logger.log_metrics({
            'loss': avg_train_loss, 
            'classification_loss': avg_class_loss, 
            'similarity_loss': avg_sim_loss,
            'lr': current_lr
        }, epoch, 'train')
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_class_loss = 0.0
        total_sim_loss = 0.0
        correct = 0
        total = 0
        epoch_val_preds = []
        epoch_val_labels = []
        
        with torch.no_grad():
            for prints, references, labels in val_loader:
                prints, references = prints.to(device), references.to(device)
                labels = labels.to(device)
                
                classifications, similarities = model(prints, references)
                
                # Calculate validation loss
                class_loss = classification_criterion(classifications, labels)
                similarity_target = (labels == 1).float().unsqueeze(1)
                sim_loss = similarity_criterion(similarities, similarity_target)
                loss = classification_weight * class_loss + similarity_weight * sim_loss
                
                total_val_loss += loss.item()
                total_class_loss += class_loss.item()
                total_sim_loss += sim_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(classifications.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for confusion matrix
                epoch_val_preds.append(predicted)
                epoch_val_labels.append(labels)
        
        # Calculate average validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        avg_sim_loss = total_sim_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Log validation metrics
        logger.log_metrics({
            'loss': avg_val_loss, 
            'classification_loss': avg_class_loss, 
            'similarity_loss': avg_sim_loss, 
            'accuracy': accuracy
        }, epoch, 'validation')
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Training Classification Loss: {avg_class_loss:.4f}')
        print(f'Training Similarity Loss: {avg_sim_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Classification Loss: {avg_class_loss:.4f}')
        print(f'Validation Similarity Loss: {avg_sim_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        logger.save_checkpoint(model, optimizer, epoch, avg_val_loss, accuracy)
            
        # Log predictions and confusion matrix at specified intervals
        checkpoint_interval = config.get('checkpoint_interval', 5)
        if epoch % checkpoint_interval == 0 and epoch_val_preds:
            # Concatenate predictions and labels from this epoch
            epoch_val_preds = torch.cat(epoch_val_preds)
            epoch_val_labels = torch.cat(epoch_val_labels)
            
            logger.log_confusion_matrix(
                epoch_val_labels.cpu().numpy(),
                epoch_val_preds.cpu().numpy(),
                epoch
            )
            
            # Generate classification report
            report = classification_report(
                epoch_val_labels.cpu().numpy(),
                epoch_val_preds.cpu().numpy(),
                target_names=['Bad Print', 'Good Print']
            )
            logger.logger.info(f"\nClassification Report - Epoch {epoch}:\n{report}")
        
        # Early stopping
        if early_stopping:
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logger.logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    logger.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
    
    # Close logger and generate final plots
    logger.close()
    
    return model, logger