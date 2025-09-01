import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import json
import cv2

# Local imports
from model import get_model
from data_utils import get_transforms


def load_model(model_path):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {'model_architecture': 'QualityControlNet'}
        print("Warning: No config found in checkpoint. Using default config.")
    
    # Create model
    model = get_model(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def process_image(image_path, config):
    """
    Load and preprocess an image for inference
    
    Args:
        image_path: Path to image file
        config: Model configuration
        
    Returns:
        torch.Tensor: Processed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get validation transforms (no augmentation)
    transform = get_transforms(config, is_training=False)
    
    # Apply transforms
    processed = transform(image=np.array(image))['image']
    
    # Add batch dimension
    processed = processed.unsqueeze(0)
    
    return processed


def process_reference(reference_path, config):
    """
    Load and preprocess a reference image for inference
    
    Args:
        reference_path: Path to reference image file
        config: Model configuration
        
    Returns:
        torch.Tensor: Processed reference tensor
    """
    # Load image
    reference = Image.open(reference_path).convert('RGB')
    
    # Get image size from config
    img_size = config.get('img_size', 224)
    
    # Apply transforms
    reference_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    processed = reference_transform(reference)
    
    # Add batch dimension
    processed = processed.unsqueeze(0)
    
    return processed


def predict(model, image_tensor, reference_tensor, device='cpu'):
    """
    Make a prediction for a single image
    
    Args:
        model: Trained model
        image_tensor: Processed image tensor
        reference_tensor: Processed reference tensor
        device: Device to use for inference
        
    Returns:
        prediction: Class prediction (0=bad, 1=good)
        confidence: Confidence score
        similarity: Similarity score
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    reference_tensor = reference_tensor.to(device)
    
    with torch.no_grad():
        classification, similarity = model(image_tensor, reference_tensor)
        
        # Get prediction and confidence
        probabilities = torch.softmax(classification, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        
    return prediction.item(), confidence.item(), similarity.item()


def visualize_prediction(image_path, reference_path, prediction, confidence, similarity):
    """
    Visualize prediction results
    
    Args:
        image_path: Path to original image
        reference_path: Path to reference image
        prediction: Class prediction (0=bad, 1=good)
        confidence: Confidence score
        similarity: Similarity score
    """
    # Load images
    image = Image.open(image_path).convert('RGB')
    reference = Image.open(reference_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images
    ax1.imshow(image)
    ax1.set_title('Print Image')
    ax1.axis('off')
    
    ax2.imshow(reference)
    ax2.set_title('Reference Image')
    ax2.axis('off')
    
    # Add prediction text
    prediction_text = "GOOD PRINT" if prediction == 1 else "BAD PRINT"
    color = "green" if prediction == 1 else "red"
    
    plt.figtext(
        0.5, 0.01, 
        f"Prediction: {prediction_text} (Confidence: {confidence:.2f}, Similarity: {similarity:.2f})",
        ha="center", fontsize=14, bbox={"facecolor": color, "alpha": 0.2, "pad": 5}
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save result
    output_path = Path(f"prediction_result_{Path(image_path).stem}.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    # Show figure
    plt.show()


def batch_inference(model_path, data_dir, output_dir=None, visualize=False):
    """
    Run inference on all images in a directory
    
    Args:
        model_path: Path to model checkpoint
        data_dir: Directory containing shape folders
        output_dir: Directory to save results
        visualize: Whether to visualize predictions
    """
    # Load model
    model, config = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all shape directories
    data_dir = Path(data_dir)
    shape_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('ML_')]
    
    results = []
    
    for shape_dir in shape_dirs:
        shape_type = shape_dir.name.replace('ML_', '')
        
        # Find reference image
        reference_files = list(shape_dir.glob('*reference*.png')) + list(shape_dir.glob('*reference*.jpg'))
        if not reference_files:
            print(f"Warning: No reference image found for {shape_type}. Skipping.")
            continue
            
        reference_path = str(reference_files[0])
        reference_tensor = process_reference(reference_path, config)
        
        # Find all print images
        print_images = []
        for subdir in ['Good', 'Bad']:
            img_dir = shape_dir / subdir
            if img_dir.exists():
                for img_path in img_dir.glob('*.png'):
                    print_images.append({
                        'path': str(img_path),
                        'ground_truth': 1 if subdir == 'Good' else 0,
                        'shape_type': shape_type
                    })
        
        # Process each print image
        for img_data in print_images:
            img_path = img_data['path']
            ground_truth = img_data['ground_truth']
            
            # Process image
            image_tensor = process_image(img_path, config)
            
            # Make prediction
            prediction, confidence, similarity = predict(
                model, image_tensor, reference_tensor, device
            )
            
            # Add to results
            result = {
                'image_path': img_path,
                'reference_path': reference_path,
                'shape_type': shape_type,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'confidence': confidence,
                'similarity': similarity,
                'correct': prediction == ground_truth
            }
            results.append(result)
            
            # Print result
            print(f"Image: {Path(img_path).name}")
            print(f"Shape: {shape_type}")
            print(f"Ground truth: {'Good' if ground_truth == 1 else 'Bad'}")
            print(f"Prediction: {'Good' if prediction == 1 else 'Bad'}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Correct: {prediction == ground_truth}")
            print("-" * 50)
            
            # Visualize if requested
            if visualize:
                visualize_prediction(
                    img_path, reference_path, 
                    prediction, confidence, similarity
                )
    
    # Calculate overall accuracy
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"\nOverall Results:")
    print(f"Total images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Calculate per-shape accuracy
    shape_results = {}
    for r in results:
        shape = r['shape_type']
        if shape not in shape_results:
            shape_results[shape] = {'correct': 0, 'total': 0}
        shape_results[shape]['total'] += 1
        if r['correct']:
            shape_results[shape]['correct'] += 1
    
    print("\nPer-shape Results:")
    for shape, stats in shape_results.items():
        shape_accuracy = 100 * stats['correct'] / stats['total']
        print(f"{shape}: {shape_accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    # Save results to JSON
    if output_dir:
        results_file = output_dir / 'inference_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'overall': {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                },
                'shape_results': shape_results,
                'image_results': results
            }, f, indent=4)
        print(f"\nResults saved to {results_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='3D Print Quality Control Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference image (for single image inference)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing shape folders for batch inference')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    
    args = parser.parse_args()
    
    # Check arguments
    if args.image is not None and args.reference is not None:
        # Single image inference
        model, config = load_model(args.model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Process images
        image_tensor = process_image(args.image, config)
        reference_tensor = process_reference(args.reference, config)
        
        # Make prediction
        prediction, confidence, similarity = predict(
            model, image_tensor, reference_tensor, device
        )
        
        # Print result
        print(f"Image: {Path(args.image).name}")
        print(f"Prediction: {'Good' if prediction == 1 else 'Bad'}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Similarity: {similarity:.4f}")
        
        # Visualize
        if args.visualize:
            visualize_prediction(
                args.image, args.reference, 
                prediction, confidence, similarity
            )
    
    elif args.data_dir is not None:
        # Batch inference
        batch_inference(
            args.model, 
            args.data_dir, 
            args.output_dir, 
            args.visualize
        )
    
    else:
        parser.print_help()
        print("\nError: Please provide either --image and --reference for single image inference,")
        print("or --data_dir for batch inference.")


if __name__ == '__main__':
    main()