# import torch
# import torch.nn as nn
# from torchvision import models

# class QualityControlNet(nn.Module):
#     def __init__(self, config):
#         """
#         Initialize the Quality Control Network with configurable parameters
        
#         Args:
#             config: Configuration dictionary
#         """
#         super(QualityControlNet, self).__init__()
        
#         # Parse config
#         backbone_name = config.get('backbone', 'resnet18')
#         pretrained = config.get('pretrained', True)
#         dropout_rate = config.get('dropout_rate', 0.5)
#         num_classes = config.get('num_classes', 2)
        
#         # Load backbones based on configuration
#         self.print_backbone = self._get_backbone(backbone_name, pretrained)
#         self.reference_backbone = self._get_backbone(backbone_name, pretrained)
        
#         # Get feature dimension - this is crucial for correct operation
#         feature_dim = self._get_feature_dim(self.print_backbone)
        
#         # Remove the last layer properly based on backbone type
#         if hasattr(self.print_backbone, 'fc'):
#             # For ResNet family
#             self.print_backbone.fc = nn.Identity()
#             self.reference_backbone.fc = nn.Identity()
#         elif hasattr(self.print_backbone, 'classifier'):
#             # For EfficientNet, MobileNet
#             self.print_backbone.classifier = nn.Identity()
#             self.reference_backbone.classifier = nn.Identity()
        
#         # Important: Update the configuration with the actual feature dimension
#         # This ensures the head layers are built with the correct input size
#         config['feature_dim'] = feature_dim
        
#         # Build similarity head from config
#         sim_head_layers = config.get('similarity_head_layers', [feature_dim * 2, 256, 1])
#         # Make sure the first layer has the correct input dimension
#         sim_head_layers[0] = feature_dim * 2
        
#         self.similarity_head = self._build_mlp(
#             input_dim=feature_dim * 2,
#             hidden_dims=sim_head_layers[1:-1],
#             output_dim=sim_head_layers[-1],
#             dropout_rate=dropout_rate,
#             final_activation=nn.Sigmoid()
#         )
        
#         # Build classification head from config
#         cls_head_layers = config.get('classification_head_layers', [feature_dim * 2, 256, num_classes])
#         # Make sure the first layer has the correct input dimension
#         cls_head_layers[0] = feature_dim * 2
        
#         self.classification_head = self._build_mlp(
#             input_dim=feature_dim * 2,
#             hidden_dims=cls_head_layers[1:-1],
#             output_dim=cls_head_layers[-1],
#             dropout_rate=dropout_rate
#         )

#     def _get_backbone(self, backbone_name, pretrained):
#         """Get backbone model based on name"""
#         if backbone_name == 'resnet18':
#             return models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
#         elif backbone_name == 'resnet34':
#             return models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
#         elif backbone_name == 'resnet50':
#             return models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
#         elif backbone_name == 'efficientnet_b0':
#             return models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
#         elif backbone_name == 'efficientnet_b1':
#             return models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
#         elif backbone_name == 'mobilenet_v2':
#             return models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
#         else:
#             raise ValueError(f"Unsupported backbone: {backbone_name}")
    
#     def _get_feature_dim(self, backbone):
#         """Get feature dimension from backbone"""
#         if hasattr(backbone, 'fc'):
#             # ResNet family
#             return backbone.fc.in_features
#         elif hasattr(backbone, 'classifier'):
#             # MobileNet, EfficientNet
#             if isinstance(backbone.classifier, nn.Sequential):
#                 # MobileNetV2 has a sequential classifier
#                 # We need to find the first Linear layer to get input features
#                 for layer in backbone.classifier:
#                     if isinstance(layer, nn.Linear):
#                         return layer.in_features
#                 # Fallback for MobileNetV2
#                 return 1280
#             else:
#                 # EfficientNet has a single Linear layer as classifier
#                 return backbone.classifier.in_features
#         else:
#             raise ValueError(f"Unsupported backbone type: {type(backbone)}")
    
#     def _build_mlp(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5, final_activation=None):
#         """
#         Build a multi-layer perceptron
        
#         Args:
#             input_dim: Input dimension
#             hidden_dims: List of hidden dimensions
#             output_dim: Output dimension
#             dropout_rate: Dropout rate
#             final_activation: Activation function for the final layer
            
#         Returns:
#             nn.Sequential: MLP model
#         """
#         layers = []
        
#         # Input layer
#         layers.append(nn.Linear(input_dim, hidden_dims[0] if hidden_dims else output_dim))
#         if hidden_dims:
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_rate))
            
#             # Hidden layers
#             for i in range(len(hidden_dims) - 1):
#                 layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
#                 layers.append(nn.ReLU())
#                 layers.append(nn.Dropout(dropout_rate))
                
#             # Output layer
#             layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
#         # Final activation (if provided)
#         if final_activation:
#             layers.append(final_activation)
            
#         return nn.Sequential(*layers)

#     def forward(self, print_img, ref_img):
#         # Extract features
#         print_features = self.print_backbone(print_img)
#         ref_features = self.reference_backbone(ref_img)
        
#         # Concatenate features
#         combined_features = torch.cat((print_features, ref_features), dim=1)
        
#         # Get similarity score and classification
#         similarity = self.similarity_head(combined_features)
#         classification = self.classification_head(combined_features)
        
#         return classification, similarity


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class QualityControlNet(nn.Module):
    def __init__(self, config):
        """
        Initialize the Quality Control Network with configurable parameters
        
        Args:
            config: Configuration dictionary
        """
        super(QualityControlNet, self).__init__()
        
        # Parse config
        backbone_name = config.get('backbone', 'resnet18')
        pretrained = config.get('pretrained', True)
        dropout_rate = config.get('dropout_rate', 0.5)
        num_classes = config.get('num_classes', 2)
        use_shared_backbone = config.get('use_shared_backbone', True)
        use_bilinear = config.get('use_bilinear', True)
        bilinear_out_features = config.get('bilinear_out_features', 128)
        
        # Load backbone(s) - shared or separate
        if use_shared_backbone:
            self.backbone = self._get_backbone(backbone_name, pretrained)
            self.print_backbone = self.backbone
            self.reference_backbone = self.backbone
        else:
            self.print_backbone = self._get_backbone(backbone_name, pretrained)
            self.reference_backbone = self._get_backbone(backbone_name, pretrained)
        
        # Get feature dimension
        feature_dim = self._get_feature_dim(self.print_backbone)
        
        # Remove the last layer properly based on backbone type
        if hasattr(self.print_backbone, 'fc'):
            # For ResNet family
            self.print_backbone.fc = nn.Identity()
            if not use_shared_backbone:
                self.reference_backbone.fc = nn.Identity()
        elif hasattr(self.print_backbone, 'classifier'):
            # For EfficientNet, MobileNet
            self.print_backbone.classifier = nn.Identity()
            if not use_shared_backbone:
                self.reference_backbone.classifier = nn.Identity()
        
        # Calculate combined feature dimension for rich interactions
        # [a, b, |a-b|, a*b] = 4 * feature_dim
        combined_dim = 4 * feature_dim
        
        # Optional bilinear interaction term
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.bilinear = nn.Bilinear(feature_dim, feature_dim, bilinear_out_features)
            combined_dim += bilinear_out_features
        
        # Update config with actual dimensions
        config['feature_dim'] = feature_dim
        config['combined_dim'] = combined_dim
        
        # Build similarity head from config
        sim_head_layers = config.get('similarity_head_layers', [combined_dim, 256, 1])
        sim_head_layers[0] = combined_dim  # Ensure correct input dimension
        
        # Note: Remove final sigmoid - use BCEWithLogitsLoss instead
        self.similarity_head = self._build_mlp(
            input_dim=combined_dim,
            hidden_dims=sim_head_layers[1:-1],
            output_dim=sim_head_layers[-1],
            dropout_rate=dropout_rate,
            final_activation=None  # No sigmoid - output logits
        )
        
        # Build classification head from config
        cls_head_layers = config.get('classification_head_layers', [combined_dim, 256, num_classes])
        cls_head_layers[0] = combined_dim  # Ensure correct input dimension
        
        self.classification_head = self._build_mlp(
            input_dim=combined_dim,
            hidden_dims=cls_head_layers[1:-1],
            output_dim=cls_head_layers[-1],
            dropout_rate=dropout_rate
        )

    def _get_backbone(self, backbone_name, pretrained):
        """Get backbone model based on name"""
        if backbone_name == 'resnet18':
            return models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone_name == 'resnet34':
            return models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone_name == 'resnet50':
            return models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone_name == 'efficientnet_b0':
            return models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone_name == 'efficientnet_b1':
            return models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone_name == 'mobilenet_v2':
            return models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _get_feature_dim(self, backbone):
        """Get feature dimension from backbone"""
        if hasattr(backbone, 'fc'):
            # ResNet family
            return backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            # MobileNet, EfficientNet
            if isinstance(backbone.classifier, nn.Sequential):
                # MobileNetV2 has a sequential classifier
                for layer in backbone.classifier:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
                # Fallback for MobileNetV2
                return 1280
            else:
                # EfficientNet has a single Linear layer as classifier
                return backbone.classifier.in_features
        else:
            raise ValueError(f"Unsupported backbone type: {type(backbone)}")
    
    def _build_mlp(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5, final_activation=None):
        """
        Build a multi-layer perceptron
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout_rate: Dropout rate
            final_activation: Activation function for the final layer
            
        Returns:
            nn.Sequential: MLP model
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0] if hidden_dims else output_dim))
        if hidden_dims:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                
            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Final activation (if provided)
        if final_activation:
            layers.append(final_activation)
            
        return nn.Sequential(*layers)

    def forward(self, print_img, ref_img):
        # Extract features
        print_features = self.print_backbone(print_img)
        ref_features = self.reference_backbone(ref_img)
        
        # L2 normalize features for stable training
        a = F.normalize(print_features, dim=1)
        b = F.normalize(ref_features, dim=1)
        
        # Rich vector interactions: [a, b, |a-b|, a*b]
        diff = torch.abs(a - b)  # Element-wise absolute difference
        prod = a * b             # Element-wise product
        pair = torch.cat([a, b, diff, prod], dim=1)
        
        # Optional bilinear interaction term
        if self.use_bilinear:
            bil = self.bilinear(a, b)
            combined_features = torch.cat([pair, bil], dim=1)
        else:
            combined_features = pair
        
        # Get similarity score (logits) and classification
        similarity_logits = self.similarity_head(combined_features)
        classification_logits = self.classification_head(combined_features)
        
        return classification_logits, similarity_logits



def get_model(config):
    """
    Factory function to create a model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        nn.Module: Model instance
    """
    model_arch = config.get('model_architecture', 'QualityControlNet')
    
    if model_arch == 'QualityControlNet':
        return QualityControlNet(config)
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")


# Test function to verify the model works with different backbones
def test_model_with_different_backbones():
    """Test the model with different backbone architectures"""
    import torch
    
    # Test configuration
    test_configs = [
        {
            'backbone': 'resnet18',
            'expected_feature_dim': 512
        },
        {
            'backbone': 'resnet50',
            'expected_feature_dim': 2048
        },
        {
            'backbone': 'mobilenet_v2',
            'expected_feature_dim': 1280
        },
        {
            'backbone': 'efficientnet_b0',
            'expected_feature_dim': 1280
        }
    ]
    
    for test_config in test_configs:
        print(f"\nTesting with backbone: {test_config['backbone']}")
        
        config = {
            'backbone': test_config['backbone'],
            'pretrained': False,  # False for faster testing
            'num_classes': 2,
            'dropout_rate': 0.5
        }
        
        model = QualityControlNet(config)
        
        # Test forward pass
        batch_size = 4
        img_size = 224
        print_img = torch.randn(batch_size, 3, img_size, img_size)
        ref_img = torch.randn(batch_size, 3, img_size, img_size)
        
        classification, similarity = model(print_img, ref_img)
        
        print(f"Classification output shape: {classification.shape}")
        print(f"Similarity output shape: {similarity.shape}")
        
        # Verify the feature dimension
        actual_feature_dim = model._get_feature_dim(model.print_backbone)
        expected_feature_dim = test_config['expected_feature_dim']
        assert actual_feature_dim == expected_feature_dim, \
            f"Expected feature dim {expected_feature_dim}, got {actual_feature_dim}"
        
        print(f"Feature dimension verified: {actual_feature_dim}")
        print("Test passed!")


if __name__ == "__main__":
    test_model_with_different_backbones()