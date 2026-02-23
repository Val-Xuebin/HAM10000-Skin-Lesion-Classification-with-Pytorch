import os
import sys
import torch
from torch import nn
from torchvision import models
import timm

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from .custom_cnn import create_custom_cnn


def set_parameter_requires_grad(model, feature_extracting):
    """Freeze or unfreeze model parameters"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """Initialize pretrained model and modify final layer"""
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "custom_cnn":
        # Custom CNN model (trained from scratch, no pretrained weights)
        # Use default input size 224 for custom CNN
        input_size = 224
        model_ft = create_custom_cnn(
            num_classes=num_classes,
            input_size=input_size,
            feature_extract=feature_extract,
            use_pretrained=False  # Custom CNN doesn't have pretrained weights
        )
        # For custom CNN, feature_extract can freeze early layers if needed
        if feature_extract:
            # Freeze convolutional layers, only train FC layers
            for name, param in model_ft.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    elif model_name == "swin":
        # Swin Transformer model using timm library
        model_ft = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=use_pretrained,
            num_classes=num_classes
        )
        # For Swin Transformer, we need to handle feature extraction differently
        # because timm models use 'head' as the classifier name
        if feature_extract:
            # Freeze all backbone parameters
            for name, param in model_ft.named_parameters():
                # Unfreeze only the head (classifier) layer
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        print(f"Available models: resnet, vgg, densenet, inception, custom_cnn, swin")
        exit()

    return model_ft, input_size

