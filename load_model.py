import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, ResNet50_Weights
from PIL import Image

class Predictor:
    def __init__(self, model_path, attribute_count, device='cpu'):
        self.device = device
        self.model = self._load_model(model_path, attribute_count, device)
        self.transform = self._get_transform()

    def _load_model(self, model_path, attribute_count, device):
        # Load pre-trained ResNet-50 model
        resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modify the model's fully connected layer for the specified number of attributes
        num_ftrs = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Linear(num_ftrs, attribute_count)

        # Load the pre-trained weights from the specified path
        resnet50_model.load_state_dict(torch.load(model_path))

        # Move the model to the specified device and set to evaluation mode
        resnet50_model = resnet50_model.to(device)
        resnet50_model.eval()

        return resnet50_model

    def _get_transform(self):
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def predict(self, image_array):

        image_array = Image.fromarray(image_array)
        # Apply transformations
        input_tensor = self.transform(image_array)
        input_batch = input_tensor.unsqueeze(0)

        # Move input tensor to the device
        input_batch = input_batch.to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)[0]

        return output.numpy()