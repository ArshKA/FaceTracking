import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from deepface import DeepFace

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

class Predictor:
    def __init__(self, model_path, attribute_count, device='cpu'):
        self.device = device
        self.model = self._load_model(model_path, attribute_count, device)
        self.transform = self._get_transform()

    def _load_model(self, model_path, attribute_count, device):
        resnet50_model = resnet50()
        num_ftrs = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Linear(num_ftrs, attribute_count)
        resnet50_model.load_state_dict(torch.load(model_path, map_location=self.device))
        resnet50_model = resnet50_model.to(device)
        resnet50_model.eval()
        return resnet50_model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_deepface(self, predictions):
        # Define the order of keys for gender and race
        gender_order = ['Woman', 'Man']
        race_order = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

        # Initialize the output list for probabilities using OrderedDict to reinforce the intention
        probabilities = []

        # Extract gender probabilities in a specific order
        for gender in gender_order:
            probabilities.append(predictions['gender'].get(gender, 0))  # Default to 0 if key is missing

        # Extract race probabilities in a specific order
        for race in race_order:
            probabilities.append(predictions['race'].get(race, 0))  # Default to 0 if key is missing

        # Text attributes in a specific order, directly aligned with the order above


        return probabilities

    def deepface_analysis(self, image):
        # DeepFace expects PIL Image
        analysis = DeepFace.analyze(image, actions=['age', 'gender', 'race'], enforce_detection=False)
        if analysis:
            analysis = self.extract_deepface(analysis[0])
        else:
            analysis = [0]*8
        return np.array(analysis)/100

    def predict(self, image_array):
        image_pil = Image.fromarray(image_array)
        input_tensor = self.transform(image_pil)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)[0]
        output = F.sigmoid(output)

        deepface_output = self.deepface_analysis(image_array)
        print(deepface_output)
        full_output = np.concatenate((output.cpu().numpy(), deepface_output))
        full_output[17] = 0

        return full_output
