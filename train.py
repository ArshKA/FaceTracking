import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg19, VGG19_Weights, resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from data_loader import CelebADataset
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
run = 16




# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.RandomHorizontalFlip(),
    ], p=0.5),
    transforms.Resize((224, 224)),  # Resize images to 224x224 (ResNet-50 input size)
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values (ResNet-50 normalization)
])

root_dir = '/scratch/arsh/flawless'
image_path = '/scratch/arsh/flawless/img_align_celeba/img_align_celeba'

# Create a dataset instance
train_dataset = CelebADataset(root_dir, image_path, 'train', transform=transform)
test_dataset = CelebADataset(root_dir, image_path, 'test', transform=transform)


attribute_names = train_dataset.attr_df.columns
print(f"Training Size: {len(train_dataset)}, Test Size: {len(test_dataset)}")

# Define batch size for DataLoader
batch_size = 256

# Create DataLoader for training and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet-50 model
resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

num_attributes = len(attribute_names)  # Number of attributes
num_ftrs = resnet50_model.fc.in_features
resnet50_model.fc = nn.Linear(num_ftrs, num_attributes)
# resnet50_model.load_state_dict(torch.load('run4/epoch_7_loss_88.788.pth'))

resnet50_model = resnet50_model.to(device)


# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()#weight = weights)  # Binary cross-entropy loss
optimizer = optim.Adam(resnet50_model.parameters(), lr=0.01)

num_epochs = 50

if not os.path.exists(f'run{run}'):
    os.mkdir(f'run{run}')

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for i, data in enumerate(pbar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = resnet50_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        pbar.set_postfix({'Epoch': epoch + 1, 'Loss': running_loss / (i + 1)})  # Update progress bar

    print(running_loss)
    # Calculate precision and recall on the test set
    precision = torch.zeros(len(attribute_names), dtype=torch.float)
    recall = torch.zeros(len(attribute_names), dtype=torch.float)
    correct = torch.zeros(len(attribute_names), dtype=torch.float).to(device)
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = resnet50_model(images)
            test_loss += criterion(outputs, labels)
            predicted = torch.round(torch.sigmoid(outputs))

            # Compute precision and recall for each class
            correct += (predicted == labels).sum(dim=0)
            precision += torch.tensor(precision_score(labels.cpu(), predicted.cpu(), average=None, zero_division=0))
            recall += torch.tensor(recall_score(labels.cpu(), predicted.cpu(), average=None, zero_division=0))
            total += len(labels)


    accuracy = (correct / total) * 100
    precision /= len(test_dataloader)
    recall /= len(test_dataloader)

    print(f'Epoch [{epoch + 1}], Total Accuracy : {accuracy.mean():.2f}%, Total Loss: {test_loss:.3f}')

    # Print precision and recall of each class
    for idx, attribute_name in enumerate(attribute_names):
        print(f'Class: {attribute_name:20s}, Accuracy: {accuracy[idx]:.2f}%, Precision: {precision[idx]*100:.2f}%, Recall: {recall[idx]*100:.2f}%')

    # Save model state
    torch.save(resnet50_model.state_dict(), f'run{run}/epoch_{epoch}_loss_{test_loss:.3f}.pth')

print('Finished Training')

