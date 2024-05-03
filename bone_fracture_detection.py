import numpy as np
import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the directory paths
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = 'D:/Data for machine learning/Bone Fracture/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification'

# Define the directories for fractured and not fractured images
fractured_dir = "D:/Data for machine learning/Bone Fracture/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/fractured"
not_fractured_dir = "D:/Data for machine learning/Bone Fracture/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train/not fractured"

# Get list of files in each directory
fractured_files = os.listdir(fractured_dir)
not_fractured_files = os.listdir(not_fractured_dir)

# Randomly select 10 images from each directory
fractured_images = np.random.choice(fractured_files, 10, replace=False)
not_fractured_images = np.random.choice(not_fractured_files, 10, replace=False)

# Plot the images
plt.figure(figsize=(20, 10))
for i, img_name in enumerate(np.concatenate([fractured_images, not_fractured_images])):
    img_path = os.path.join(fractured_dir if img_name in fractured_images else not_fractured_dir, img_name)
    img = Image.open(img_path)
    plt.subplot(2, 10, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Fractured' if img_name in fractured_images else 'Not Fractured')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'sample_images.png'), dpi=300, facecolor='white')
plt.show()

# Define transformations to apply to the images
data_transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize images to a consistent size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to RGB
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

# Define function to load dataset
def load_dataset(root, transform=None):
    """Load dataset from given root directory."""
    dataset = datasets.ImageFolder(root=root, transform=transform)
    data = []
    targets = []
    for path, target in dataset.samples:
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                if transform is not None:
                    img = transform(img)
                data.append(img)
                targets.append(target)
        except OSError:
            print(f"Skipping {path} as it is corrupted or incomplete.")
    return torch.stack(data), torch.tensor(targets)

# Load the datasets
train_data, train_targets = load_dataset(os.path.join(data_dir, 'train'), transform=data_transform)
val_data, val_targets = load_dataset(os.path.join(data_dir, 'val'), transform=data_transform)
test_data, test_targets = load_dataset(os.path.join(data_dir, 'test'), transform=data_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_targets), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_data, val_targets), batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_targets), batch_size=32, shuffle=False)

# Define the CNN model
class ConvNet(nn.Module):
    """CNN Model for Bone Fracture Binary Classification."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 128),  # Adjusted based on the actual feature map size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define the number of classes
num_classes = 2

# Initialize the model
model = ConvNet(num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move data loaders to GPU
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data.to(device), train_targets.to(device)), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_data.to(device), val_targets.to(device)), batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data.to(device), test_targets.to(device)), batch_size=32, shuffle=False)

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
counter = 0

# Lists to store training statistics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(20):
    # Train the model
    model.train()
    epoch_train_losses = []
    epoch_train_predictions = []
    epoch_train_targets = []
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        epoch_train_predictions.extend(predicted.tolist())
        epoch_train_targets.extend(labels.tolist())
    train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    train_accuracy = accuracy_score(epoch_train_targets, epoch_train_predictions)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validate the model
    model.eval()
    epoch_val_losses = []
    epoch_val_predictions = []
    epoch_val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_val_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            epoch_val_predictions.extend(predicted.tolist())
            epoch_val_targets.extend(labels.tolist())
    val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
    val_accuracy = accuracy_score(epoch_val_targets, epoch_val_predictions)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # Print epoch statistics
    print(f'Epoch [{epoch+1}/20], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Evaluate the model on test dataset
model.eval()
test_losses = []
test_predictions = []
test_targets = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.tolist())
        test_targets.extend(labels.tolist())
test_loss = sum(test_losses) / len(test_losses)
test_accuracy = accuracy_score(test_targets, test_predictions)

# Print test statistics
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Plotting train and validation accuracy
plt.figure(figsize=(10, 5))
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss & Accuracy')
plt.title(f'Train and Validation Loss & Accuracy (Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f})')
plt.legend()
plt.savefig(os.path.join(script_dir, 'train_val_loss_acc_plot.png'), dpi=300, facecolor='white')
plt.show()

# Save model weights
torch.save(model.state_dict(), os.path.join(script_dir, 'bone_fracture_detection_model_weights.pth'))

print("Model weights and plot saved successfully!")
