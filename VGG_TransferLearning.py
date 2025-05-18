import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained VGG16
vgg16 = models.vgg16(pretrained=True)

# Modify the classifier for binary classification
vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Unfreeze only block5 layers
for param in vgg16.features.parameters():
    param.requires_grad = False
for layer in list(vgg16.features.children())[24:]:
    for param in layer.parameters():
        param.requires_grad = True

vgg16 = vgg16.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=1e-5)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Datasets and loaders
train_dataset = datasets.ImageFolder('path/to/train', transform=transform)
val_dataset = datasets.ImageFolder('path/to/val', transform=transform)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
}

# Training function
def train_model(model, dataloaders, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (outputs > 0.5).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print("\u2705 New best model saved.")
            elif phase == 'val' and epoch_loss > best_val_loss:
                print("\u26A0\uFE0F Possible overfitting: Validation loss increased.")

    print(f'\nBest Validation Loss: {best_val_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(vgg16, dataloaders, num_epochs=25)

# Optionally save the best model
# torch.save(model.state_dict(), 'best_vgg16_binary.pth')
