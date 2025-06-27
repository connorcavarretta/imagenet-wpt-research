# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import get_imagenet_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None, num_classes=1000).to(device)

# Load data
train_loader, val_loader = get_imagenet_loaders(
    data_dir=os.environ.get("IMAGENET_PATH", "/mnt/imagenet"),
    batch_size=128,
    num_workers=8,
)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Training loop
for epoch in range(10):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Train Acc: {acc:.2f}%")
