import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataloader import get_imagenet_loaders
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from pathlib import Path
from tqdm import tqdm


CHECKPOINT_DIR = "checkpoints/convnext_base"
LOG_PATH = "logs/convnext_base_training_log.csv"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")


def save_checkpoint(state, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, scheduler, model_ema):
    if not os.path.isfile(LATEST_CKPT):
        return 0  # Start from scratch

    print(f"Loading checkpoint from {LATEST_CKPT}")
    checkpoint = torch.load(LATEST_CKPT, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    model_ema.ema.load_state_dict(checkpoint['ema'])
    return checkpoint['epoch'] + 1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy


def log_epoch(epoch, train_loss, train_acc, val_loss, val_acc):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # Load ConvNeXt model
    from models.convnext import ConvNeXt as model
    model = model(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=1000).to(device)
    model_ema = ModelEmaV2(model, decay=0.9999, device=device)

    # Load data
    train_loader, val_loader = get_imagenet_loaders(
        data_dir=os.environ.get("IMAGENET_PATH", "../imagenet"),
        batch_size=64,
        num_workers=4,
        use_mixup=False,
        use_cutmix=False,
        rand_augment=True,
    )

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999), weight_decay=0.05)

    warmup_epochs = 20
    total_epochs = 300
    iters_per_epoch = len(train_loader)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs) * iters_per_epoch)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_epochs * iters_per_epoch)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs * iters_per_epoch])

    # Resume if checkpoint exists
    start_epoch = load_checkpoint(model, optimizer, scheduler, model_ema)
    best_val_acc = 0.0

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss, total, correct = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model_ema.update(model)

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = 100. * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        log_epoch(epoch+1, avg_train_loss, train_acc, val_loss, val_acc)

        # Save latest checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ema': model_ema.ema.state_dict()
        }, LATEST_CKPT)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': model_ema.ema.state_dict()
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"Saved new best model at epoch {epoch+1} with val acc: {val_acc:.2f}%")


if __name__ == '__main__':
    main()
