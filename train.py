# python -m torch.distributed.run --nproc_per_node=3 train.py --batch-size 256 --num-workers 8
# or: torchrun --nproc_per_node=3 train.py --batch-size 256 --num-workers 8

import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from tqdm import tqdm

from dataloader import get_imagenet_loaders

CHECKPOINT_DIR = "checkpoints/convnext_base"
LOG_PATH = "logs/convnext_base_training_log.csv"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")


# ----------------------------- utils -----------------------------

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main_process():
    return get_rank() == 0

def save_checkpoint(state, filename):
    if is_main_process():
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(state, filename)

def load_checkpoint(model, optimizer, scheduler, model_ema):
    if not os.path.isfile(LATEST_CKPT):
        return 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % torch.cuda.current_device()} if torch.cuda.is_available() else 'cpu'
    if is_main_process():
        print(f"Loading checkpoint from {LATEST_CKPT}")
    ckpt = torch.load(LATEST_CKPT, map_location=map_location)

    target = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    target.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    if model_ema is not None:
        model_ema.module.load_state_dict(ckpt['ema'])

    if is_dist():
        dist.barrier()
    return ckpt['epoch'] + 1

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    if is_dist():
        tensor = torch.tensor([total_loss, total_correct, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = tensor.tolist()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy

def log_epoch(epoch, train_loss, train_acc, val_loss, val_acc):
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}"])

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0; world_size = 1; local_rank = 0
    torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return local_rank, world_size


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt on ImageNet (DDP)")
    parser.add_argument('--batch-size', type=int, default=128, help='Per-process batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='Warmup epochs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    local_rank, world_size = init_distributed() if use_ddp else (0, 1)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"Using DDP: {use_ddp} | world_size={world_size} | device={device}")

    # Reproducibility
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    # Dataset path sanity (all ranks check; rank0 prints counts)
    data_root = os.environ.get("IMAGENET_PATH", "../imagenet")
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Missing ImageNet dirs: {train_dir} or {val_dir}")
    if is_main_process():
        n_train = sum(1 for e in os.scandir(train_dir) if e.is_dir())
        n_val   = sum(1 for e in os.scandir(val_dir) if e.is_dir())
        print(f"[RANK0] IMAGENET_PATH={data_root} | classes(train)={n_train} | classes(val)={n_val}")

    # Model
    from models.convnext import ConvNeXt as ConvNeXt
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=1000).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, gradient_as_bucket_view=True)

    # EMA on the unwrapped module
    ema_target = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    model_ema = ModelEmaV2(ema_target, decay=0.9999, device=device)

    # Data
    train_loader, val_loader = get_imagenet_loaders(
        data_dir=data_root,
        batch_size=args.batch_size,                # per-process batch
        num_workers=args.num_workers,
        use_mixup=False,
        use_cutmix=False,
        rand_augment=True,
        distributed=use_ddp,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Criterion / Optimizer (AdamW)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)

    # Linear LR scaling from base: 4e-3 at global batch 1024
    global_batch = args.batch_size * world_size
    base_lr = 4e-3
    lr = base_lr * (global_batch / 1024.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)

    # Schedulers (per-iter stepping)
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    iters_per_epoch = len(train_loader)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs) * iters_per_epoch)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_epochs * iters_per_epoch)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                             milestones=[warmup_epochs * iters_per_epoch])

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Resume
    start_epoch = load_checkpoint(model, optimizer, scheduler, model_ema)
    best_val_acc = 0.0

    # ----------------------------- train -----------------------------
    for epoch in range(start_epoch, total_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum, total, correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False) if is_main_process() else train_loader

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA update (use unwrapped module)
            ema_target = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            model_ema.update(ema_target)

            bs = images.size(0)
            train_loss_sum += loss.item() * bs
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += bs

        # Reduce training metrics across ranks
        train_totals = torch.tensor([train_loss_sum, correct, total], dtype=torch.float64, device=device)
        if is_dist():
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss_sum, correct, total = train_totals.tolist()
        avg_train_loss = train_loss_sum / total
        train_acc = 100.0 * correct / total

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if is_main_process():
            print(f"Epoch {epoch+1}: "
                  f"Train Loss {avg_train_loss:.4f} | Train Acc {train_acc:.2f}% | "
                  f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}%")

        log_epoch(epoch+1, avg_train_loss, train_acc, val_loss, val_acc)

        # Save latest
        save_checkpoint({
            'epoch': epoch,
            'model': (model.module if isinstance(model, (DDP, nn.DataParallel)) else model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ema': model_ema.module.state_dict(),
        }, LATEST_CKPT)

        # Save best
        if is_main_process() and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model': (model.module if isinstance(model, (DDP, nn.DataParallel)) else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': model_ema.module.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"Saved new best model at epoch {epoch+1} with val acc: {val_acc:.2f}%")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
