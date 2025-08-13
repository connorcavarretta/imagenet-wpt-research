import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from timm.data import create_transform
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # robustness to occasional bad JPEGs


def _is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _rank_world():
    if _is_dist():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def get_imagenet_loaders(
    data_dir: str,
    batch_size: int = 64,         # per-process batch size
    num_workers: int = 8,
    use_mixup: bool = False,      # kept for API continuity (unused here)
    use_cutmix: bool = False,     # kept for API continuity (unused here)
    rand_augment: bool = True,
    distributed: bool = False,    # if True, uses DistributedSampler
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 4,
    seed: int = 42,
):
    """Builds train/val loaders for ImageNet with optional DDP sharding."""

    # --- transforms ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tfms = create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1' if rand_augment else None,
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # --- datasets ---
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)

    # --- distributed / samplers ---
    rank, world_size = _rank_world()
    use_dist = distributed and world_size > 1

    if use_dist:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True
        )
        # Shard validation as well; metrics are all-reduced in evaluate()
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    # --- worker seeding ---
    # Ensure different seeds per rank/worker, but deterministic base
    base_seed = seed + rank

    def _worker_init_fn(worker_id: int):
        wseed = base_seed * 1000 + worker_id
        random.seed(wseed)
        np.random.seed(wseed & 0x7FFFFFFF)
        torch.manual_seed(wseed)

    # torch.Generator for DataLoader shuffling when not using sampler
    g = torch.Generator()
    g.manual_seed(base_seed)

    # --- loaders ---
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )
    # Only pass prefetch_factor if workers > 0 and user provided it
    if num_workers > 0 and prefetch_factor is not None:
        common_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        drop_last=use_dist,   # keeps per-rank batch aligned
        generator=g if not use_dist else None,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        generator=g if not use_dist else None,
        **common_kwargs,
    )

    return train_loader, val_loader
