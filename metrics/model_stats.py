import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_MB(path):
    import os
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)

# Optional: using ptflops for MACs/FLOPs
def compute_flops(model, input_size=(3, 224, 224)):
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, input_size, as_strings=False)
    return macs, params
