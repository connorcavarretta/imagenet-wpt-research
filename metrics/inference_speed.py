import torch
import time

def measure_latency(model, input_tensor, runs=100):
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_tensor)

        # Timing
        start = time.time()
        for _ in range(runs):
            _ = model(input_tensor)
        end = time.time()
    
    avg_latency = (end - start) / runs
    return avg_latency * 1000  # milliseconds

def measure_throughput(model, batch_tensor):
    model.eval()
    with torch.no_grad():
        start = time.time()
        _ = model(batch_tensor)
        end = time.time()
    return batch_tensor.size(0) / (end - start)
