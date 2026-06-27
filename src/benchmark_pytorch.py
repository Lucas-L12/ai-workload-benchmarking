import torch
import torchvision.models as models
import numpy as np
import csv
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import measure_latency

BATCH_SIZES = [1, 4, 16, 64]
THREADS = [1, 2, 4, 8]
RUNS = 100
WARMUP = 10
RESULTS_PATH = Path("results/benchmark_pytorch.csv")


def benchmark_pytorch(model, batch_size, num_threads):
    torch.set_num_threads(num_threads)
    
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        fn = lambda: model(input_tensor)
        results = measure_latency(fn, runs=RUNS, warmup=WARMUP)
    
    throughput = (batch_size / results["mean_ms"]) * 1000
    
    return {
        "runtime": "pytorch",
        "batch_size": batch_size,
        "threads": num_threads,
        "mean_ms": results["mean_ms"],
        "std_ms": results["std_ms"],
        "min_ms": results["min_ms"],
        "max_ms": results["max_ms"],
        "throughput_img_s": throughput,
    }


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["runtime", "batch_size", "threads", "mean_ms", 
                  "std_ms", "min_ms", "max_ms", "throughput_img_s"]
    
    write_header = not path.exists()
    
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(results)


def main():
    print("Loading model...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    
    print(f"Starting PyTorch benchmark...")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Threads: {THREADS}")
    
    for batch_size in BATCH_SIZES:
        for num_threads in THREADS:
            print(f"  batch={batch_size}, threads={num_threads}...", end=" ")
            result = benchmark_pytorch(model, batch_size, num_threads)
            save_results(result, RESULTS_PATH)
            print(f"mean={result['mean_ms']:.1f}ms, throughput={result['throughput_img_s']:.1f}img/s")
    
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()