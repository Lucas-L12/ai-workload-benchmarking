import onnxruntime as ort
import numpy as np
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import measure_latency

BATCH_SIZES = [1, 4, 16, 64]
THREADS = [1, 2, 4, 8]
RUNS = 100
WARMUP = 10
ONNX_PATH = Path("models/resnet18.onnx")
RESULTS_PATH = Path("results/benchmark_onnx.csv")

def benchmark_onnx(session, batch_size):
    input_tensor = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    fn = lambda: session.run(None, {'input': input_tensor})
    results = measure_latency(fn, runs=RUNS, warmup=WARMUP)
    
    throughput = (batch_size / results["mean_ms"]) * 1000
    
    return {
        "runtime": "onnx",
        "batch_size": batch_size,
        "threads": session.get_session_options().intra_op_num_threads,
        "mean_ms": results["mean_ms"],
        "std_ms": results["std_ms"],
        "min_ms": results["min_ms"],
        "max_ms": results["max_ms"],
        "throughput_img_s": throughput,
    }
def create_session(onnx_path, num_threads):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    return session
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
    print("Starting ONNX Runtime benchmark...")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Threads: {THREADS}")
    
    for num_threads in THREADS:
        session = create_session(ONNX_PATH, num_threads)
        for batch_size in BATCH_SIZES:
            print(f"  batch={batch_size}, threads={num_threads}...", end=" ")
            result = benchmark_onnx(session, batch_size)
            save_results(result, RESULTS_PATH)
            print(f"mean={result['mean_ms']:.1f}ms, throughput={result['throughput_img_s']:.1f}img/s")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()