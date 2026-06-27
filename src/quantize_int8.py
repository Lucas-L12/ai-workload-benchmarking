import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import measure_latency

ONNX_PATH = Path("models/resnet18.onnx")
INT8_PATH = Path("models/resnet18_int8.onnx")
BATCH_SIZES = [1, 4, 16, 64]
THREADS = [1, 2, 4, 8]
RUNS = 100
WARMUP = 10
RESULTS_PATH = Path("results/benchmark_int8.csv")


def quantize_model(onnx_path, int8_path):
    print("Quantizing model to INT8...")
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8
    )
    
    fp32_size = onnx_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    
    print(f"FP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {int8_size:.2f} MB")
    print(f"Size reduction: {fp32_size/int8_size:.1f}x")
    
    return int8_path


def verify_accuracy(fp32_path, int8_path):
    print("\nVerifying accuracy difference...")
    
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    fp32_session = ort.InferenceSession(fp32_path)
    fp32_output = fp32_session.run(None, {'input': test_input})[0]
    
    int8_session = ort.InferenceSession(int8_path)
    int8_output = int8_session.run(None, {'input': test_input})[0]
    
    max_diff = np.max(np.abs(fp32_output - int8_output))
    mean_diff = np.mean(np.abs(fp32_output - int8_output))
    
    fp32_top1 = np.argmax(fp32_output)
    int8_top1 = np.argmax(int8_output)
    
    print(f"Max difference: {max_diff:.4f}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"FP32 top-1 class: {fp32_top1}, INT8 top-1 class: {int8_top1}")
    print(f"Top-1 match: {fp32_top1 == int8_top1}")


def benchmark_int8(int8_path):
    print("\nBenchmarking INT8 model...")
    
    for num_threads in THREADS:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        
        session = ort.InferenceSession(
            int8_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        for batch_size in BATCH_SIZES:
            input_tensor = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            
            fn = lambda: session.run(None, {'input': input_tensor})
            results = measure_latency(fn, runs=RUNS, warmup=WARMUP)
            
            throughput = (batch_size / results["mean_ms"]) * 1000
            
            result = {
                "runtime": "onnx_int8",
                "batch_size": batch_size,
                "threads": num_threads,
                "mean_ms": results["mean_ms"],
                "std_ms": results["std_ms"],
                "min_ms": results["min_ms"],
                "max_ms": results["max_ms"],
                "throughput_img_s": throughput,
            }
            
            save_results(result, RESULTS_PATH)
            print(f"  batch={batch_size}, threads={num_threads}... mean={result['mean_ms']:.1f}ms, throughput={result['throughput_img_s']:.1f}img/s")


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
    quantize_model(ONNX_PATH, INT8_PATH)
    verify_accuracy(ONNX_PATH, INT8_PATH)
    benchmark_int8(INT8_PATH)
    
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()