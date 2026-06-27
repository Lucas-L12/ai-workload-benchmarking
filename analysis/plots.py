import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

THREADS = [1, 2, 4, 8]
BATCH_SIZES = [1, 4, 16, 64]


def plot_throughput_vs_threads(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    runtimes = {
        'pytorch': 'PyTorch FP32',
        'onnx': 'ONNX FP32',
        'onnx_int8': 'ONNX INT8'
    }
    
    colors = {
        'pytorch': 'blue',
        'onnx': 'green',
        'onnx_int8': 'red'
    }
    
    for runtime, label in runtimes.items():
        data = df[(df['runtime'] == runtime) & (df['batch_size'] == 1)]
        data = data.sort_values('threads')
        ax.plot(data['threads'], data['throughput_img_s'],
                marker='o', label=label, color=colors[runtime], linewidth=2)
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Throughput (img/s)')
    ax.set_title('Throughput vs Threads (batch=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(THREADS)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'throughput_vs_threads.png', dpi=150)
    plt.close()
    print("Saved: throughput_vs_threads.png")


def plot_throughput_vs_batch(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    runtimes = {
        'pytorch': 'PyTorch FP32',
        'onnx': 'ONNX FP32',
        'onnx_int8': 'ONNX INT8'
    }
    
    colors = {
        'pytorch': 'blue',
        'onnx': 'green',
        'onnx_int8': 'red'
    }
    
    for runtime, label in runtimes.items():
        data = df[(df['runtime'] == runtime) & (df['threads'] == 4)]
        data = data.sort_values('batch_size')
        ax.plot(data['batch_size'], data['throughput_img_s'],
                marker='o', label=label, color=colors[runtime], linewidth=2)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (img/s)')
    ax.set_title('Throughput vs Batch Size (threads=4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BATCH_SIZES)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'throughput_vs_batch.png', dpi=150)
    plt.close()
    print("Saved: throughput_vs_batch.png")


def plot_latency_vs_batch(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    runtimes = {
        'pytorch': 'PyTorch FP32',
        'onnx': 'ONNX FP32',
        'onnx_int8': 'ONNX INT8'
    }
    
    colors = {
        'pytorch': 'blue',
        'onnx': 'green',
        'onnx_int8': 'red'
    }
    
    for runtime, label in runtimes.items():
        data = df[(df['runtime'] == runtime) & (df['threads'] == 4)]
        data = data.sort_values('batch_size')
        ax.plot(data['batch_size'], data['mean_ms'],
                marker='o', label=label, color=colors[runtime], linewidth=2)
        ax.fill_between(data['batch_size'],
                        data['mean_ms'] - data['std_ms'],
                        data['mean_ms'] + data['std_ms'],
                        alpha=0.15, color=colors[runtime])
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs Batch Size (threads=4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BATCH_SIZES)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_vs_batch.png', dpi=150)
    plt.close()
    print("Saved: latency_vs_batch.png")


def plot_model_sizes():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['ResNet-18 FP32', 'ResNet-18 INT8']
    sizes = [44.58, 11.20]
    colors = ['green', 'red']
    
    bars = ax.bar(models, sizes, color=colors, alpha=0.7, width=0.4)
    
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{size} MB', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size: FP32 vs INT8')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_sizes.png', dpi=150)
    plt.close()
    print("Saved: model_sizes.png")


def main():
    print("Loading results...")
    
    pytorch_df = pd.read_csv(RESULTS_DIR / "benchmark_pytorch.csv")
    onnx_df = pd.read_csv(RESULTS_DIR / "benchmark_onnx.csv")
    int8_df = pd.read_csv(RESULTS_DIR / "benchmark_int8.csv")
    
    df = pd.concat([pytorch_df, onnx_df, int8_df], ignore_index=True)
    
    print(f"Total configurations: {len(df)}")
    print(df[['runtime', 'batch_size', 'threads', 'mean_ms', 'throughput_img_s']].to_string())
    
    print("\nGenerating plots...")
    plot_throughput_vs_threads(df)
    plot_throughput_vs_batch(df)
    plot_latency_vs_batch(df)
    plot_model_sizes()
    
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()