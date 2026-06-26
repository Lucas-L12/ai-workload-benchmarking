
import time
import numpy as np

def measure_latency(fn, runs=100, warmup=10):
    """
    Mide la latencia de una función en milisegundos.
    
    Args:
        fn: función a medir (sin argumentos)
        runs: número de mediciones reales
        warmup: ejecuciones previas que se descartan
    
    Returns:
        dict con mean, std, min, max en ms
    """
    # Warmup: las primeras ejecuciones son más lentas
    # (caches frías, inicializaciones lazy). Las descartamos.
    for _ in range(warmup):
        fn()
    
    # Medición real
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convertimos a ms
    
    times = np.array(times)
    
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }

