import time
import random
import multiprocessing as mp
import matplotlib.pyplot as plt

def monte_carlo_pi_chunk(samples):
    count = 0
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            count += 1
    return count

def estimate_pi(total_samples, num_workers):
    samples_per_worker = total_samples // num_workers
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(monte_carlo_pi_chunk, [samples_per_worker] * num_workers)
    total_inside = sum(results)
    return 4 * total_inside / total_samples

def run_ejercicio1(total_samples=50_000_000):
    workers_list = [1, 2, 4, 8]
    times = []
    print("--- Ejercicio 1: Parallel Speedup ---")
    for w in workers_list:
        start = time.perf_counter()
        pi_est = estimate_pi(total_samples, w)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Procesos: {w} | Tiempo: {elapsed:.4f}s | Pi aprox: {pi_est:.5f}")

    base_time = times[0]
    speedups = [base_time / t for t in times]
    plt.figure(figsize=(8, 5))
    plt.plot(workers_list, speedups, 'o-', label='Speedup Medido', color='#1f77b4', linewidth=2)
    plt.plot(workers_list, workers_list, 'k--', label='Speedup Ideal', linewidth=1.5)
    plt.xlabel('Número de Procesos')
    plt.ylabel('Aceleración (Speedup)')
    plt.title('Aceleración Paralela en la Estimación de Pi')
    plt.xticks(workers_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/ej1_speedup.png', dpi=300)
    plt.close()
