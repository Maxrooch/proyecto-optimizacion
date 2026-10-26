import time, math
import numpy as np
import matplotlib.pyplot as plt
from src.ej3_tsp_2opt import compute_tour_length

def simulated_annealing(dist_matrix, schedule='exp', T0=1000.0, Tf=0.01, max_iter=10000):
    n = dist_matrix.shape[0]
    curr_tour = list(np.random.permutation(n))
    curr_cost = compute_tour_length(curr_tour, dist_matrix)
    best_tour, best_cost = curr_tour.copy(), curr_cost
    history, T = [], T0
    start_time = time.perf_counter()
    for k in range(max_iter):
        if T <= Tf: break
        T = T0 * (0.995 ** k) if schedule == 'exp' else T0 - (T0 - Tf) * (k / max_iter)
        i, j = sorted(np.random.choice(n, 2, replace=False))
        if i == 0 and j == n - 1: continue
        neighbor = curr_tour[:i] + curr_tour[i:j][::-1] + curr_tour[j:]
        neighbor_cost = compute_tour_length(neighbor, dist_matrix)
        delta = neighbor_cost - curr_cost
        if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-10)):
            curr_tour, curr_cost = neighbor, neighbor_cost
            if curr_cost < best_cost:
                best_tour, best_cost = curr_tour.copy(), curr_cost
        history.append(curr_cost)
    return best_tour, best_cost, history, time.perf_counter() - start_time

def run_ejercicio4(dist_matrix):
    print("\n--- Ejercicio 4: Simulated Annealing ---")
    results = {}
    plt.figure(figsize=(9, 5))
    for sched in ['linear', 'exp']:
        _, cost, history, t_exec = simulated_annealing(dist_matrix, schedule=sched)
        results[sched] = (cost, t_exec)
        plt.plot(history, label=f'{sched.capitalize()} (Costo: {cost:.1f})')
        print(f"Esquema {sched:<8} | Costo: {cost:.2f} | Tiempo: {t_exec:.4f}s")
    plt.legend(); plt.tight_layout()
    plt.savefig('figures/ej4_sa_convergence.png', dpi=300); plt.close()

    plt.figure(figsize=(7, 5))
    for sched, (cost, t_exec) in results.items():
        plt.scatter(t_exec, cost, s=120, label=f'{sched.capitalize()}')
    plt.legend(); plt.tight_layout()
    plt.savefig('figures/ej4_sa_time_vs_quality.png', dpi=300); plt.close()
