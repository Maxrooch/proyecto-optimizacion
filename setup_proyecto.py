import os

files_content = {
    "requirements.txt": "numpy\nmatplotlib\nseaborn\nnetworkx\nscipy\n",
    "src/__init__.py": "",
    "src/ej1_parallel_speedup.py": '''import time
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
''',
    "src/ej2_greedy_knapsack.py": '''import matplotlib.pyplot as plt
import numpy as np

def greedy_knapsack(weights, values, capacity, criterion='ratio'):
    n = len(weights)
    if criterion == 'ratio':
        indices = sorted(range(n), key=lambda i: values[i]/weights[i], reverse=True)
    elif criterion == 'min_weight':
        indices = sorted(range(n), key=lambda i: weights[i])
    selected, total_weight, total_value = [], 0, 0
    for i in indices:
        if total_weight + weights[i] <= capacity:
            selected.append(i)
            total_weight += weights[i]
            total_value += values[i]
    return selected, total_weight, total_value

def run_ejercicio2():
    weights, values, capacity = [2, 3, 10], [3, 4, 20], 10
    sel_ratio, w_ratio, v_ratio = greedy_knapsack(weights, values, capacity, 'ratio')
    sel_minw, w_minw, v_minw = greedy_knapsack(weights, values, capacity, 'min_weight')
    print("\\n--- Ejercicio 2: Greedy Knapsack ---")
    print(f"Ratio  -> Objetos: {sel_ratio}, Valor: {v_ratio}")
    print(f"Menor Peso -> Objetos: {sel_minw}, Valor: {v_minw}")

    methods = ['Valor / Peso (Ratio)', 'Menor Peso First']
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(2) - 0.175, [v_ratio, v_minw], 0.35, label='Valor Total', color='#2ca02c')
    plt.bar(np.arange(2) + 0.175, [w_ratio, w_minw], 0.35, label='Capacidad Usada', color='#ff7f0e')
    plt.axhline(y=capacity, color='r', linestyle='--', label='Capacidad Máxima')
    plt.xticks(np.arange(2), methods)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/ej2_knapsack_comparison.png', dpi=300)
    plt.close()
''',
    "src/ej3_tsp_2opt.py": '''import numpy as np
import matplotlib.pyplot as plt

def compute_tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def nearest_neighbor_tour(dist_matrix, start_node=0):
    n = dist_matrix.shape[0]
    unvisited = set(range(n)) - {start_node}
    tour = [start_node]
    curr = start_node
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_matrix[curr, x])
        unvisited.remove(nxt)
        tour.append(nxt)
        curr = nxt
    return tour

def two_opt(tour, dist_matrix):
    n = len(tour)
    best_tour, best_len = tour.copy(), compute_tour_length(tour, dist_matrix)
    improvements, improved = 0, True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1: continue
                new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                new_len = compute_tour_length(new_tour, dist_matrix)
                if new_len < best_len:
                    best_tour, best_len, improvements, improved = new_tour, new_len, improvements + 1, True
                    break
            if improved: break
    return best_tour, best_len, improvements

def run_ejercicio3(n_cities=30, runs=30):
    np.random.seed(42)
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    rand_imp, rand_costs, nn_imp, nn_costs = [], [], [], []
    print("\\n--- Ejercicio 3: TSP 2-opt ---")
    for _ in range(runs):
        _, cost_r, imp_r = two_opt(list(np.random.permutation(n_cities)), dist_matrix)
        rand_imp.append(imp_r); rand_costs.append(cost_r)
        _, cost_nn, imp_nn = two_opt(nearest_neighbor_tour(dist_matrix, np.random.randint(0, n_cities)), dist_matrix)
        nn_imp.append(imp_nn); nn_costs.append(cost_nn)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([rand_imp, nn_imp], labels=['Aleatorio', 'Vecino Más Cercano'])
    axes[0].set_title('Número de Mejoras')
    axes[1].boxplot([rand_costs, nn_costs], labels=['Aleatorio', 'Vecino Más Cercano'])
    axes[1].set_title('Calidad del Óptimo')
    plt.tight_layout()
    plt.savefig('figures/ej3_2opt_improvements.png', dpi=300)
    plt.close()
    return dist_matrix, coords
''',
    "src/ej4_simulated_annealing.py": '''import time, math
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
    print("\\n--- Ejercicio 4: Simulated Annealing ---")
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
''',
    "src/ej5_mst_approximation.py": '''import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.ej3_tsp_2opt import compute_tour_length, two_opt
from src.ej4_simulated_annealing import simulated_annealing

def solve_tsp_exact(dist_matrix):
    return min(compute_tour_length(p, dist_matrix) for p in itertools.permutations(range(dist_matrix.shape[0])))

def mst_approximation(dist_matrix):
    n = dist_matrix.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    mst = nx.minimum_spanning_tree(G)
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(mst.edges())
    eulerian = list(nx.eulerian_circuit(multigraph, source=0))
    visited, tour = set(), []
    for u, v in eulerian:
        if u not in visited:
            tour.append(u); visited.add(u)
    return tour

def run_ejercicio5(num_instances=100, n_cities=9):
    print("\\n--- Ejercicio 5: MST Approximation ---")
    approx_ratios, costs_mst, costs_mst_2opt, costs_2opt_rand, costs_sa = [], [], [], [], []
    for _ in range(num_instances):
        coords = np.random.rand(n_cities, 2) * 100
        dist_matrix = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
        exact_cost = solve_tsp_exact(dist_matrix)

        tour_mst = mst_approximation(dist_matrix)
        cost_mst = compute_tour_length(tour_mst, dist_matrix)
        approx_ratios.append(cost_mst / exact_cost)
        costs_mst.append(cost_mst)

        _, cost_m_2opt, _ = two_opt(tour_mst, dist_matrix)
        costs_mst_2opt.append(cost_m_2opt)

        _, cost_r_2opt, _ = two_opt(list(np.random.permutation(n_cities)), dist_matrix)
        costs_2opt_rand.append(cost_r_2opt)

        _, cost_sa_val, _, _ = simulated_annealing(dist_matrix, schedule='exp', max_iter=2000)
        costs_sa.append(cost_sa_val)

    plt.figure(figsize=(8, 5))
    plt.hist(approx_ratios, bins=15, color='#2ca02c', edgecolor='black', alpha=0.7)
    plt.axvline(x=2.0, color='r', linestyle='--', linewidth=2, label='Límite Teórico (2.0)')
    plt.legend(); plt.tight_layout()
    plt.savefig('figures/ej5_mst_approx_ratio.png', dpi=300); plt.close()

    plt.figure(figsize=(9, 5))
    plt.boxplot([costs_mst, costs_mst_2opt, costs_2opt_rand, costs_sa], labels=['MST Puro', 'MST + 2-opt', '2-opt (Rand)', 'Sim. Annealing'])
    plt.tight_layout()
    plt.savefig('figures/ej5_method_comparison.png', dpi=300); plt.close()
''',
    "main.py": '''import os
from src.ej1_parallel_speedup import run_ejercicio1
from src.ej2_greedy_knapsack import run_ejercicio2
from src.ej3_tsp_2opt import run_ejercicio3
from src.ej4_simulated_annealing import run_ejercicio4
from src.ej5_mst_approximation import run_ejercicio5

if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    print("=== INICIANDO EXPERIMENTOS Y GENERACIÓN DE GRÁFICAS ===")
    run_ejercicio1()
    run_ejercicio2()
    dist_matrix, _ = run_ejercicio3()
    run_ejercicio4(dist_matrix)
    run_ejercicio5()
    print("\\n=== EXPERIMENTOS COMPLETADOS EXITOSAMENTE. Gráficas guardadas en 'figures/' ===")
''',
    "README.md": "# Práctica de Optimización\n\nEjecutar `python main.py` para generar experimentos.\n"
}

if __name__ == "__main__":
    print("Creando estructura del proyecto...")
    os.makedirs("src", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    for path, content in files_content.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [+] Creado: {path}")
    print("\n¡Estructura generada exitosamente!")