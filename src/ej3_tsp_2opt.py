import numpy as np
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
    print("\n--- Ejercicio 3: TSP 2-opt ---")
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
