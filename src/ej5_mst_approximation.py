import itertools
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
    print("\n--- Ejercicio 5: MST Approximation ---")
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
