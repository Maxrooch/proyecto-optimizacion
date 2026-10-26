import os
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
    print("\n=== EXPERIMENTOS COMPLETADOS EXITOSAMENTE. Gráficas guardadas en 'figures/' ===")
