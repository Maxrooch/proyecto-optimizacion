import matplotlib.pyplot as plt
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
    print("\n--- Ejercicio 2: Greedy Knapsack ---")
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
