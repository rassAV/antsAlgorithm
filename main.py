import numpy as np
import random
from collections import defaultdict

def read_graph(file_name):
    """Читает граф из файла и нормализует индексы вершин."""
    edges = []
    nodes = set()

    with open(file_name, 'r') as file:
        for line in file:
            u, v, w = map(int, line.split())
            edges.append((u, v, w))
            nodes.update([u, v])

    n = len(nodes)
    graph = np.full((n, n), float('inf'))
    for u, v, w in edges:
        graph[u][v] = w

    return graph

def ant_colony_optimization(graph, n_ants, n_iterations, alpha, beta, evaporation_rate, q):
    """Муравьиный алгоритм."""
    n_nodes = graph.shape[0]
    pheromones = np.ones((n_nodes, n_nodes))
    best_path = None
    best_length = float('inf')

    for iteration in range(n_iterations):
        paths = []
        lengths = []

        for ant in range(n_ants):
            start_node = (iteration + ant) % n_nodes  # Начальная точка по циклу
            path = [start_node]
            visited = set(path)

            while len(path) < n_nodes:
                current_node = path[-1]
                probabilities = []
                for next_node in range(n_nodes):
                    if next_node not in visited:
                        pheromone = pheromones[current_node][next_node]
                        distance = graph[current_node][next_node]
                        if distance < float('inf'):  # Избегаем недоступных ребер
                            attractiveness = (pheromone ** alpha) * ((1 / distance) ** beta)
                            probabilities.append((next_node, attractiveness))

                if probabilities:
                    total = sum(p[1] for p in probabilities)
                    if total > 0:
                        probabilities = [(node, attractiveness / total) for node, attractiveness in probabilities]
                        next_node = random.choices(
                            [p[0] for p in probabilities],
                            weights=[p[1] for p in probabilities]
                        )[0]
                    else:
                        # Если вероятность равна 0, выбираем случайного доступного соседа
                        next_node = random.choice([p[0] for p in probabilities])
                else:
                    break  # Выход, если нет доступных узлов

                path.append(next_node)
                visited.add(next_node)

            # Добавляем возвращение в начальную точку
            if len(path) == n_nodes:
                path.append(path[0])
                length = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
                paths.append(path)
                lengths.append(length)

        # Выбираем лучший путь
        if lengths:
            min_length = min(lengths)
            if min_length < best_length:
                best_length = min_length
                best_path = paths[lengths.index(min_length)]

        # Обновляем феромоны
        pheromones *= (1 - evaporation_rate)
        for path, length in zip(paths, lengths):
            pheromone_deposit = q / length
            for i in range(len(path) - 1):
                pheromones[path[i]][path[i + 1]] += pheromone_deposit
                pheromones[path[i + 1]][path[i]] += pheromone_deposit

        # Выводим путь и его длину на каждой итерации
        print(f"Итерация {iteration + 1}:")
        print(f"Лучший путь на этой итерации: {best_path}")
        print(f"Длина пути: {best_length}")
        print("-" * 40)

    return best_path, best_length

if __name__ == "__main__":
    graph = read_graph("graph2.dat")
    n_ants = 600
    n_iterations = 10
    alpha = 2.0
    beta = 4.0
    evaporation_rate = 0.001
    q = 0.0001

    best_path, best_length = ant_colony_optimization(graph, n_ants, n_iterations, alpha, beta, evaporation_rate, q)
    print("Лучший путь:", best_path)
    print("Длина пути:", best_length)
