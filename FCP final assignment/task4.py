import numpy as np
import random
import matplotlib.pyplot as plt

def generate_ring_lattice(N, k):
    adj_matrix = np.zeros((N, N), dtype=int)
    for node in range(N):
        for neighbor in range(1, k//2+1):
            adj_matrix[node][(node + neighbor) % N] = 1
            adj_matrix[node][(node - neighbor) % N] = 1
    return adj_matrix

def rewire_small_world(adj_matrix, p):
    N = len(adj_matrix)
    for i in range(N):
        for j in range(i+1, i+k//2+1):
            if random.random() < p:
                potential_new_edges = [n for n in range(N) if adj_matrix[i][n] == 0 and n != i]
                if potential_new_edges:
                    new_j = random.choice(potential_new_edges)
                    adj_matrix[i][j % N] = 0
                    adj_matrix[j % N][i] = 0
                    adj_matrix[i][new_j] = 1
                    adj_matrix[new_j][i] = 1
    return adj_matrix

def plot_network(adj_matrix):
    fig, ax = plt.subplots()
    N = len(adj_matrix)
    x = np.cos(np.linspace(0, 2*np.pi, N, endpoint=False))
    y = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    ax.scatter(x, y, s=100)
    for i in range(N):
        for j in range(i+1, N):
            if adj_matrix[i][j] == 1:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.6)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

N = 10
k = 2
p = 0.1

ring_lattice = generate_ring_lattice(N, k)
small_world = rewire_small_world(ring_lattice, p)
plot_network(small_world)
