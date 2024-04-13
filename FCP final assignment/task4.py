import numpy as np
import random
import matplotlib.pyplot as plt


class SmallWorldNetwork:
    def __init__(self, N, k, p):
        self.N = N  # Number of nodes
        self.k = k  # Each node is joined with k nearest neighbors
        self.p = p  # Rewiring probability
        self.adj_matrix = self.generate_ring_lattice()

    def generate_ring_lattice(self):
        adj_matrix = np.zeros((self.N, self.N), dtype=int)
        for node in range(self.N):
            for neighbor in range(1, self.k // 2 + 1):
                adj_matrix[node][(node + neighbor) % self.N] = 1
                adj_matrix[node][(node - neighbor) % self.N] = 1
        return adj_matrix

    def rewire_small_world(self):
        for i in range(self.N):
            for j in range(i + 1, i + self.k // 2 + 1):
                if random.random() < self.p:
                    potential_new_edges = [n for n in range(self.N) if self.adj_matrix[i][n] == 0 and n != i]
                    if potential_new_edges:
                        new_j = random.choice(potential_new_edges)
                        self.adj_matrix[i][j % self.N] = 0
                        self.adj_matrix[j % self.N][i] = 0
                        self.adj_matrix[i][new_j] = 1
                        self.adj_matrix[new_j][i] = 1
        return self.adj_matrix

    def plot_network(self):
        fig, ax = plt.subplots()
        x = np.cos(np.linspace(0, 2 * np.pi, self.N, endpoint=False))
        y = np.sin(np.linspace(0, 2 * np.pi, self.N, endpoint=False))
        ax.scatter(x, y, s=100)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.adj_matrix[i][j] == 1:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.6)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


# Example usage
if __name__ == "__main__":
    N = 10
    k = 2
    p = 0.3
    network = SmallWorldNetwork(N, k, p)
    original_matrix = np.copy(network.adj_matrix)
    network.rewire_small_world()
    network.plot_network()

