import argparse
import random
import matplotlib.pyplot as plt
import numpy as np


class SmallWorldnetwork:
    def __init__(self, n, k, p):
        self.n = n  # number of nodes
        self.k = k  # Range of distances over which nodes are connected
        self.p = p  # Rewiring probability
        self.collection_nodes = self.generate_node_ring()
        self.rewire_small_world()

    def generate_node_ring(self):
        collection_nodes = np.zeros((self.n, self.n), dtype=int)
        for node in range(self.n):
            for distance in range(1, self.k + 1):
                collection_nodes[node][(node + distance) % self.n] = 1
                collection_nodes[node][(node - distance) % self.n] = 1
        return collection_nodes

    def rewire_small_world(self):
        num_rewired = 0
        for i in range(self.n):
            for j in range(1, self.k + 1):
                if random.random() < self.p:
                    old_j = (i + j) % self.n
                    potential_new_edges = [n for n in range(self.n) if self.collection_nodes[i][n] == 0 and n != i]
                    if potential_new_edges:
                        new_j = random.choice(potential_new_edges)
                        self.collection_nodes[i][old_j] = 0
                        self.collection_nodes[old_j][i] = 0
                        self.collection_nodes[i][new_j] = 1
                        self.collection_nodes[new_j][i] = 1
                        num_rewired += 1
        return self.collection_nodes, num_rewired

    def plot_network(self):
        fig, ax = plt.subplots()
        x = np.cos(np.linspace(0, 2 * np.pi, self.n, endpoint=False))
        y = np.sin(np.linspace(0, 2 * np.pi, self.n, endpoint=False))
        ax.scatter(x, y, s=100)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.collection_nodes[i][j] == 1:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.6)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate different types of networks")
    parser.add_argument('-small_world_nodes', type=int, default=10,
                        help="Generate a small-world network with a specified number of nodes (default 10)")
    parser.add_argument('-re_wire', type=float, default=0.2,
                        help="Rewiring probability for the small-world network (default to using a re-wiring "
                             "probability of p=0.2,unless the flag -re-wire <probability> is included)")
    parser.add_argument('-connection_range', type=int, default=1,
                        help="Range of distances over which nodes are connected (default 1)")

    args = parser.parse_args()

    if args.small_world_nodes is not None:
        network_test = SmallWorldnetwork(n=args.small_world_nodes, k=args.connection_range, p=args.re_wire)
        network_test.plot_network()


if __name__ == "__main__":
    main()
