import numpy as np
import random
import matplotlib.pyplot as plt


# Generate a ring lattice graph
def generate_ring_lattice(N, k):
    adj_matrix = np.zeros((N, N), dtype=int)  # Initialize adjacency matrix with zeros
    for node in range(N):
        for neighbor in range(1, k//2+1):
            adj_matrix[node][(node + neighbor) % N] = 1  # Connect to neighbor clockwise
            adj_matrix[node][(node - neighbor) % N] = 1  # Connect to neighbor counter-clockwise
    return adj_matrix


# Rewire the graph to introduce small-world properties
def rewire_small_world(adj_matrix, p, k):
    N = len(adj_matrix)
    for i in range(N):
        for j in range(i+1, i+k//2+1):
            if random.random() < p:  # With probability p, rewire an edge
                potential_new_edges = [n for n in range(N) if adj_matrix[i][n] == 0 and n != i]
                if potential_new_edges:
                    new_j = random.choice(potential_new_edges)
                    adj_matrix[i][j % N] = 0  # Remove existing edge
                    adj_matrix[j % N][i] = 0
                    adj_matrix[i][new_j] = 1  # Add new edge
                    adj_matrix[new_j][i] = 1
    return adj_matrix

# Plot the network using matplotlib


def plot_network(adj_matrix):
    fig, ax = plt.subplots()
    N = len(adj_matrix)
    x = np.cos(np.linspace(0, 2*np.pi, N, endpoint=False))  # X-coordinates
    y = np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))  # Y-coordinates
    ax.scatter(x, y, s=100)  # Plot nodes
    for i in range(N):
        for j in range(i+1, N):
            if adj_matrix[i][j] == 1:  # If there's an edge, plot it
                ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.6)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# Test functions to ensure the ring lattice and small world properties are correct
def test_ring_lattice_properties(adj_matrix, N, k):
    for i in range(N):
        assert np.sum(adj_matrix[i]) == k, f"Node {i} does not have {k} connections."


def test_small_world_properties(orig_adj_matrix, rewired_adj_matrix, N):
    original_edges = np.sum(orig_adj_matrix) / 2
    rewired_edges = np.sum(rewired_adj_matrix) / 2
    assert original_edges == rewired_edges, "The number of edges changed after rewiring."
    assert np.array_equal(rewired_adj_matrix, rewired_adj_matrix.T), "Adjacency matrix is not symmetric after rewiring."


# Generate the ring lattice and small world network
N = 10
k = 2
p = 0.3
ring_lattice = generate_ring_lattice(N, k)
small_world = rewire_small_world(np.copy(ring_lattice), p, k)

# Perform tests
test_ring_lattice_properties(ring_lattice, N, k)
test_small_world_properties(ring_lattice, small_world, N)

# Plot the final network
plot_network(small_world)
