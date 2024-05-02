import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value


class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def make_ring_network(self, N, neighbour_range=1):
        """
        Initializes a ring network with N nodes. Each node is connected to its immediate neighbours
        specified by the neighbour_range on both sides.

        Args:
        - N (int): Number of nodes in the network
        - neighbour_range (int): Number of adjacent neighbours each node is connected to on both sides
        """
        self.nodes = []  # Resetting nodes to an empty list for new network
        for i in range(N):
            connections = [0] * N  # Initialize connections for current node with no connections
            for j in range(1, neighbour_range + 1):
                right_neighbour = (i + j) % N  # Circular index for right neighbour
                left_neighbour = (i - j + N) % N  # Circular index for left neighbour using modulo
                connections[right_neighbour] = 1  # Connect to right neighbour
                connections[left_neighbour] = 1  # Connect to left neighbour
            self.nodes.append(Node(np.random.random(), i, connections))

    def make_small_world_network(self, N, re_wire_prob=0.2):
        """
        Transforms a ring network into a small-world network by randomly rewiring connections
        with a specified probability.

        Args:
        - N (int): Number of nodes in the network
        - re_wire_prob (float): Probability of rewiring each edge

        Returns:
        - int: Number of connections that were rewired
        """
        self.make_ring_network(N)  # Start by creating a ring network

        for i, node in enumerate(self.nodes):
            for j in range(len(node.connections)):
                if node.connections[j] == 1 and np.random.random() < re_wire_prob:
                    node.connections[j] = 0  # Disconnect the current node from its neighbour

                    # Generate a list of potential new connection targets excluding self and current connections
                    potential_new_connections = [
                        k for k in range(N) if k != i and node.connections[k] == 0
                    ]

                    if potential_new_connections:  # Ensure there are eligible nodes to connect to
                        new_connection = np.random.choice(potential_new_connections)
                        node.connections[new_connection] = 1  # Create a new connection
                        self.nodes[new_connection].connections[i] = 1  # Ensure the connection is bidirectional

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            """
            cm.hot is a colour map that starts of at black (for low values) and through to white (for high values)
            In between black and white is red (closer to black) and yellow (closer to white).
            This is useful for implementing the ising model on a network as we can visually see the difference 
            in colour between two nodes which we can understand as the agreement between two nodes
            """
            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


def calculateAgreementOnNetwork(node, nodes, external):  # Function calculates the change in agreement for a node within a network
    changeInAgreement = external * node.value  # Initialise agreement by multiplying current node value with the external factor
    for i, connected in enumerate(node.connections):  # Iterate over the neighbours of the current node
        if connected:  # Check that the current node is actually connected
            changeInAgreement += node.value * nodes[i].value  # Add to the agreement sum with the interaction between current node and its neighbour
    return changeInAgreement  # Return the sum agreement


def isingStepOnNetwork(network, alpha, external):  # Performs a single step of the model
    node = network.nodes[np.random.randint(len(network.nodes))]  # Randomly select a node within the network. This is an exclusive function which ensures that a non-existent node isn't selected
    agreement = calculateAgreementOnNetwork(node, network.nodes, external)  # Calculates agreement
    if agreement < 0 or np.random.rand() < np.exp(-2 * agreement / alpha):  # np.random.rand() generates a random float between 0 and 1. if the probability function is less than this value, a flip occurs. If agreement < 0 a flip occurs
        node.value *= -1


# This function will run the Ising model simulation on a network
def isingMainOnNetwork(network, alpha, external):  # Actually runs the model
    for frame in range(100):  # Loop runs for the specified number of frames [100 frames]

        for step in range(1000):  # Iterate single steps 1000 times to form an update
            isingStepOnNetwork(network, alpha, external)

        network.plot()  # Plots current network frame [plots every 1000 steps]
        plt.pause(0.5)  # Pauses the plot for .5s
        plt.close()  # Closes the current plot


def main():
    N = 20  # Number of nodes
    network = Network()  # Creates an instance of the Network class
    network.make_small_world_network(N)  # Creates small world network [obv can be changed but, I used this for testing]

    alpha = 10  # Set alpha value
    external = 0.1  # Set external value
    isingMainOnNetwork(network, alpha, external)


if __name__ == "__main__":
    main()
