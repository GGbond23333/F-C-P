import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


class Queue:
    """
    A class representing a queue that uses the principle of First-In, First-Out (FIFO)
    """

    def __init__(self):
        """
        Initializes a Queue object.
        """

        # creates an empty list to represent the queue
        self.queue = []

    def push(self, item):
        """
        Add an element to the queue

        Inputs:
        item: the element to be added
        """

        # add the element to the queue list
        self.queue.append(item)

    def pop(self):
        """
        Remove the first element from the list of queue

        Returns:
            a queue object, the first element from the queue list
            If the element in the queue list is less than one, return no value
        """

        # If there is no element in the list, return no value
        if len(self.queue) < 1:
            return None

        # return the first element in the list that is removed
        return self.queue.pop(0)

    def is_empty(self):
        """
        Determine the queue is empty or not
        by comparing the number of elements in the list with Zero

        Returns:
            Boolean value of the expression which compares
            the no. of elements in the queue list
        """

        # Determine whether the queue object is empty
        return len(self.queue) == 0


class Node:
    """
    A class representing a node

    Attributes:
        value: the node value
        number: the node index
        connections: a list or numpy array representing the row of the adjacency matrix that corresponds to the node
    """

    def __init__(self, value, number, connections=None):
        """
        Initializes a Node object.

        value: the node value
        number: the node index
        connections: a list or numpy array representing the row of the adjacency matrix that corresponds to the node
        """

        self.index = number
        self.connections = connections
        self.value = value
        self.parent = None

    def get_neighbours(self):
        """
        translate from the connections list into
        a list of indexes that the node is connected to
        """

        # create a list of indexes that the node is connected to
        return np.where(np.array(self.connections) == 1)[0]


class Network:
    """
    A class representing the network
    Attributes:

    """

    def __init__(self, nodes=None):
        """
        Initializes the Network object
        Args:
            nodes: the list of the node objects
        """

        # Check if the list of the nodes is provided when creating the network

        # if the list is not given, create an empty list that can accept the node elements
        if nodes is None:
            self.nodes = []
        # if the node is provided, store
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        """
        Calculate the average of the degree of all nodes in the network

        Returns:
            mean_degree(float): the mean degree of the network

        """

        # create a list that stores the degree of each node
        nodes_deg = []
        # loop over the nodes
        for node in self.nodes:
            # create a list that represent the number of edges that a node has
            node_edges = [edge for edge, connection in enumerate(node.connections) if connection == 1]
            # count the number of edges (degree of a node) and add it to the nodes_deg list
            node_deg = len(node_edges)
            nodes_deg.append(node_deg)

        # calculate the mean degree of all nodes in the network
        total_deg = sum(nodes_deg)
        num_deg = len(nodes_deg)
        mean_degree = total_deg / num_deg

        return mean_degree

    def clustering_coefficient(self, node):
        """
        Calculate the coefficient of a node

        Args:
            node: a node object in the network

        Returns:
            clustering_coeff(float) : the s the fraction of a node's neighbours that connect to each other,
            forming a triangle that includes the original node
        """

        # Create a list of the neighbours of a node
        node_neighbours = [self.nodes[conn_index] for conn_index in node.get_neighbours()]
        num_neighbours = len(node_neighbours)

        # Initialize the counter of number of triangle connection in a network with zero
        num_triangles = 0

        # Make a condition that the number of neighbours is greater than 1
        # because there can be a triangle connection only if there is at least one neighbour
        if num_neighbours > 1:
            # i --> index of a neighbour of a particular node
            for i in range(num_neighbours):
                # Loop over to get index of the rest of neighbours
                for j in range(i+1, num_neighbours):

                    # Obtain a neighbour node by accessing the node_neighbours list with index i
                    node_neighbour = node_neighbours[i]

                    # Obtain the index of the rest of neighbours
                    idx_rest_neighbour = node_neighbours[j].index

                    # check whether a node's neighbour is connected to the rest of that node's neighbours
                    if node_neighbour.connections[idx_rest_neighbour]:
                        num_triangles += 1

            # calculate the number of possible connections between a node's neighbours
            possible_triangles = num_neighbours * (num_neighbours - 1) / 2

            # calculate the clustering coefficient of a node
            clustering_coeff = num_triangles / possible_triangles

            return clustering_coeff

        # If the number of neighbour is less than 1, return 0.0, as the clustering coefficient
        else:
            return 0.0

    def get_mean_clustering(self):
        """
        find the mean clustering coefficient for the network by averaging
        the clustering coefficient of a node over for all nodes in the network.

        Returns:
            mean_clustering_coeff (float)
        """

        # Initialize the total clustering coefficient with zero
        total_clustering_coefficient = 0
        num_nodes = len(self.nodes)

        # loop over the nodes to add the clustering coefficient of each node in the list
        for node in self.nodes:
            total_clustering_coefficient += self.clustering_coefficient(node)

        # calculate the mean clustering coefficient for the network
        mean_clustering_coeff = total_clustering_coefficient / num_nodes

        return mean_clustering_coeff

    def get_start_end_nodes(self):
        """
        Get start_node list and end_node list
        by considering each node in the nodes of network as a start node
        and the rest of nodes as end_nodes

        Returns:
            starts: a list that contains all the nodes in the network, that are considered as start_nodes
            ends: a nested list that contains the end_nodes list of each node in the network
        """

        starts = []
        ends = []

        # loop over the index and value of all the nodes in the network
        for index, start_node in enumerate(self.nodes):
            # store one of the nodes as a start node
            starts.append(start_node)
            # store the rest of nodes as end_nodes
            end_nodes = [end_node for idx, end_node in enumerate(self.nodes) if idx != index]
            # store end_nodes for a particular node in the end_node list
            ends.append(end_nodes)
        return starts, ends

    def find_path_length(self):
        """
        Find path lengths from a node to rest of the nodes using breadth-first-search

        Returns:
            path_lengths: the nested list of the number of the path lengths
            from each node in a network to the rest of the nodes
        """

        # retrieve start_node and end-node lists using get_start_node methods
        starts, ends = self.get_start_end_nodes()

        path_lengths = []

        # Loop over start and end_node lists
        for start_node, end_nodes in zip(starts, ends):

            path_len = []

            for end_node in end_nodes:

                # create search_queue object
                search_queue = Queue()
                # Add a start_node to the queue list
                search_queue.push(start_node)

                # create an empty list to store the index of the node that has been visited
                visited = []

                while not search_queue.is_empty():

                    # Pop the next node from the queue
                    node_to_check = search_queue.pop()
                    # If the current node that is checked is the end_node, then finish
                    if node_to_check == end_node:
                        break
                    # If not, add all the neighbours of the current node to the search queue
                    # Loop through all the neighbours
                    for neighbour_index in node_to_check.get_neighbours():
                        # Get a neighbour node based on the index
                        neighbour = self.nodes[neighbour_index]
                        # Determine whether we have visited the neighbour
                        if neighbour_index not in visited:
                            # priority = calculate_priority(start_node.index, neighbour_index)
                            # search_queue.push(priority, neighbour)

                            # if not, add it to the search queue and store in visited
                            search_queue.push(neighbour)
                            visited.append(neighbour_index)

                            # Set the parent property to allow for backtracking
                            neighbour.parent = node_to_check

                # Backtrace to get the path after having found the end_node
                # Start at the end_node
                node_to_check = end_node
                # Make sure that start-node has no parent
                start_node.parent = None

                paths = []
                # Loop over node parents until we reach the start_node
                while node_to_check.parent:
                    # Add node to paths
                    paths.append(node_to_check)
                    # Update node to the parent of the current node
                    node_to_check = node_to_check.parent

                # Add the start node to the path
                paths.append(node_to_check)

                # Reverse the paths
                paths.reverse()

                # Count the number of path length between two nodes and store in path_len
                no_path_length = len(paths) - 1

                path_len.append(no_path_length)

            # store the path_length from each node to the rest of the nodes in path_lengths
            path_lengths.append(path_len)

        return path_lengths

    def get_mean_path_length(self):
        """
        Calculate the mean path length by averaging
        the average path length of a node over all nodes in the network

        Returns:
            mean_path_length(float)
        """

        # Get the nested list of path_length from each node in network to rest of nodes
        nodes_path_lengths = self.find_path_length()

        average_paths = []
        # loop through to get the node_path list that store path_length between a node to other node
        for node_path in nodes_path_lengths:
            # sum all the path_length between two nodes
            total_node_paths = sum(node_path)
            # calculate the average path length for a particular node
            average_path = total_node_paths/len(node_path)
            average_paths.append(average_path)

        # Calculate the mean path_length for all nodes in the network
        mean_path_length = sum(average_paths)/len(self.nodes)

        return round(mean_path_length, 15)

    def make_random_network(self, N, connection_probability):
        """
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        """

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

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

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.show()


def test_networks():

    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 0), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_mean_clustering() == 1), network.get_mean_clustering()
    assert (network.get_mean_path_length() == 1), network.get_mean_path_length()

    print("All tests passed")


def main():
    """
    Combine the codes written above to produce the desired results
    Set the flags required to run program

    Returns:

    """
    # Create an argparser object
    parser = argparse.ArgumentParser(description="Metrics to compare networks")
    # Add arguments '-network', '-test_network' to the parser
    parser.add_argument('-network', type=int, help='input the size required for the network')
    parser.add_argument('-test_network', action='store_true')

    # Ask argparser to do the parsing
    args = parser.parse_args()

    # Determine which action to produce
    if args.network:
        # the size of a network is the number provided from the terminal using -network flag
        size = args.network
        # Create the network object
        network = Network()
        # build a random network
        network.make_random_network(size, 0.5)
        # plot the random network
        network .plot()

        # Printing mean degree, average path length and clustering coefficient for the network to the terminal
        mean_deg = network.get_mean_degree()
        mean_path_length = network.get_mean_path_length()
        clustering_coef = network.get_mean_clustering()
        print(f"Mean degree: {mean_deg}")
        print(f"Average path length: {mean_path_length}")
        print(f"Clustering co-efficient: {clustering_coef}")
    elif args.test_network:
        test_networks()


if __name__ == "__main__":
    main()
