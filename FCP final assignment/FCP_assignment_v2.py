import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse


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

	def get_mean_degree(self):
		# Calculate the mean degree by summing up all node degrees and dividing by the number of nodes.
		total_degrees = sum(sum(i.connections) for i in self.nodes)
		mean_degrees = total_degrees / len(self.nodes) if self.nodes else 0
		return mean_degrees

	def get_mean_clustering(self):
		# Calculate the mean clustering coefficient for the network.
		clustering_coefficients = []

		for node in self.nodes:
			# Calculate the number of connections for the current node.
			neighbours_number = sum(node.connections)
			# Calculate the total possible connections among the neighbours.
			possible_connections = (neighbours_number * (neighbours_number - 1)) / 2

			if possible_connections == 0:
				# If no possible connections, the clustering coefficient is 0.
				clustering_coefficients.append(0)
				continue

			actual_connections = 0
			# Iterate through each pair of neighbours to count actual connections.
			for i, is_connected in enumerate(node.connections):
				if is_connected:
					for j in range(i + 1, len(node.connections)):
						if node.connections[j] and self.nodes[i].connections[j]:
							actual_connections += 1

			# Calculate the clustering coefficient for this node.
			coefficient = actual_connections / possible_connections
			clustering_coefficients.append(coefficient)
			mean_clustering = sum(clustering_coefficients) / len(
				clustering_coefficients) if clustering_coefficients else 0

		return mean_clustering

	def get_mean_path_length(self):
		# Calculate the mean path length across all nodes using a breadth-first search (BFS) approach.
		total_path_lengths = 0
		total_paths = 0

		for start_node in self.nodes:
			# Start BFS from each node.
			queue = [(start_node, 0)]
			visited = {start_node: True}
			distances = {start_node: 0}

			while queue:
				current_node, distance = queue.pop(0)

				for i, is_connected in enumerate(current_node.connections):
					if is_connected:
						neighbor = self.nodes[i]
						if neighbor not in visited:
							queue.append((neighbor, distance + 1))
							visited[neighbor] = True
							distances[neighbor] = distance + 1

			total_path_lengths += sum(distances.values())
			total_paths += len(distances)
			mean_path_length = total_path_lengths / total_paths if total_paths > 0 else 0
		return mean_path_length

	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):

		collection_nodes = np.zeros((N, N), dtype=int)
		for node in range(N):
			for distance in range(1, neighbour_range + 1):
				collection_nodes[node][(node + distance) % N] = 1
				collection_nodes[node][(node - distance) % N] = 1

		return collection_nodes

	def make_small_world_network(self, N, re_wire_prob=0.2):
		collection_nodes = self.make_ring_network(N)
		num_rewired = 0
		for i in range(N):
			for j in range(1, re_wire_prob):
				if random.random() < re_wire_prob:
					old_j = (i + j) % N
					potential_new_edges = [n for n in range(N) if collection_nodes[i][n] == 0 and n != i]
					if potential_new_edges:
						new_j = random.choice(potential_new_edges)
						collection_nodes[i][old_j] = 0
						collection_nodes[old_j][i] = 0
						collection_nodes[i][new_j] = 1
						collection_nodes[new_j][i] = 1
						num_rewired += 1
		return collection_nodes, num_rewired

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
		plt.show()
def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	#Your code for task 1 goes here
	return np.random.random() * population
	pass
def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1
	#Your code for task 1 goes here
	pass

def plot_ising(im, population):

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)
    pass
def test_ising():

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==14), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==-6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
	#Your code for task 2 goes here
	pass
def test_defuant():
	#Your code for task 2 goes here
	pass

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-network', type=int, help='Create and plot a random network of the specified size')
	parser.add_argument('-ring_network', type=int, help='Create and plot a ring network of the specified size')
	parser.add_argument('-small_world', type=int, help='Create and plot a small world network of the specified size')
	parser.add_argument('-re_wire', type=float, default=0.2, help='Re-wiring probability for small world network')
	parser.add_argument('-test_networks', action='store_true', help='Run the test functions')
	args = parser.parse_args()

	if args.network:
		N = args.network
		connection_probability = 0.5
		network = Network()
		network.make_random_network(N, connection_probability)
		print(f"Mean degree: {network.get_mean_degree()}")
		print(f"Average path length: {network.get_mean_path_length()}")
		print(f"Clustering co-efficient: {network.get_mean_clustering()}")
		network.plot()
		plt.show()
	if args.ring_network:
		N = args.ring_network
		network = Network()
		network.make_ring_network(N)
		print(f"Mean degree: {network.get_mean_degree()}")
		print(f"Average path length: {network.get_mean_path_length()}")
		print(f"Clustering co-efficient: {network.get_mean_clustering()}")
		network.plot()
		plt.show()

	if args.small_world:
		N = args.small_world
		rewire_probability = args.re_wire
		network = Network()
		network.make_small_world_network(N, re_wire_prob=rewire_probability)
		print(f"Mean degree: {network.get_mean_degree()}")
		print(f"Average path length: {network.get_mean_path_length()}")
		print(f"Clustering co-efficient: {network.get_mean_clustering()}")
		network.plot()
		plt.show()

	if args.test_networks:
		test_networks()


if __name__=="__main__":
	main()

networkT = Network()
networkT.make_random_network(10, 0.5)
networkT.plot()
