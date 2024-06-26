import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
		global mean_clustering
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
		global mean_path_length
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
		num_rewired = 0  # Initialize counter for the number of rewiring

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
						num_rewired += 1  # Increment the rewired counter

		return num_rewired  # Return the total number of rewiring

	def plot(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		buffer = 0.1 * network_radius if network_radius != 0 else 1  # 添加一个条件，确保不为零

		# 设置x轴的左右限制，并确保两者不相同，通过添加一个小的缓冲区。
		ax.set_xlim([-1.1 * network_radius - buffer, 1.1 * network_radius + buffer])
		# 同样的策略用于y轴
		ax.set_ylim([-1.1 * network_radius - buffer, 1.1 * network_radius + buffer])

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
	assert (network.get_clustering() == 0), network.get_clustering()
	assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

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
	assert (network.get_clustering() == 0), network.get_clustering()
	assert (network.get_path_length() == 5), network.get_path_length()

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
	assert (network.get_clustering() == 1), network.get_clustering()
	assert (network.get_path_length() == 1), network.get_path_length()

	print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


# This function calculates the change_in_agreement
# Inputs: population, current row, current column, "external"
def calculate_agreement(population, row, col, external):
	# Finding the number of rows and columns from the population matrix
	numberRows, numberCols = population.shape

	# Finds the coordinates of the neighbours of a given position in the matrix
	neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

	# Sets initial total_agreement to 0 for a given cell
	totalAgreement = 0

	# Nested loop which calculates change_in_agreement
	for r, c in neighbors:
		if 0 <= r < numberRows and 0 <= c < numberCols:  # Ensures that neighbour is within bounds of grid (this is for positions on the edge of the grid)
			totalAgreement += population[row, col] * population[
				r, c]  # Calculates agreement between current cell and one neighbour then sums it to total_agreement
	changeInAgreement = totalAgreement + external * population[
		row, col]  # Calculates the change_in_agreement by summing the total_agreement with the external pull
	return changeInAgreement


# This function performs a single update of the model
def ising_step(population, alpha, external):
	numberRows, numberCols = population.shape
	row = np.random.randint(0, numberRows)
	col = np.random.randint(0, numberCols)

	agreement = calculate_agreement(population, row, col, external)

	# The code below results in random flips. if agreement < 0 a flip occurs
	# np.random.rand() generates a random float between 0 and 1. if the probability function is less than this value, a flip occurs
	if agreement < 0 or np.random.rand() < np.exp(-agreement / alpha):
		population[row, col] *= -1


# This function displays a plot of the model
def plot_ising(im, population):
	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)


# This function tests the calculate_agreement function
def test_ising():
	print("Testing calculate_agreement calculations\n")

	population = -np.ones((3, 3))
	assert (calculate_agreement(population, 1, 1, 0) == 4), "Test 1"
	print("Test 1 Complete")
	population[1, 1] = 1.
	assert (calculate_agreement(population, 1, 1, 0) == -4), "Test 2"
	print("Test 2 Complete")
	population[0, 1] = 1.
	assert (calculate_agreement(population, 1, 1, 0) == -2), "Test 3"
	print("Test 3 Complete")
	population[1, 0] = 1.
	assert (calculate_agreement(population, 1, 1, 0) == 0), "Test 4"
	print("Test 4 Complete")
	population[2, 1] = 1.
	assert (calculate_agreement(population, 1, 1, 0) == 2), "Test 5"
	print("Test 5 Complete")
	population[1, 2] = 1.
	assert (calculate_agreement(population, 1, 1, 0) == 4), "Test 6"
	print("Test 6 Complete\n")

	print("Testing external pull\n")
	population = -np.ones((3, 3))
	assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
	print("Test 7 Complete")
	assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
	print("Test 8 Complete")
	assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 9"
	print("Test 9 Complete")
	assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 10"
	print("Test 10 Complete\n")

	print("Tests passed")


def ising_main(population, alpha, external):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, alpha, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def defuant_main():
	# Your code for task 2 goes here
	pass


def test_defuant():
	# Your code for task 2 goes here
	pass


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
	# Create an arg parser object
	parser = argparse.ArgumentParser(description="Create and plot different types of networks.")

	# Network Arguments
	parser.add_argument('--type', type=str, choices=['ring', 'small_world', 'random'], default='random',
						help='Type of network to create')
	parser.add_argument('--nodes', type=int, default=10, help='Number of nodes in the network')
	parser.add_argument('--connection_probability', type=float, default=0.5,
						help='Connection probability for random network')
	parser.add_argument('--neighbour_range', type=int, default=1,
						help='Neighbour range for ring network')
	parser.add_argument('--re_wire_prob', type=float, default=0.2,
						help='Rewire probability for small world network')
	parser.add_argument('--test_network', action='store_true', help='Run predefined network tests')
	# Ising Model Arguments
	parser.add_argument("-ising_model", action="store_true")
	parser.add_argument("-test_ising", action="store_true")
	parser.add_argument("-external", type=float, default=0.0)
	parser.add_argument("-alpha", type=float, default=1.0)

	# Parse arguments
	args = parser.parse_args()

	# Create the network object
	network = Network()

	# Determine which type of network to create based on the type argument
	if args.type == 'ring':
		network.make_ring_network(args.nodes, args.neighbour_range)
	elif args.type == 'small_world':
		network.make_small_world_network(args.nodes, args.re_wire_prob)
	elif args.type == 'random':
		network.make_random_network(args.nodes, args.connection_probability)

	# Plot the created network
	network.plot()

	# Calculate and print metrics if the network flag is provided
	if args.test_network:
		test_networks()
	else:
		# Assuming you always want to print metrics unless it's a test run
		print(f"Mean degree: {network.get_mean_degree()}")
		print(f"Average path length: {network.get_mean_path_length()}")
		print(f"Clustering coefficient: {network.get_mean_clustering()}")

	if args.test_ising:
		test_ising()  # Runs the testing function
	elif args.ising_model:
		initialPopulation = np.random.choice([-1, 1], size=(100, 100))  # Randomly generates a 100x100 grid containing 1s and -1s (e.g. agree/disagree ...)
		ising_main(initialPopulation, alpha=args.alpha, external=args.external)  # Runs the main program with the grid input (initial_population), alpha and external
	else:
		print("Invalid arguments provided.")  # If command line inputs aren't float values, this will be printed


if __name__ == "__main__":
	main()

# networkT = Network()
# networkT.make_small_world_network(10, 0.1)
# networkT.plot()
# print(networkT.make_small_world_network(10, 1))
