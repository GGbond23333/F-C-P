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

	def get_clustering(self):
		# Calculate the mean clustering coefficient for the network.
		mean_clustering = 0
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

	def get_path_length(self):
		total_path_lengths = 0  # Initialise variable which will store the sum of all path lengths
		total_paths = 0  # Initialise variable which will store the total number of paths

		for start_node in self.nodes:  # Initiate the breadth-first search which will loop over and use each node as the starting node
			queue = [(start_node, 0)]  # Initialise the queue for the breadth-first search with the start node (with a distance of 0 from itself)
			visited = {start_node}  # Keeps track of all visited nodes so that a search is not initiated from a given node twice
			distances = {start_node: 0}  # Initialise dictionary which will store distance from the current start node to other nodes

			while queue:  # Creates a loop which runs until queue is empty
				# Assigns "current_node" to the node that is first in the queue, assigns "distance" to distance from start node, pops the first item in the queue off
				current_node, distance = queue.pop(0)

				# Initiate a loop over the connections of the current node to explore adjacent nodes (second part of the breadth-first search)
				for i, is_connected in enumerate(current_node.connections):
					if is_connected:  # Insures node is actually connected
						neighbor = self.nodes[i]  # Get the neighbouring node
						if neighbor not in visited:  # Insures that neighbouring node has not already been visited
							queue.append((neighbor,
							              distance + 1))  # Adds neighbour and updated distance to the tail of the queue
							visited.add(neighbor)  # Mark neighbour as visited
							distances[neighbor] = distance + 1  # Add the distance to the dictionary
							total_path_lengths += distance + 1  # Add distance to sum of path lengths variable
							total_paths += 1  # Increment the total path count

		mean_path_length = total_path_lengths / total_paths if total_paths > 0 else 0  # Calculating the mean whilst avoiding zero error
		return round(mean_path_length, 15)  # Rounds to 15dp for accuracy

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
		fig, ax = plt.subplots()
		self.plotNetwork(ax)
		ax.set_axis_off()  # Clears the axis
		plt.show()

	def plotNetwork(self, ax):  # The plot method has been slightly modified so that it now takes ax as an input
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
			totalAgreement += population[row, col] * population[r, c]  # Calculates agreement between current cell and one neighbour then sums it to total_agreement
	changeInAgreement = totalAgreement + external * population[row, col]  # Calculates the change_in_agreement by summing the total_agreement with the external pull
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

grid = []
count = 0
size = 100
while count < size:
	x = round(np.random.random(), 3)
	grid.append(x)
	count += 1
iterations = 100000
iterate = 0
# defining some main variables and the grid needed for this task in which there is continuous float data between 0 and 1

large_dataset = []
large_dataset.append(grid)


def defuant_main_opinion(grid, threshold, beta):
	i = np.random.randint(0, 100)  # random variable to choose a random grid value
	x = np.random.randint(0, 2)  # variable to decide which neighbour is chosen
	if i != 99:
		if x == 0:
			abs_diff = abs(grid[i] - grid[i - 1])  # stores the absolute difference of the two values
			if abs_diff > threshold:  # checks for the difference being less than the given threshold, else it does nothing
				grid = grid
			if abs_diff <= threshold:
				if grid[i] > grid[i - 1]:
					grid[i] = round((grid[i] - (beta * (abs_diff))), 3)
					grid[i - 1] = round((grid[i - 1] + (beta * (abs_diff))), 3)
				if grid[i - 1] > grid[i]:
					grid[i] = round((grid[i] + (beta * (abs_diff))), 3)
					grid[i - 1] = round((grid[i - 1] - (beta * (abs_diff))), 3)
		if x == 1:
			abs_diff = abs(grid[i] - grid[i + 1])
			if abs_diff > threshold:
				grid = grid
			if abs_diff <= threshold:
				if grid[i] > grid[i + 1]:
					grid[i] = round((grid[i] - (beta * (abs_diff))), 3)
					grid[i + 1] = round((grid[i + 1] + (beta * (abs_diff))), 3)
				if grid[i + 1] > grid[i]:
					grid[i] = round((grid[i] + (beta * (abs_diff))), 3)
					grid[i + 1] = round((grid[i + 1] - (beta * (abs_diff))), 3)
				# Lines 22-44 will use the random variables to choose a random cell and one neighbour to change them
				# according to the equations defined in the task sheet.

	if i == 99:
		if x == 0:
			abs_diff = abs(grid[99] - grid[98])
			if abs_diff > threshold:
				grid = grid
			if abs_diff <= threshold:
				if grid[99] > grid[98]:
					grid[99] = round((grid[99] - (beta * (abs_diff))), 3)
					grid[98] = round((grid[98] + (beta * (abs_diff))), 3)
				if grid[98] > grid[99]:
					grid[99] = round((grid[99] + (beta * (abs_diff))), 3)
					grid[98] = round((grid[98] - (beta * (abs_diff))), 3)
		if x == 1:
			abs_diff = abs(grid[99] - grid[0])
			if abs_diff > threshold:
				grid = grid
			if abs_diff <= threshold:
				if grid[99] > grid[0]:
					grid[99] = round((grid[99] - (beta * abs_diff)), 3)
					grid[0] = round((grid[0] + (beta * abs_diff)), 3)
				if grid[0] > grid[i]:
					grid[99] = round((grid[99] + (beta * abs_diff)), 3)
					grid[0] = round((grid[0] - (beta * abs_diff)), 3)
				# Lines 48-70 will do the same task however for the special case of i = 99 since i+1 should be zero
				# but this cannot be defined in the previous if statement so I created a separate one.
	return grid


def defuant_main(grid, threshold, beta):
	iterate = 0
	fig, axs = plt.subplots(
		2)  # creates two subplots to show the change in opinion, (I had issues with trying to show the change in opinion over time so I decided to use this as an alternative)
	plt.title("Change in opinion over time using two histograms.", loc='center')
	axs[0].hist(grid)
	while iterate < iterations:  # this will run the opinion function for a fixed number of iterations defined at the start
		grid = defuant_main_opinion(grid, threshold, beta)
		large_dataset.append(grid)
		iterate += 1
	axs[1].hist(large_dataset[0])
	plt.xlim(0, 1)
	plt.xlabel("Opinion")
	plt.ylabel("Frequency density")
	plt.show()
	plt.savefig('change.png')  # shows the plots and saves the figure


def test_defuant(grid, threshold,
                 beta):  # this function tests certain aspects of the continuous data to ensure the programme works as expected
	for i in range(
			99):  # and also that the data fulfils numerical requirements to begin with e.g all the numbers are between 0 and 1
		assert 0 <= grid[i] <= 1
		assert abs(grid[i] - grid[i - 1]) < 1
		assert abs(grid[i] - grid[i + 1]) < 1
	assert abs(grid[99] - grid[98]) < 1
	assert abs(grid[99] - grid[0]) < 1
	print("Tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model Network implementation - Task 5
==============================================================================================================
'''


def calculateAgreementOnNetwork(node, nodes,
                                external):  # Function calculates the change in agreement for a node within a network
	changeInAgreement = external * node.value  # Initialise agreement by multiplying current node value with the external factor
	for i, connected in enumerate(node.connections):  # Iterate over the neighbours of the current node
		if connected:  # Check that the current node is actually connected
			changeInAgreement += node.value * nodes[
				i].value  # Add to the agreement sum with the interaction between current node and its neighbour
	return changeInAgreement  # Return the sum agreement


def isingStepOnNetwork(network, alpha, external):  # Performs a single step of the model
	node = network.nodes[np.random.randint(
		len(network.nodes))]  # Randomly select a node within the network. This is an exclusive function which ensures that a non-existent node isn't selected
	agreement = calculateAgreementOnNetwork(node, network.nodes, external)  # Calculates change in agreement
	if agreement < 0 or np.random.rand() < np.exp(
			-2 * agreement / alpha):  # np.random.rand() generates a random float between 0 and 1. if the probability function is less than this value, a flip occurs. If agreement < 0 a flip occurs
		node.value *= -1


def isingMainOnNetwork(network, alpha, external):  # Actually runs the model

	ax = plt.figure().add_subplot()  # Create figure and add a default subplot to it

	for frame in range(100):  # Loop runs for the specified number of frames [100 frames]
		for step in range(1000):  # Iterate single steps 1000 times to form an update
			isingStepOnNetwork(network, alpha, external)

		ax.clear()  # Clears current content off of the plot - this results in faster execution as previous plots aren't stored
		ax.set_axis_off()  # Clears the axis
		network.plotNetwork(ax)  # Plots current network frame [plots every 1000 steps]
		plt.pause(0.1)  # Pauses the plot for .1s without closing it

	plt.show()  # Keep final frame open


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
	parser = argparse.ArgumentParser(
		description="Create and plot different types of networks.")  # Create arg parser object

	# Network Arguments
	parser.add_argument('-network', type=int, nargs='?', const=10, help='Create a random network')
	parser.add_argument('-ring_network', type=int, nargs='?', const=10, help='Create a ring network')
	parser.add_argument('-small_world', type=int, nargs='?', const=10, help='Create a small world network')
	parser.add_argument('-re_wire_prob', type=float, default=0.2, help='Rewire probability for small world network')
	parser.add_argument('-connection_probability', type=float, default=0.2,
	                    help='Connection probability for random network')
	parser.add_argument('-test_network', action='store_true', help='Run predefined network tests')
	# Ising Model Arguments
	parser.add_argument("-ising_model", action="store_true", help='Run Ising Model with default parameters')
	parser.add_argument("-test_ising", action="store_true", help='Run predefined Ising Model tests')
	parser.add_argument("-external", type=float, default=0.0, help='Set a value for the external factor')
	parser.add_argument("-alpha", type=float, default=1.0, help='Set a value for alpha')
	# Ising Model on a Network
	parser.add_argument("-use_network", type=int,
	                    help='Run Ising Model on a small world network with N nodes')  # IDK if it is required to make it run on all networks
	parser.add_argument('-defuant', action='store_true', help="Simulate the Defuant model.")
	parser.add_argument('-beta', type=float, default=0.2, help='Set the coupling parameter.')
	parser.add_argument('-threshold', type=float, default=0.2, help='Set the opinion threshold.')
	parser.add_argument('-test_defuant', action='store_true', help='Run the testing function.')
	args = parser.parse_args()

	# Processing test flags
	if args.test_network:
		test_networks()
		return
	elif args.test_ising:
		test_ising()
		return

	# Processing Ising Model Network flag
	if args.use_network:
		network = Network()
		network.make_small_world_network(args.use_network, args.connection_probability)
		isingMainOnNetwork(network, args.alpha, args.external)
		return

	# Processing defuant model
	if args.defuant:
		defuant_main(grid, args.threshold, args.beta)
	elif args.test_defuant:
		test_defuant(grid, args.threshold, args.beta)

	# Processing all other flags
	network = None
	if args.network is not None:
		network = Network()
		network.make_random_network(args.network, args.connection_probability)
	elif args.ring_network is not None:
		network = Network()
		network.make_ring_network(args.ring_network)
	elif args.small_world is not None:
		network = Network()
		network.make_small_world_network(args.small_world, args.re_wire_prob)
	elif args.ising_model:
		initialPopulation = np.random.choice([-1, 1], size=(100, 100))
		ising_main(initialPopulation, alpha=args.alpha, external=args.external)
		return

	# If a network was created, plot it and output its metrics
	if network:
		network.plot()
		print(f"Mean degree: {network.get_mean_degree()}")
		print(f"Average path length: {network.get_path_length()}")
		print(f"Clustering coefficient: {network.get_clustering()}")
	else:
		print("No valid arguments provided.")  # If invalid or no arguments then this is printed


if __name__ == "__main__":
	main()
