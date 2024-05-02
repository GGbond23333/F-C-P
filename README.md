
FCP Assignment 


Overview:
The script assignment.py defines and manages networks and simulates two different models: the Ising model and the Defuant model. The Ising model can also be run on a small world network. The script can create three different types of networks: random, ring and small world. The script also computes the following network metrics for any created network: mean degree, average path length and clustering coefficient. Any created network is also plotted. The script also allows for testing and validation of the models through predefined test functions.

Networks within the script:
Random Network - A randomly generated network where each node is connected to each other node with a specified probability
Ring Network - A network where each node is connected to it's immediate neighbours
Small World Network - A ring network where each node has a specified probability of randomly rewiring connection

Models within the script:
Ising Model - The Ising model simulates a populations opinion on a topic, with +1 representing FOR and -1 representing AGAINST. The key assumption is that individuals want to hold similar opinions to those around them, and as such an individual is more likely to change their opinion if everyone around them disagrees.
Defuant Model - The Defuant model simulates opinions on a continuous scale between 0 and 1. Similarly to the Ising model, individuals wish to hold similar opinions to those around them however a difference within this model is that a constraint is introduced where individuals will only consider opinions within a certain distance of their own.

Usage:
The script is designed to be run from the command line with various flags to control its operation. Here are some of the key commands:

	Network Creation:
	-network [N]: Creates a random network with N nodes. [By default, N=10]
	-ring_network [N]: Creates a ring network with N nodes. [By default, N=10]
	-small_world [N]: Creates a small world network with N nodes. [By default, N=10]
		-re_wire_prob [p]: An additional argument to the small worlds network flag which can be used to specify rewire probability

	Model Simulation:
	-ising_model: Runs the Ising model simulation.
		-external [x]: An additional argument to the Ising model flag which can be used to specify a value for the external factor [By default, external=0]
		-alpha [x]: An additional argument to the Ising model flag which can be used to specify a value for the alpha parameter [By default, alpha=1]
		-use_network [N]: An additional argument to the Ising model flag which can be used to run the Ising model on a small world network with N nodes. [By default, N=100] 
	-defuant: Runs the Defuant model simulation
		-beta [x]: An additional argument to the Defuant model flag which can be used to specify the value for the beta parameter [By default, beta=0.2]
		-threshold [x]: An additional argument to the Defuant model flag which can be used to specify the opinion threshold parameter [By default, threshold=0.2]

	Testing:
	-test_network: Runs tests on predefined network configurations
	-test_ising: Tests the Ising Model's calculate agreement function
	-test_defuant: Tests the Defuant models opinion dynamics

Example Commands:
python3 assignment.py -small_world 100 -re_wire_prob 0.1 -ising_model -alpha 0.5 -external 1.0
python3 assignment.py -ising_model -use_network
Python3 assignment.py -defuant -beta 0.1 -threshold 0.3


Required Libraries:
numpy, matplotlib, argparse
