Ovrview
The script defines and manages networks, simulating three different models: the Ising Model, Deffuant Model, and Ising Model on Networks. It includes a system for creating and interacting with networks (e.g., random, ring, and small world networks), computing network metrics (e.g., mean degree, clustering coefficient), and plotting network structures. The script also allows testing and validating these models through predefined test functions.

Model Simulations:
Ising Model: Simulates magnetic alignment using Ising Model principles.
Deffuant Model: Simulates opinion formation based on the proximity of initial opinions.
Ising Model on Networks: Applies Ising Model principles to the network structure.
Plotting: Networks are visualized with matplotlib, showing node connections and their properties.
Testing: Functions to test various network structures and model behaviors ensure correctness.
Main Function: Handles command-line arguments to run specific models or tests, allowing users to specify network types, model parameters, and test executions.

Usage
The script is designed to be run from the command line with various flags to control its operation. Here are some of the key commands:

Network Creation:
-network [N]: Creates a random network with N nodes.
-ring_network [N]: Creates a ring network with N nodes.
-small_world [N]: Creates a small world network with N nodes and a specified rewire probability (-re_wire_prob).

Model Simulation:
-ising_model: Runs the Ising Model simulation.
-defuant: Runs the Deffuant Model simulation with specified threshold (-threshold) and beta (-beta).

Testing:
-test_network: Runs tests on predefined network configurations.
-test_ising: Tests the Ising Model's calculation functions.
-test_defuant: Tests the Deffuant Model's opinion dynamics.

Example Command
python FCP_assignment_v6.py -small_world 100 -re_wire_prob 0.1 -ising_model -alpha 0.5 -external 1.0
This command sets up a small world network of 100 nodes with a rewire probability of 0.1, then runs the Ising Model on this network with specific model parameters.

Important Notes
Ensure you have the necessary Python packages installed (numpy, matplotlib) to run the script.
The script is flexible, allowing various combinations of flags and parameters to tailor the simulations and tests according to user needs.
This guide should help you understand and effectively utilize the provided script for network simulations and modeling.
