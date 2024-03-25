import numpy as np
import matplotlib.pyplot as plt


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
        if 0 <= r < numberRows and 0 <= c < numberCols:     # Ensures that neighbour is within bounds of grid (this is for positions on the edge of the grid)
            totalAgreement += population[row, col] * population[r, c]  # Calculates agreement between current cell and one neighbour then sums it to total_agreement
    changeInAgreement = totalAgreement + external * population[row, col]  # Calculates the change_in_agreement by summing the total_agreement with the external pull
    return changeInAgreement


# This function performs a single update of the model
def ising_step(population, alpha, external=0.0):
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


def ising_main(population, alpha, external=0.0):

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


# This function tests the calculate_agreement function
def test_ising():

    print("Testing calculate_agreement calculations")

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
    print("Test 6 Complete")

    print("Testing external pull")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    print("Test 7 Complete")
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    print("Test 8 Complete")
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 9"
    print("Test 9 Complete")
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 10"
    print("Test 10 Complete")

    print("Tests passed")


# Running the code
# test_ising()

initial_population = np.random.choice([-1, 1], size=(100, 100))  # Randomly generates a 100x100 grid containing 1s and -1s 
ising_main(initial_population, alpha=0.01, external=0)  # Runs the main function with a given alpha and external value 
