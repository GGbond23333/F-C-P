import numpy as np
import matplotlib.pyplot as plt
import argparse
#importing the relevant packages for this task

grid = []
count = 0
size = 100
while count < size:
    x = round(np.random.random(),3)
    grid.append(x)
    count+=1
iterations = 100000
iterate = 0
#defining some main variables and the grid needed for this task in which there is continuous float data between 0 and 1

large_dataset = []
large_dataset.append(grid)
def defuant_main_opinion(grid,threshold,beta):
    i = np.random.randint(0, 100) #random variable to choose a random grid value
    x = np.random.randint(0, 2) #variable to decide which neighbour is chosen
    if i != 99:
        if x == 0:
            abs_diff = abs(grid[i] - grid[i - 1]) #stores the absolute difference of the two values
            if abs_diff > threshold: #checks for the difference being less than the given threshold, else it does nothing
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
                    #Lines 22-44 will use the random variables to choose a random cell and one neighbour to change them
                    #according to the equations defined in the task sheet.

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
                    #Lines 48-70 will do the same task however for the special case of i = 99 since i+1 should be zero
                    #but this cannot be defined in the previous if statement so I created a separate one.
    return grid




def defuant_main(grid,threshold,beta):
    iterate = 0
    fig, axs = plt.subplots(2)#creates two subplots to show the change in opinion, (I had issues with trying to show the change in opinion over time so I decided to use this as an alternative)
    plt.title("Change in opinion over time using two histograms.",loc = 'center')
    axs[0].hist(grid)
    while iterate < iterations:   #this will run the opinion function for a fixed number of iterations defined at the start
        grid = defuant_main_opinion(grid, threshold, beta)
        large_dataset.append(grid)
        iterate += 1
    axs[1].hist(large_dataset[0])
    plt.xlim(0, 1)
    plt.xlabel("Opinion")
    plt.ylabel("Frequency density")
    plt.show()
    plt.savefig('change.png')     #shows the plots and saves the figure

def test_defuant(grid,threshold,beta):     #this function tests certain aspects of the continuous data to ensure the programme works as expected
    for i in range(99):                    #and also that the data fulfils numerical requirements to begin with e.g all the numbers are between 0 and 1 
        assert 0 <= grid[i] <= 1
        assert abs(grid[i]-grid[i-1])<1
        assert abs(grid[i] - grid[i + 1]) < 1
    assert abs(grid[99]-grid[98])<1
    assert abs(grid[99]-grid[0])<1
    

def main(): #this function ties everything together and uses arguement parsers to change the coupling and threshold parameters and choose which function to run
    parser = argparse.ArgumentParser(description = "Simulate Defuant model")
    parser.add_argument('-defuant', action ='store_true', help = "Simulate the Defuant model.")
    parser.add_argument('-beta',type=float,default=0.2,help='Set the coupling parameter.')
    parser.add_argument('-threshold',type=float,default=0.2, help='Set the opinion threshold.')
    parser.add_argument('-test_defuant',action='store_true',help='Run the testing function.')
    args = parser.parse_args()
    if args.defuant:
        defuant_main(grid,args.threshold,args.beta)
    elif args.test_defuant:
        test_defuant(grid,args.threshold,args.beta)
if __name__ == "__main__" :
    main()
