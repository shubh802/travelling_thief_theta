import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from ttp_aco import Tsp
from ttp_multi_obj import TTMultiObjective
from ks_single_obj import KnapsackGA




def ks_multi_obj(route, distance_matrix, setup, color, index):
    """This function calls the multi objective class and is run for all the routes
    gathered from the ACO. This function also intitialises the variables for the
    multi objective class. Lastly, this function saves the data collected to text files."""

    #Initialising variables
    no_of_cities = len(route)
    max_items = 5
    min_weight = 1
    max_weight = 10
    min_value = 1
    max_value = 10
    pop_size = 20
    v_max = 10
    v_min = 1
    max_pickup = 2

    #Initialising variables according to test instance
    if no_of_cities >= 4461: pop_size = 10

    if setup == 0:
        no_of_items = np.array([1 for i in range(no_of_cities)])
        no_of_items[0] = 0
        total_items = sum(no_of_items)

        max_weight = 1000
        weight_list = np.random.randint(min_weight, max_weight, total_items)
        value_list = weight_list + 100
        max_pickup = 5

        file_name = 'theta_a' + str(no_of_cities) + '-n' + str(total_items)
    elif setup == 1:
        no_of_items = np.array([5 for i in range(no_of_cities)])
        no_of_items[0] = 0
        total_items = sum(no_of_items)

        max_weight = 1010
        min_weight = 1000
        max_value = 1000
        weight_list = np.random.randint(min_weight, max_weight, total_items)
        value_list = np.random.randint(min_value, max_value, total_items)

        file_name = 'theta_a' + str(no_of_cities) + '-n' + str(total_items)
    elif setup == 2:
        no_of_items = np.array([10 for i in range(no_of_cities)])
        no_of_items[0] = 0
        total_items = sum(no_of_items)

        max_weight = 1000
        max_value = 1000
        weight_list = np.random.randint(min_weight, max_weight, total_items)
        value_list = np.random.randint(min_value, max_value, total_items)

        file_name = 'theta_a' + str(no_of_cities) +'-n'+ str(total_items)


    print("Items per city:", no_of_items)
    print("total no. of items:", total_items)
    print("weights list:",weight_list)
    print("values list:", value_list)
    bag_max_weight = sum(weight_list)*(max_pickup-1)/2
    print("Max Bag Weight:", bag_max_weight)
    k = total_items//5

    mobj = TTMultiObjective(route,max_items,bag_max_weight, min_weight, max_weight,
                    min_value, max_value, pop_size, v_max, v_min, distance_matrix, setup)
                
    #Generating Population
    rows, cols = (pop_size, total_items)
    chromosomes = [[0 for i in range(cols)] for j in range(rows)]

    for i in range(rows):
        for j in range(cols):
            chromosomes[i][j] = np.random.randint(max_pickup)
        
        
    chromosomes = np.array(chromosomes)
    print("Initial chromosomes:", chromosomes)

    chromosomes, times, values = mobj.algorithm(1000, 5, 1, chromosomes, no_of_items, value_list, weight_list, color)

    print(f"Final chromo\n: {chromosomes} \n times: {times} \nvalues:{values}")

    #Saving the deliverables
    mobj.save_route_data(file_name+'.x', route, chromosomes)
    mobj.save_value_data(file_name+'.f', values,  times)




def main():
    #Picking up number of cities and test instance from runtime
    n_cities = int(sys.argv[1])
    setup = int(sys.argv[2])

    #Running Ant colony to get routes and distance matrix
    tsp = Tsp(n_cities, n_ants=5,sc=1000,tau_init=1, res_type=1, res_len= 5)
    total_route = tsp.output
    print(total_route)
    distance_matrix = tsp.distance

    print("Distance matrix:\n",distance_matrix)
    color=['red', 'blue','green','black','yellow']
    for idx, route in enumerate(total_route):
        #Calling the multi objective algorithm
        ks_multi_obj(route, distance_matrix, setup, color = color[idx], index = idx)

    #Saving the plot
    file_name = 'pareto-a' + str(n_cities) + '-' + str(setup) + '.png'

    red_patch = mpatches.Patch(color='red', label='Route 1')
    blue_patch = mpatches.Patch(color='blue', label='Route 2')
    green_patch = mpatches.Patch(color='green', label='Route 3')
    cyan_patch = mpatches.Patch(color='black', label='Route 4')
    purple_patch = mpatches.Patch(color='yellow', label='Route 5')
    plt.legend(handles=[red_patch,blue_patch,green_patch,cyan_patch,purple_patch])

    plt.savefig(file_name)
    plt.close()


if __name__ == "__main__":
    main()



