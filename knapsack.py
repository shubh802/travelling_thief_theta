# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Knapsack via EA file.
"""
import numpy as np
import pandas as pd
from numpy.random import choice
import matplotlib.pyplot as plt


class KnapsackGA:
    
    def __init__(self, cities, max_item, v_min,bag_max_weight, dist_mat):
        self.n_cities = cities
        self.max_item = max_item
        self.v_min = v_min
        self.bag_max_weight = bag_max_weight
        self.dist_mat = dist_mat
    
    
    def ks_population(self, route, wts, values):
        
        np.random.seed(5)
        total_pop,total_wt,total_val = [],[],[]
        # No of item in each city 
        # City should have min 1 item
        item_per_city = np.random.randint(1, self.max_item, size= self.n_cities)
        pop_len = item_per_city.sum()
        # Calculating weight for each item in the route
        # Weight of the item cannot be 0
        wt = np.random.randint(1,wts, size= pop_len)
        # Calculating value for each item in the route
        # Value of the item cannot be 0
        value = [np.random.randint(1,values)*10 for i in range(pop_len)]
        
        for i in range(route):
            pop = np.random.randint(2, size = pop_len)
            total_pop.append(pop)
            total_wt.append(wt)
            total_val.append(value)
               
        return item_per_city, total_pop, total_wt, total_val
    
    def ks_tournament_selection(self, pop, scores):
        rnd_var1 = np.random.randint(len(pop))
        rnd_var2 = np.random.randint(len(pop))
        # print(f"rnd_var1: {rnd_var1}, rnd_var2: {rnd_var2}")
        # Selecting parent based on the fitness
        if(scores[rnd_var1] > scores[rnd_var2]): index = rnd_var1
        elif(scores[rnd_var1] == scores[rnd_var2]): index = choice([rnd_var1, rnd_var2])
        else: index = rnd_var2
        return pop[index], index
    
    def ks_crossover(self, prnt1, prnt2, perform):
        child1, child2 = prnt1.copy(), prnt2.copy()
        if perform == 1:
            # Selecting random point from the length of the parent
            pt = np.random.randint(1, len(prnt1)-1)
            # print(f"Index for crossover: {pt}")
            # crossover performed from the point selected between the two parents
            child1 = np.concatenate([prnt1[:pt], prnt2[pt:]])
            child2 = np.concatenate([prnt2[:pt], prnt1[pt:]])
        return child1, child2
    
    def ks_mutate(self, chromo, k, perform_mutation):
        # Mutating items picked up by the thief based on k
        if perform_mutation == 1:
            for i in range(k):
                n = np.random.randint(len(chromo))
                # print(f"ks_mutate chromo value: {chromo[n]} at index {n}")
                if chromo[n] == 1: chromo[n] = 0
                else: chromo[n] = 1
            
        return chromo;
    
    def ks_weakest_replacement(self, pop, scores, mutation, route, items_per_city, value, wt):

        mut_val = self.ks_chromo_value(value, mutation) 
        mut_time = self.ks_time_fitness(mutation, route, items_per_city, wt)
        mut_val_time_fit = round(mut_val / mut_time,2)
        print(f"Mutation val-time score: {mut_val_time_fit}")
        min_fit = min(scores)
        if mut_val_time_fit > min_fit:
            min_fit_idx_lst = []
            for i in range(len(scores)):
                if scores[i] == min_fit : min_fit_idx_lst.append(i)
            idx = np.random.choice(min_fit_idx_lst)
            print(f"Chromosome to be replaced: {pop[idx]}, Route index: {idx}")
            pop[idx] = mutation
            scores[idx] = mut_val_time_fit
        return pop, scores
    
    def ks_plot(self, value, time, colormap, marker):
        
        for i in range(len(value)):
            plt.scatter(time[i], value[i],color=colormap[i], alpha=.50, zorder=2, marker=marker[i])
        plt.xlabel("Time")
        plt.ylabel("Total values per chromosome")

        

    
        #Calculating the weight of the bag at each city
    def ks_bag_weight(self, chromo, no_of_items,weight_list, city_index):
        items_considered = sum(no_of_items[:city_index])
        weights_considered = weight_list[:items_considered]
        chromosome_considered = chromo[:items_considered]  
        weight = np.dot(weights_considered, chromosome_considered)
        return weight
    
    
    def ks_chromo_value(self, value, chromo):
        return np.dot(value, chromo)

    
        #calculating time taken to travel between cities
    def ks_intercity_time(self, chromo, distance_matrix, route, items_per_city, chromo_weight, city_index):
        city1, city2 = route[city_index-1], route[city_index]
        # print(f"city1 {city1}:city2: {city2}")
        # Reducing 1 from city to match distance matrix (0,1,2,3) 
        distance = distance_matrix[city1-1][city2-1]
        weight = self.ks_bag_weight(chromo,items_per_city,chromo_weight, city_index)
        # print("Incremental bag weight: ",weight)      
        if weight > self.bag_max_weight:
            velocity = self.v_min
        else:
            velocity = v_max - (weight/self.bag_max_weight)*(v_max-self.v_min)
        time = distance / velocity
            
        return time
    
    def ks_time_fitness(self, chromo, route, item_per_city, chromo_weight):
        time_per_chromo = 0
        for idx in range(len(route)):
            # Calculate time per city by iterating through each city index
            time_per_city = self.ks_intercity_time(chromo, self.dist_mat,
                                                   route,items_per_city, chromo_weight, idx)
            # print(f"city covered {idx} time_per_city: {round(time_per_city,2)} ")
            time_per_chromo += time_per_city
        return time_per_chromo
    
    
    def ks_val_time_fitness(self, population, item_per_city, complete_values, total_route, complete_weights):
        
        tot_val_time = []
        for i in range(route_len):
            # Getting the value fetched for items
            values_per_chromo = self.ks_chromo_value(complete_values[i], population[i])
            # Getiing time for chromosome for one particular route
            time_route = self.ks_time_fitness(population[i], total_route[i], items_per_city, complete_weights[i])
            # print("Total time for a route: ", time_route)
            # print("------------------------")
            val_time = round(values_per_chromo / time_route, 2)
            tot_val_time.append(val_time)
            
        return tot_val_time
    
        

cities = 4; max_items = 5; max_wt = 20; max_val = 20; v_min =1; v_max=20; k = 2 # mutation based on no of cities
bag_max_weight = 20; total_route = [[2,1,3,4], [1,3,2,4],[2,3,4,1], [4,3,2,1]]; 
perform_crossover = 1;perform_mutation=1; distance_matrix =  [[0, 5, 1, 2], [5, 0, 1, 2],[1, 2, 0, 9],[1, 2, 9, 0]]
term_criteria = 0; colormap = ['b','g','c','m']; marker= ['^','<','o','v']

route_len = len(total_route)
kga = KnapsackGA(cities, max_items, v_min, bag_max_weight, distance_matrix)
items_per_city, population, complete_weights, complete_values = kga.ks_population(route_len, max_wt, max_val)


print(f" Population:\n {population}")
print("------------------------")
print(f" Weigths:\n {complete_weights}")
print("------------------------")
print(f"Values:\n {complete_values}")


tot_route_val , tot_route_time = [],[]


while(term_criteria < 5):
    term_criteria +=1;
    print("term_criteria pop\n",population)
    val_time_fit = kga.ks_val_time_fitness(population, items_per_city, complete_values, total_route, complete_weights)
    print("Val-Time fitness for all routes: ",val_time_fit)
    
    
    parent_1, route1 = kga.ks_tournament_selection(population, val_time_fit)
    parent_2, route2 = kga.ks_tournament_selection(population, val_time_fit)
    print(f"Parent 1: {parent_1}, Route Selected: {route1}")
    print(f"Parent 2: {parent_2},  Route Selected: {route2}")
    
    ch1, ch2 = kga.ks_crossover(parent_1, parent_1, perform_crossover)
    print(f"Child1: {ch1}, Child2: {ch2}")
    mut1 = kga.ks_mutate(ch1, k, perform_mutation)
    mut2 = kga.ks_mutate(ch2, k, perform_mutation)
    print(f"Mutation1: {mut1}, Mutation2: {mut2}")
    print("------------------------")
    population, val_time_fit = kga.ks_weakest_replacement(population, val_time_fit, mut1, total_route[route1], items_per_city,
                                complete_values[route1], complete_weights[route1])
    print(f"Population after replacement1:\n {population}\n Mutated Val-Time Fitness1:{val_time_fit}")
    population, val_time_fit = kga.ks_weakest_replacement(population, val_time_fit, mut2, total_route[route2], items_per_city,
                                complete_values[route2], complete_weights[route2])
    print(f"Population after replacement2:\n {population}\n Mutated Val-Time Fitness2:{val_time_fit}")
    
    final_val = [ kga.ks_chromo_value(population[i],complete_values[i]) for i in range(route_len)] 
    
    final_time = [kga.ks_time_fitness(population[i], total_route[i], items_per_city, complete_weights[i]) for i in range(route_len) ]
    
    tot_route_val.append(final_val)
    tot_route_time.append(final_time)

print(f"Final pop val: {tot_route_val}")
print(f"Final pop time: {tot_route_time}")

for i in range(len(tot_route_val)):
    kga.ks_plot(tot_route_val[i], tot_route_time[i], colormap, marker)  
    
label=['Route'+str(i+1) for i in range(route_len)]
plt.legend(label, bbox_to_anchor=(1.01, 1.0))
plt.show()
