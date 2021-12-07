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
    
    
    def ks_population(self, no_of_chromo, total_items):
        
        return [np.random.randint(2, size = total_items) for i in range(no_of_chromo)]
    
    
    def ks_tournament_selection(self, pop, scores):
        rnd_var1 = np.random.randint(len(pop))
        rnd_var2 = np.random.randint(len(pop))
        # print(f"rnd_var1: {rnd_var1}, rnd_var2: {rnd_var2}")
        # Selecting parent based on the fitness
        if(scores[rnd_var1] > scores[rnd_var2]): index = rnd_var1
        elif(scores[rnd_var1] == scores[rnd_var2]): index = choice([rnd_var1, rnd_var2])
        else: index = rnd_var2
        return pop[index]
    
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

        print("route: ",route)
        mut_val_time_fit = self.ks_fitness(mutation, items_per_city, value, route, wt)
        
        print(f"Mutation val-time score: {mut_val_time_fit}")
        print(f"ks_weakest_replacement fitness scores: ",scores)
        
        min_fit = min(scores)
        if mut_val_time_fit > min_fit:
            min_fit_idx_lst = []
            for i in range(len(scores)):
                if scores[i] == min_fit : min_fit_idx_lst.append(i)
            idx = np.random.choice(min_fit_idx_lst)
            print(f"Chromosome to be replaced: {pop[idx]}, Chromosome index: {idx}")
            pop[idx] = mutation
            scores[idx] = mut_val_time_fit
        return pop, scores
    
    def ks_plot(self, value, time, colormap, marker):
        
        for i in range(len(value)):
            plt.scatter(time[i], value[i],color=colormap, alpha=.50, zorder=2, marker=marker)
        plt.xlabel("Time")
        plt.ylabel("Total values per chromosome")

        
        #Calculating the weight of the bag at each city
    def ks_bag_weight(self, chromo, no_of_items,weight_list, city_index):
        items_considered = sum(no_of_items[:city_index+1])
        weights_considered = weight_list[:items_considered]
        chromosome_considered = chromo[:items_considered]  
        weight = np.dot(weights_considered, chromosome_considered)
        
        return weight
    
    
    def ks_chromo_value(self, value, chromo):
        return np.dot(value, chromo)

    
        #calculating time taken to travel between cities
    def ks_intercity_time(self, chromo, distance_matrix, route, items_per_city, chromo_weight, city_index):
        # print("ks_intercity_time route: ", route)
        # print("city_index : ", city_index)
        
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
    
    def ks_fitness(self, chromo, item_per_city, values, route, complete_weights):
        
        values_per_chromo = self.ks_chromo_value(values, chromo)
        time_route = self.ks_time_fitness(chromo, route, items_per_city, complete_weights)
        # print(f"values_per_chromo: {values_per_chromo}, time_route: {time_route}, index: {i}")
        val_time = values_per_chromo / time_route
        return round(val_time,2)
    
    
    def ks_algorithm(self, population, items_per_city, value, route, weights, 
                     perform_crossover, perform_mutation, term_criteria, no_of_chromo):
        
        # Running knapsack algo for each route, 
        tot_route_val , tot_route_time = [],[]
        while(term_criteria < 5):
            # Each route is being run in 5 iterations
            term_criteria +=1 ;
            val_time_fit = [self.ks_fitness(population[i], items_per_city, value, route, weights) 
                                    for i in range(no_of_chromo)]
            print("Val-Time fitness for all chromosome: ",val_time_fit)
            parent_1 = self.ks_tournament_selection(population, val_time_fit)
            parent_2 = self.ks_tournament_selection(population, val_time_fit)
            print(f"Parent 1: {parent_1}")
            print(f"Parent 2: {parent_2}")
            ch1, ch2 = self.ks_crossover(parent_1, parent_2, perform_crossover)
            print(f"Child1: {ch1}, Child2: {ch2}")
            mut1 = self.ks_mutate(ch1, k, perform_mutation)
            mut2 = self.ks_mutate(ch2, k, perform_mutation)
            print(f"Mutation1: {mut1}, Mutation2: {mut2}")
            print("------------------------")
            population, val_time_fit = self.ks_weakest_replacement(population, val_time_fit, mut1, route, items_per_city,
                                            value, weights)
            print(f"Population after replacement1:\n {population}\n Mutated Val-Time Fitness1:{val_time_fit}")
            
            population, val_time_fit = self.ks_weakest_replacement(population, val_time_fit, mut1, route, items_per_city,value, weights)
            print(f"Population after replacement2:\n {population}\n Mutated Val-Time Fitness2:{val_time_fit}")
            
            # Best chromosome fitness is selected per iterations
            max_chromo_fitness = max(val_time_fit)
            max_chromo_idx = val_time_fit.index(max_chromo_fitness)
            print(f"max_chromo_fitness: {max_chromo_fitness}, index: {max_chromo_idx}")
            best_chromo = population[max_chromo_idx]
            
            # # Max best value is calculated per iterations
            # max_best_val = self.ks_chromo_value(population[max_chromo_idx], value)
            # max_best_time = self.ks_time_fitness(population[max_chromo_idx], route, items_per_city, weights)
            
            # print(f"Best Chromosome Total Val: {max_best_val} Best Chromosome Total Time: {max_best_time}")

            # tot_route_val.append(max_best_val)
            # tot_route_time.append(max_best_time)
            
            
            final_val = [ self.ks_chromo_value(population[i],value) for i in range(no_of_chromo)] 
            final_time = [self.ks_time_fitness(population[i], route, items_per_city,
                                                  weights) for i in range(no_of_chromo) ]
            
            print(f" Final Value: {final_val}, Final time: {final_time}")
            
            
        return final_val, final_time, best_chromo   
            
        # return tot_route_val, tot_route_time, best_chromo
    

    
    
        

max_items = 5; max_wt = 20; max_val = 20; v_min =1; v_max=20; k = 2 # mutation based on no of cities
bag_max_weight = 20;  route = [2,1,3,4]; no_of_chromo = 5
perform_crossover = 1; perform_mutation=1; 
# distance_matrix =  [[0, 5, 1, 2], [5, 0, 1, 2],[1, 2, 0, 9],[1, 2, 9, 0]]
distance_matrix = [[0., 9., 2., 1., 7.],
[9., 0., 3., 8., 6.],
[2., 1., 0., 1., 4.],
[7., 3., 8., 0., 4.],
[6., 1., 4., 4., 0.]]
term_criteria = 0; colormap = ['b','g','c','m','#7BC8F6']; marker= ['^','<','o','v','>']
# total_route = [[2,1,3,4], [1,3,2,4],[2,3,4,1], [4,3,2,1]];
total_route= [[4, 3, 1, 2, 0], [4, 1, 2, 0, 3], [4, 1, 0, 2, 3], [4, 1, 2, 3, 0]]
cities = len(total_route[0])
print("cities: ", cities)



np.random.seed(5)

items_per_city = np.random.randint(1, max_items, size= cities)
total_items = sum(items_per_city)
# Calculating weight for each item in the route, Weight of the item cannot be 0
weights = np.random.randint(1,max_wt, size= total_items)
# Calculating value for each item in the route, Value of the item cannot be 0
value = [np.random.randint(1,max_val)*10 for i in range(total_items)]

kga = KnapsackGA(cities, max_items, v_min, bag_max_weight, distance_matrix)


population = kga.ks_population(no_of_chromo, total_items)

print(f" Item per city:\n {items_per_city}")
print("------------------------")
print(f" Population:\n {population}")
print("------------------------")
print(f" Weigths:\n {weights}")
print("------------------------")
print(f"Values:\n {value}")

best_route_val, best_route_time, best_route_chromo = [],[],[]

for idx, route in enumerate(total_route):

    # tot_route_val, tot_route_time, best_chromo = kga.ks_algorithm(population, items_per_city, value, route, weights, 
    #                                                               perform_crossover, perform_mutation, term_criteria, no_of_chromo)
    
    tot_route_val, tot_route_time, best_chromo = kga.ks_algorithm(population, items_per_city, value, route, weights, 
                                                                  perform_crossover, perform_mutation, term_criteria, no_of_chromo)
    kga.ks_plot(tot_route_val, tot_route_time, colormap[idx], marker[idx])  
    
# label=['Route '+str(i+1) for i in range(len(total_route))]
plt.legend(bbox_to_anchor=(1.01, 1.0))
plt.show()




