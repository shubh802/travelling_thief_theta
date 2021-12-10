# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Knapsack via EA file.
"""
import os
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
    
    
    def ks_population(self, no_of_chromo):
        """ 
            Knapsack population creation, initial weights and value of items is created.
            Return: Population, Items per city, Weights, Values
        """
        items_per_city = np.random.randint(1, max_items, size= self.n_cities)
        total_items = sum(items_per_city)
        # Calculating weight for each item in the route, Weight of the item cannot be 0
        weights = np.random.randint(1,self.bag_max_weight, size= total_items)
        # Calculating value for each item in the route, Value of the item cannot be 0
        value = [np.random.randint(1,max_val)*10 for i in range(total_items)]
        # Population of the chromosome
        population = [np.random.randint(2, size = total_items) for i in range(no_of_chromo)]
                
        return population, items_per_city, weights, value
    
    
    def ks_tournament_selection(self, pop, scores):
        """
            Random toutnament selection based on the fitness scores
            Return: Chromosome 
        """
        rnd_var1 = np.random.randint(len(pop))
        rnd_var2 = np.random.randint(len(pop))
        # Selecting parent based on the fitness
        if(scores[rnd_var1] > scores[rnd_var2]): index = rnd_var1
        elif(scores[rnd_var1] == scores[rnd_var2]): index = choice([rnd_var1, rnd_var2])
        else: index = rnd_var2
        return pop[index]
    
    def ks_crossover(self, prnt1, prnt2, perform):
        """
            Crossover between both the parents to create two child.
            Return: Child1, Child2
        """
        child1, child2 = prnt1.copy(), prnt2.copy()
        if perform == 1:
            # Selecting random point from the length of the parent
            pt = np.random.randint(1, len(prnt1)-1)
            # crossover performed from the point selected between the two parents
            child1 = np.concatenate([prnt1[:pt], prnt2[pt:]])
            child2 = np.concatenate([prnt2[:pt], prnt1[pt:]])
        return child1, child2
    
    def ks_mutate(self, chromo, k, perform_mutation):
        """
            Mutating chromosome k no of times with random selection of index
            Return: Chromosome
        """
        # Mutating items picked up by the thief based on k
        if perform_mutation == 1:
            for i in range(k):
                # Selecting index randomly
                n = np.random.randint(len(chromo))
                # print(f"ks_mutate chromo value: {chromo[n]} at index {n}")
                if chromo[n] == 1: chromo[n] = 0
                else: chromo[n] = 1
            
        return chromo;
    
    def ks_weakest_replacement(self, pop, scores, mutation, route, items_per_city, value, wt):
        """
            Mutating chromosome k no of times with random selection of index
            Return: Population, Fitness Score
        """
        #  fitness score of the mutation
        mut_val_time_fit = self.ks_fitness(mutation, items_per_city, value, route, wt)
        print(f"Mutation val-time score: {mut_val_time_fit}")
        min_fit = min(scores)
        if mut_val_time_fit > min_fit:
            min_fit_idx_lst = []
            for i in range(len(scores)):
                if scores[i] == min_fit : min_fit_idx_lst.append(i)
            # Breaking ties randomly in a list of low fitness values
            idx = np.random.choice(min_fit_idx_lst)
            print(f"Chromosome to be replaced: {pop[idx]}, Chromosome index: {idx}")
            pop[idx] = mutation
            # adding mutation score to the original score list
            scores[idx] = mut_val_time_fit
        return pop, scores
    
    def ks_plot(self, value, time, colormap, marker, label):
        """
            Plotting total values per route and time taken to cover it
        """
        
        for i in range(len(value)):
            plt.scatter(time[i], value[i],color=colormap, alpha=.50, zorder=2, marker=marker, label=label)
        plt.xlabel("Time")
        plt.ylabel("Total values per chromosome")
        plt.savefig('value_time.png')

        

    def ks_bag_weight(self, chromo, no_of_items,weight_list, city_index):
        """
            Calculating the knapsack bag weight for each city movement.
            Return: Weight
        """
        #Calculating the weight of the bag at each city
        items_considered = sum(no_of_items[:city_index+1])
        weights_considered = weight_list[:items_considered]
        chromosome_considered = chromo[:items_considered]  
        # Calculating weight of bag 
        weight = np.dot(weights_considered, chromosome_considered)
        return weight
    
    
    def ks_chromo_value(self, value, chromo):
        """
            Calculating value of the chromosome based on items selected and its values.
            Return: Values
        """
        return np.dot(value, chromo)


    def ks_intercity_time(self, chromo, distance_matrix, route, items_per_city, chromo_weight, city_index):
        #calculating time taken to travel between cities
        city1, city2 = route[city_index-1], route[city_index]
        # print(f"city1 {city1}:city2: {city2}")
        # Reducing 1 from city to match distance matrix (0,1,2,3) 
        distance = distance_matrix[city1-1][city2-1]
        weight = self.ks_bag_weight(chromo,items_per_city,chromo_weight, city_index)
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
    
    def ks_save_route_data(self, filename, list1, list2):
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path,'w+') as file:
            for item1, item2 in zip(list1, list2): 
                file.write(" ".join(map(str,item1))+'\n')
                file.write(" ".join(map(str, item2))+'\n\n')
                
    def ks_save_value_data(self, filename, value1, value2):
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path,'w+') as file:
            for item1, item2 in zip(value1, value2): 
                file.write(str(item1)+' '+ str(item2)+'\n')
    
    
    def ks_algorithm(self, population, items_per_city, value, route, weights, k, 
                     perform_crossover, perform_mutation, term_criteria, no_of_chromo):
        
        # Running knapsack algo for each route based on termination criteria
        while(term_criteria > 0):
            term_criteria -=1 ;
            val_time_fit = [self.ks_fitness(population[i], items_per_city, value, route, weights) 
                                    for i in range(no_of_chromo)]
            print("Val-Time fitness for all chromosome: ",val_time_fit)
            parent_1 = self.ks_tournament_selection(population, val_time_fit)
            parent_2 = self.ks_tournament_selection(population, val_time_fit)
            print(f"Parent 1: {parent_1}\nParent 2: {parent_2}")
            ch1, ch2 = self.ks_crossover(parent_1, parent_2, perform_crossover)
            print(f"Child1: {ch1}\n Child2: {ch2}")
            mut1 = self.ks_mutate(ch1, k, perform_mutation)
            mut2 = self.ks_mutate(ch2, k, perform_mutation)
            print(f"Mutation1: {mut1}, Mutation2: {mut2}")
            population, val_time_fit = self.ks_weakest_replacement(population, val_time_fit, mut1, route, items_per_city,
                                            value, weights)
            print(f"Population after replacement1:\n {population}\n Mutated Val-Time Fitness1:{val_time_fit}")
            
            population, val_time_fit = self.ks_weakest_replacement(population, val_time_fit, mut1, route, items_per_city,value, weights)
            print(f"Population after replacement2:\n {population}\n Mutated Val-Time Fitness2:{val_time_fit}")
            
            # Best chromosome fitness is selected per iterations
            max_chromo_fitness = max(val_time_fit)
            max_chromo_idx = val_time_fit.index(max_chromo_fitness)
            best_chromo = population[max_chromo_idx]
            print(f"max_chromo_fitness: {max_chromo_fitness}, index: {max_chromo_idx}")
            
            final_val = [ self.ks_chromo_value(population[i],value) for i in range(no_of_chromo)] 
            final_time = [self.ks_time_fitness(population[i], route, items_per_city,
                                                  weights) for i in range(no_of_chromo) ]
            
            print(f" Final Value: {final_val}, Final time: {final_time}")
           
        best_route_value = self.ks_chromo_value(population[max_chromo_idx], value)
        best_route_time = self.ks_time_fitness(population[max_chromo_idx], route, items_per_city, weights)
            
        return final_val, final_time, best_chromo, best_route_value, best_route_time


    def ks_experiments(self, population, items_per_city, value, total_route, weights, k, perform_crossover,
                             perform_mutation, term_criteria, no_of_chromo):

        route_val_lst, route_time_lst, route_chromo_lst = [], [], []

        for idx, route in enumerate(total_route):

            label = 'Route'+str(idx+1)
            tot_route_val,tot_route_time,best_chromo,best_route_value,best_route_time = self.ks_algorithm(population, 
                            items_per_city, value, route, weights, k, perform_crossover, perform_mutation, term_criteria, no_of_chromo)
            print(f"best_route_value {best_route_value},best_route_time {best_route_time}, best_chromo {best_chromo}")
        
            route_val_lst.append(best_route_value)
            route_time_lst.append(best_route_time)
            route_chromo_lst.append(best_chromo)
            
            self.ks_plot(tot_route_val, tot_route_time, colormap[idx], marker[idx], label)  

        return route_chromo_lst, route_val_lst, route_time_lst
            







