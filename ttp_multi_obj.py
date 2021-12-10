import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import os

class TTMultiObjective():
    
    def __init__(self,route,max_items,bag_max_weight, min_weight, max_weight, 
                 min_value, max_value, pop_size, v_max, v_min, distance_matrix, setup):
    #Basic Parameters
        self.route = route
        self.no_of_cities = len(route)
        self.max_items = max_items
        self.bag_max_weight = bag_max_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_value = min_value
        self.max_value = max_value
        self.pop_size = pop_size
        self.v_max = v_max
        self.v_min = v_min
        self.distance_matrix = distance_matrix
        self.setup = setup
    

    def bag_weight(self, chromo, city_index, no_of_items, weight_list):
        """
            Calculating the weight of the bag at each city given the chromosome
            and list of item weights
            Return: weight
        """
        items_considered = sum(no_of_items[:city_index])
        
        weights_considered = weight_list[:items_considered]
        chromo_considered = chromo[:items_considered]
        
        weight = np.dot(weights_considered, chromo_considered)
        
        return weight

    def intercity_time(self, chromo, city_index, no_of_items, weight_list):
        """
            Calculating the time taken to travel between cities based on the
            weight of the bag and the distance between cities
            and list of item weights
            Return: intercity time
        """
        city1, city2 = self.route[city_index-1], self.route[city_index]
        distance = self.distance_matrix[city1][city2]
        
        weight = self.bag_weight(chromo, city_index, no_of_items, weight_list)
        
        if weight > self.bag_max_weight:
            velocity = self.v_min
        else:
            velocity = self.v_max - (weight/self.bag_max_weight)*(self.v_max-self.v_min)
            
        time = distance/velocity
        
        return time
    
    def total_time(self, chromo, no_of_items, weight_list):
        """
            Calculating the total time taken to complete a route with
            a given packing plan (chromosome).
            Return: total time
        """
        time_t = 0
        for city_index in range(1,len(self.route)):
            intercity_t = self.intercity_time(chromo, city_index, no_of_items, weight_list)
            time_t += intercity_t
        
        return time_t

    def times_values_lists(self, chromosomes, no_of_items, value_list, weight_list):
        """
            This function finds the total times and values for each packing plan (chromosome).
            Return: list of times, list of values
        """
        times = []
        values = []

        for chromosome in chromosomes:
            time = self.total_time(chromosome, no_of_items, weight_list)
            value = np.dot(value_list, chromosome)
            times.append(time)
            values.append(value)
        return times, values

    def rankings(self, time, value):
        """
            This function ranks the chromosomes in terms of dominance.
            Return: A list of dataframes for each rank containing times, values and indices.
        """
        # Converting list to a dataframe
        df = pd.DataFrame(time, value).reset_index()
        df.rename(columns={'index': 'value', 0: 'time'}, inplace=True)
        # Empty list to store ranks
        ranks = []
        # Iterating number for rank
        n = 1
        # Storing the orginal data
        df2 = df.copy()
        while len(df2) > 0:
            l = []
            # Setting a boolean variable for Dominance
            Dominance = True

            for a in (df2.index.values):
                Dominance = True
                for b in (df2.index.values):
                    if a == b:
                        continue
                    # Comparing the multiobjective dominations
                    if df2['time'][b] <= df2['time'][a] and df2['value'][b] > df2['value'][
                        a]:  # If value is more and if time is less
                        Dominance = False
                    elif df2['time'][b] < df2['time'][a] and df2['value'][b] >= df2['value'][a]:
                        Dominance = False
                if Dominance:
                    l.append(a)
            ranks.append(df2.loc[l])
            # Dropping the value from the copy list
            df2 = df2.drop(index=l)
            n = n + 1
        # Returns the indexes of the ranks
        return ranks
    
    
    def crowding_distance(self, rank, parent1_ind, parent2_ind):
        """
            Calculating crowding distance. This function calculates the corwding distance for
            all points in a rank. Of the two parent indices passed to this function, the one
            with the larger crowding distance is returned
            Return: Index of the parent with the larger crowding distance
        """
        cd_list = []
        
        #sorting the dataframe
        sorted_rank = rank.sort_values('value')
        sorted_rank.reset_index(inplace = True)
        
        for ind in sorted_rank.index:
            if ind == 0:
                distance = float('inf')
            elif ind == len(sorted_rank)-1:
                distance = float('inf')
            else:
                value_width = sorted_rank['value'][ind+1] - sorted_rank['value'][ind-1]
                time_width = sorted_rank['time'][ind+1] - sorted_rank['time'][ind-1]
                distance = value_width + time_width
                
            cd_list.append(distance)
        
        sorted_rank.insert(2, "CrowdingD", cd_list)
        sorted_rank.set_index('index', inplace=True)
        
        cd1 = sorted_rank['CrowdingD'][parent1_ind]
        cd2 = sorted_rank['CrowdingD'][parent2_ind]

        if cd1 >= cd2:
            parent = parent1_ind
        else:
            parent = parent2_ind
        
        return parent
    
    def worst_crowding_distance(self, rank):
        """
            This function finds the index of the chromosome with the lowest crowding
            distance in the rank so that it can be replaced by a successful child
            chromosome.
            Return: Index of the lowest crowding distance chromosome in the rank.
        """
        cd_list = []
        
        #sorting the dataframe
        sorted_rank = rank.sort_values('value')
        sorted_rank.reset_index(inplace = True)
        
        for ind in sorted_rank.index:
            if ind == 0:
                distance = float('inf')
            elif ind == len(sorted_rank)-1:
                distance = float('inf')
            else:
                value_width = sorted_rank['value'][ind+1] - sorted_rank['value'][ind-1]
                time_width = sorted_rank['time'][ind+1] - sorted_rank['time'][ind-1]
                distance = value_width + time_width
                
            cd_list.append(distance)
        
        sorted_rank.insert(2, "CrowdingD", cd_list)
        sorted_rank.set_index('index', inplace=True)
        
        worst_ind = sorted_rank[['CrowdingD']].idxmin()
        
        return worst_ind

    def pareto_plot(self, time, value, color):
        """
            This function creates a pareto plot for the population, illustrating the
            pareto front by connecting all the points in the 1st rank.
        """
        #Call the ranking function
        ranks=self.rankings(time, value)
        if color == None:
            color1 = 'blue'
            color2 = 'red'
        else:
            color1 = color
            color2 = color

        for i in ranks:
            #To check if it's the rank 1 dataframe
            if i.equals(ranks[0]) is True:
                #Sorting values for line plot to go in ascending order
                i=i.sort_values(by=['time'])
                i=i.sort_values(by=['value'])
                time = i['time'].values
                value = i['value'].values
                #Plotting with blue markers
                plt.plot(time,value,marker=11, color=color1)
            else:
                #The rest of the values
                time = i['time'].values
                value = i['value'].values
                plt.scatter(time,value, color=color2)

        plt.xlabel('Total Time')
        plt.ylabel('Total Value')
    
    def pareto_tournament(self, ranks, chromosomes):
        """
            This function is used for selecting parent chromosomes from the population.
            Two parent indices are chosen and if one of the parent chromosomes ranks higher
            than the other, it's index is returned. Otherwise, the indices are passed to
            the crowding distance function for a tiebreaker.
            Return: Parent chromosome index.
        """
        
        #randomly picking parents
        [parent1_ind, parent2_ind] = random.sample(range(len(chromosomes)), 2)
        
        parent1_rank = 0
        parent2_rank = 0
        count = 1
        
        for rank in ranks:
            if parent1_ind in list(rank.index.values):
                parent1_rank = count
            
            if parent2_ind in list(rank.index.values):
                parent2_rank = count
                
            count+=1
            
        if parent1_rank < parent2_rank:
            return parent1_ind
        elif parent1_rank > parent2_rank:
            return parent2_ind
        else:
            parent_ind = self.crowding_distance(ranks[parent1_rank-1], parent1_ind, parent2_ind)
            return parent_ind
    
    def sp_crossover(self, a, b):
        """
            This function performs a single point crossover of the two chromosomes, a & b.
            Return: Two child chromosomes.
        """
        c1 = a.copy()
        c2 = b.copy()
        point = np.random.randint(len(c1)-1)
        
        c1 = np.concatenate([c1[:point], c2[point:]])
        c2 = np.concatenate([c2[:point], c1[point:]])
                
        return c1, c2
    
    def mutation(self, chromo, k, setup):
        """
            This function performs k mutations on the input chromosome.
            Return: A mutated child chromosome.
        """
        for i in range(k):
            copy = chromo.copy()
            gene = np.random.randint(len(copy)-1)
            if setup == 0:
                value = np.random.randint(5)
                copy[gene] = value
            else:
                if copy[gene] == 1: copy[gene] = 0
                else: copy[gene] = 1
        return copy
            
    def find_replacements(self, ranks, chromo, chromosomes, no_of_items, value_list, weight_list):
        """
            This function used for placing child chromosomes in the population. If the child
            dominates the chromosome in the lowest rank with the lowest crowding distance,
            the child replaces it. Otherwise, the child chromosome is discarded.
        """
        #if chromo in chromosomes:
            #return 0
        last_rank = ranks[-1]
        child_time = self.total_time(chromo, no_of_items, weight_list)
        child_value = np.dot(value_list, chromo)

        for ind in last_rank.index:
            value = last_rank['value'][ind]
            time = last_rank['time'][ind]
            if child_value >= value and child_time < time:
                chromosomes[ind] = chromo
                break
            elif child_value > value and child_time <= time:
                chromosomes[ind] = chromo
                break

    def save_route_data(self, filename, route, list2):
        """
            This function saves the route and chromosome data to a file.
        """
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path,'a+') as file:
            for item2 in list2: 
                file.write(" ".join(map(str, route))+'\n')
                file.write(" ".join(map(str, item2))+'\n\n')
                
    def save_value_data(self, filename, value1, value2):
        """
            This function saves the value and time data to a file.
        """
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path,'a+') as file:
            for item1, item2 in zip(value1, value2): 
                file.write(str(item1)+' '+ str(item2)+'\n')
    
    def algorithm(self, limit, gene_k, crossover,chromosomes, no_of_items, value_list, weight_list, color):
        """
            In this function, the algorithm is evolved. In each iteration of the loop, the times and values
            of each chromosome are evaluated first. Using these, the solutions are then ranked. Two parents
            are then selected using the pareto_tournament function. These parents are evolved using
            crossover and mutation to produce two child chromosomes.

            Each child chromosome is then compared with the existing population and if it does not already
            exist in the population, the find_replacements function is run with the child chromosome.
        """
        k = len(value_list)//gene_k
        #evolving the algorithm
        for i in range(limit):
            times, values = self.times_values_lists(chromosomes, no_of_items, value_list, weight_list)
            ranks = self.rankings(times, values)

            parent_a = self.pareto_tournament(ranks, chromosomes)
            parent_b = self.pareto_tournament(ranks, chromosomes)
            
            p1 = np.copy(chromosomes[parent_a])
            p2 = np.copy(chromosomes[parent_b])
            
            #crossover
            if crossover == 1:
                c1,c2 = self.sp_crossover(p1, p2)
            
            #mutation
            
            c1 = self.mutation(c1, k, self.setup)
            c2 = self.mutation(c2, k, self.setup)
            
            #replacements
            if c1.tolist() not in chromosomes.tolist():
                self.find_replacements(ranks, c1, chromosomes, no_of_items, value_list, weight_list)
            
            times, values = self.times_values_lists(chromosomes, no_of_items, value_list, weight_list)
            ranks = self.rankings(times, values)

            if c2.tolist() not in chromosomes.tolist():
                self.find_replacements(ranks, c2, chromosomes, no_of_items, value_list, weight_list)
            
            
        times, values = self.times_values_lists(chromosomes, no_of_items, value_list, weight_list)
        self.pareto_plot(times, values, color)
        
        return chromosomes, times, values