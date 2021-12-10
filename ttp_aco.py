import math
import numpy as np
import random
import pandas as pd
import string
import time
import cProfile
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import SubplotSpec
from collections import deque
from tqdm import tqdm



class Tsp(object):

    def __init__(self, n_cities, alpha=1, beta=2, n_ants=5, tau_init=1, sc=1000, evaporation=0.5, res_type=1,
                 res_len=5):

        #         define random distance matrix
        self.distance = self.define_distance_matrix(n_cities)
        #         define undirected graph with distances
        self.G = nx.from_numpy_matrix(self.distance)

        self.heuristic = np.array(
            [[round(1 / self.distance[i][j], 4) if i != j else 0 for j in range(n_cities)] for i in range(n_cities)])

        #         get shortest path from random start point
        self.shortest_path, self.short_path_time = self.shortest_tsp(n_cities, np.random.randint(0, n_cities))

        if tau_init == 1:
            start = np.ones((n_cities, n_cities))
            self.tau = start * (n_ants / (self.short_path_time))
        elif tau_init == 2:
            start = np.ones((n_cities, n_cities))
            self.tau = start * ((n_ants + n_cities) / (evaporation * self.short_path_time))
        else:
            self.tau = np.ones((n_cities, n_cities))

        #         initialise counter for determining stopping crierion
        c = 0

        #         initialise lists containing all paths and associated distances
        self.result_cities = [[] for i in range(n_cities)]
        self.distances_cities = [[] for i in range(n_cities)]
        self.result = []
        self.distances = []

        self.output = []

        while c < sc:
            st = time.time()

            #             get paths for all ants on single run
            self.paths = self.ant_run(n_ants, n_cities, alpha, beta)

            for i in self.paths:
                self.result_cities[i[0]].append(i)
                self.distances_cities[i[0]].append(self.get_distance(i))
                self.result.append(i)
                self.distances.append(self.get_distance(i))

            #             update pheromones
            self.pheromone_update(n_cities, evaporation=evaporation)
            c += 1
            et = time.time()
            print(f'{c}/{sc}|Time_remaining:{round(((sc - c) * (et - st)) / 60, 2)}min', end="\r")

        #         used to determine what result type is wanted
        #         result type 2 is used for output into format required for submission
        if res_type == 1:
            if res_len > n_cities:
                indexes = [y for x in self.distances_cities for y in np.argsort(x)[:1]]
                for i in range(len(indexes)):
                    self.output.append(self.result_cities[i][indexes[i]])
            else:
                indexes = [y for x in self.distances_cities for y in np.argsort(x)[:1]]
                random.shuffle(indexes)
                for i in range(res_len):
                    self.output.append(self.result_cities[i][indexes[i]])
        elif res_type == 2:
            indexes = [x for x in np.argsort(self.distances)]
            for i in indexes:
                if len(self.output) < res_len:
                    if self.result[i] not in self.output:
                        self.output.append(self.result[i])
                else:
                    break

    #     get distance matrix which will be used in Knapsack and throughout ACO
    def define_distance_matrix(self, n_cities):

        X = np.zeros((n_cities, n_cities))

        rand_distant = np.random.randint(1, 20, X[np.triu_indices(n_cities, k=1)].shape[0])

        X[np.triu_indices(n_cities, k=1)] = rand_distant
        X[np.tril_indices(n_cities, k=-1)] = rand_distant
        return X

    #     get shortest path for pheromone initialisation
    def shortest_tsp(self, n_cities, start_point):

        labels = nx.get_edge_attributes(self.G, "weight")
        names = nx.get_node_attributes(self.G, "name")

        counter = 0
        path = [start_point]
        dist = []

        #         define path by looking at nearest neighbours from start point
        while counter < n_cities - 1:

            if counter == 0:
                lengths = nx.shortest_path_length(self.G, source=start_point, weight="weight")
                for cities in path:
                    del lengths[cities]
                next_stop = min(lengths, key=lengths.get)
                dist.append(min(lengths.values()))
                path.append(next_stop)
                counter += 1
            else:
                lengths = nx.shortest_path_length(self.G, source=next_stop, weight="weight")
                for cities in path:
                    del lengths[cities]
                next_stop = min(lengths, key=lengths.get)
                dist.append(min(lengths.values()))
                path.append(next_stop)
                counter += 1

        return path, sum(dist)

    #     set ants running across routes
    def ant_run(self, n_ants, n_cities, alpha, beta):

        final_paths = []

        temp_heuristic = self.heuristic.copy()
        temp_pheremone = self.tau.copy()

        #         for each start point
        for city in range(n_cities):

            temp_city = temp_heuristic[city, :]
            temp_cities_phero = temp_pheremone[city, :]

            #             for each ant to create a path
            for ant in range(n_ants):

                try:

                    path = [city]
                    #                 for each possible stop to generate the route
                    for stop in range(n_cities - 1):
                        probabilities = ((temp_cities_phero) ** alpha) * ((temp_city) ** beta)

                        probabilities[path] = 0

                        proper_probs = probabilities / probabilities.sum()

                        next_stop = np.random.choice([i for i in range(n_cities)], p=proper_probs)
                        path.append(next_stop)
                    final_paths.append(path)

                except Exception as e:
                    print(f'path:{path} probs: {probabilities}, h_diffs: {higher_diffs}\n')
                    print(f'H: {temp_cities_phero}\n')
                    print(e)
        return final_paths

    #    use this to get distance associated with path
    def get_distance(self, path):

        return sum([self.distance[path[i]][path[i + 1]] for i in range(len(path) - 1)])

    #    use this to update pheromones after all ants have run
    def pheromone_update(self, n_cities, evaporation=0.5):
        self.tau = self.tau * evaporation

        for i in self.paths:
            delta = 1 / self.get_distance(i)
            for j in range(len(i) - 1):
                self.tau[i[j]][i[j + 1]] += delta

