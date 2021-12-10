# Travelling Thief Problem
Team Theta worked on the Travelling Thief Problem (TTP) from the 2019 Gecco list of challenges.
This problem is a bi-objective problem where the objectives come from two separate well studied NP-hard problems. The first problem is the travelling salesman problem (TSP). For this, a salesman must visit n cities and each city must be visited only once. The best solution to this problem aims to minimise the time taken to visit all cities.
The second problem is the knapsack problem. This involves the filling of a ‘knapsack’ with items that have a certain value. Each item also has a weight attached to it and so the knapsack problem is solved by maximising the profit from items packed in the knapsack, ensuring that the maximum weight the knapsack can hold is not exceeded. 



## Installation
Travelling Thief problem requires Python 3.8[https://www.python.org/downloads/] 

Create a virtual Python Env

For installing dependencies

```sh
pip install -r requirements.txt
```

For running the code we need to run the main.py, which internally calls ACO to give us the routes and the distance matrix

argv1 -> The no of cities that we want our ACO and EA to compute   

argv2 -> Test Cases we want to run code with eg (0,1,2)
0: Bounded Strongly Correlated
    - 1 item per city
    - Individual items may be picked up more than once, up to 4 times.
    - Values are correlated to weights, v_i = w_i +100
1: Uncorrelated with Similar weights
    - 5 items per city
    - weight range between 1000 and 1010
2: Uncorrelated
    - 10 items per city

```sh
python main.py argv1 argv2
```

Sample example: For 280 cities and Uncorrelated with Similar weights 

```sh
python main.py 280 1
```