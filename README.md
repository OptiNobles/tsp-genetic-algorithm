# tsp-genetic-algorithm

Genetic algorithm optimizer designed to solve Travelling Salesman Problem (TSP). 

# DEMO 

* Input should be in the form of a list [ [x_1, y_1], [x_2, y_2], ... , [x_n, y_n] ]
```python
from random import randint

towns = []

for town in range(60):
    towns.append([randint(0, 100), randint(0, 100)])
```
* How to initialize 
```python
from GeneticAlgorithm import GeneticAlgorithm

GA_optimizer = GeneticAlgorithm(elitism=50, mutation=0.01, fitness_function_degree=1.0)
GA_optimizer.fit(towns, popsize=500, gens=500, mode='tournament')
```

* Plotting results 
```python
GA_optimizer.plot()
```
![output_plot_GAO](https://user-images.githubusercontent.com/114445740/193552513-93860d5f-f650-409d-be28-671157f81a87.png)


* Visualization 
```python
GA_optimizer.show_graph(fitted=True)
```
![output_graph_GAO](https://user-images.githubusercontent.com/114445740/193552547-e637b143-7da3-46c1-a0d3-727ba5510997.png)
