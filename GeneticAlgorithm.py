import numpy as np
from random import sample, random, randint, shuffle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams['figure.facecolor'] = 'dimgrey'
import time
from itertools import combinations

class Node:
    def __init__(self, x, y):
        """
        A Node class used in GAO.
        """
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def distance_to(self, other_node):
        """
        Evaluates distance between this node and other node.
        """
        distance = np.hypot(self.x - other_node.x, self.y - other_node.y)
        return distance

class Path:
    def __init__(self, nodes):

        self.path = nodes + [nodes[0]]
        self.length = self.__evaluate_length()

    def __copy__(self):
        return type(self)(self.path[:-1])

    def __str__(self):

        string = ''
        counter = 1

        for node in self.path:
            string += str(node)
            if counter != len(self.path):
                string += ' --> '
            counter += 1
        return string

    def __evaluate_length(self):

        length = 0

        for i in range(len(self.path) - 1):
            length += self.path[i].distance_to(self.path[i+1])

        return length

    def __update_length(self):
        self.length = self.__evaluate_length()

    def swap_nodes(self, node, other_node):
        
        node_one = self.path[node]
        node_two = self.path[other_node]

        self.path[node] = node_two
        self.path[other_node] = node_one

        self.__update_length()

    def copy(self):
        return self.__copy__()

class GeneticAlgorithm:
    def __init__(self, elitism=20, mutation=0.01, fitness_function_degree=1.0):

        """
        Genetic Algorithm Optimizer. Finds path that minimizes distance travelled between nodes.
        This optimizer is devoted to solving Traveling Salesman Problem (TSP)

        :param elitism: Number of fittest individuals that will be carried out to the next generation.
        :param mutation: Mutation rate.
        :param fitness_function_degree: Exponent for the fitness function (optional).

        Created by Radosław Sergiusz Jasiewicz. Enjoy :)
        """
        
        # List and number of nodes
        self.nodes = []
        self.num_nodes = None

        # Population
        self.population = []

        # Parameters
        self.elitism = elitism
        self.mutation = mutation
        self.fitness_function_degree = fitness_function_degree

        # Number of generations
        self.generations = None

        # Internal statistics
        self.best_score = 0
        self.best = None
        self.series = []

        # Optimizer's status
        self.fit_time = 0
        self.fitted = False
        self.mode = None

    def __str__(self):

        string = "Genetic Algorithm Optimizer"
        string += "\n------------------------"
        string += "\nDesigned to solve travelling salesman problem. Optimizes the minimum distance travelled."
        string += "\n------------------------"
        string += f"\nNumber of elite individuals:\t\t{self.elitism}"
        string += f"\nMutation rate:\t\t\t\t{self.mutation}"
        string += f"\nFitness function degree:\t\t{self.fitness_function_degree}"
        string += "\n------------------------"

        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string

    def __initialize(self, nodes, popsize, gens, mode):
        """
        Creates initial population of individuals. Each individual consists of nodes and has it's own length.
        """
        for node in nodes:
            assert len(node) == 2, "These are NOT valid nodes."

        assert len(nodes) > 1, "Path has been optimized. Best score: 0 | 0.0 s"

        if self.fitness_function_degree < 0: 
            self.fitness_function_degree = 1.0
            print(f"Fitness function degree < 0 encountered! Setting it to default value of 1.0...")

        self.num_nodes = len(nodes)
        self.generations = gens + 1
        self.mode = mode

        # Create nodes
        for x, y in zip(nodes[:, 0], nodes[:, 1]):
            self.nodes.append(Node(x, y))

        # Create initial population
        for _ in range(popsize):
            _nodes = sample(self.nodes, self.num_nodes)
            self.population.append(Path(_nodes))

    def __fitness(self, individual):
        """
        Evaluates fitness of an individual.

        :param individual: An individual from the population.
        """
        return (1/individual.length)**self.fitness_function_degree

    def __rank_population(self):
        """
        Ranks population based on fitness value.

        :return: Population ranked by fitness value
        """
        return sorted(self.population, key=self.__fitness, reverse=True)

    def __evaluate_selection_probability(self):
        """
        Evaluates the fitness of each individual relative to the population. It is used to assign a probability of selection.

        :return: List of probabilities of choosing individuals from the population that are carried to selection stage.
        """
        fitness = np.zeros(len(self.population))

        for individual in range(len(self.population)):
            fitness[individual] = self.__fitness(self.population[individual])

        cumulative_sum = np.cumsum(fitness)
        probability = 100 * cumulative_sum / np.sum(fitness)

        return probability

    def __proportionate_selection(self):
        """
        The fitness of each individual relative to the population is used to assign a probability of selection.

        :return: List of individuals that have passed selection stage.
        """
        ranked_population = self.__rank_population()
        probability = self.__evaluate_selection_probability()

        selection_result = []

        # Pick best performing individuals
        for individual in range(self.elitism):
            selection_result.append(ranked_population[individual])
        
        # Select individuals based on fitness value
        for i in range(len(ranked_population) - self.elitism):
            pick_path = 100*random()
            for individual in range(len(ranked_population)):
                if pick_path <= probability[individual]:
                    selection_result.append(ranked_population[individual])
                    break

        return selection_result

    def __tournament_selection(self):
        """
        A pair of individuals are randomly selected from the population and the one with the highest fitness is chosen.

        :return: List of individuals that have passed selection stage.
        """
        ranked_population = self.__rank_population()
        
        selection_result = []

        selection_stop = len(self.population) - self.elitism
        selection_counter =  0

        # Pick best performing individuals
        for i in range(self.elitism):
            selection_result.append(ranked_population[i])

        # Select individuals in tournament
        for path, other_path in combinations(self.population, r=2):

            if self.__fitness(path) > self.__fitness(other_path):
                selection_result.append(path)
            elif self.__fitness(path) < self.__fitness(other_path):
                selection_result.append(other_path)
            else:
                if random() < 0.5:
                    selection_result.append(path)
                else:
                    selection_result.append(other_path)
            selection_counter += 1

            if selection_counter == selection_stop:
                break

        return selection_result

    def __crossover(self, first_parent, second_parent):
        """
        Ordered crossover. Randomly select a subset of the first parent genes and then fill the remainder with the genes from the second parent in the order in which they appear, 
        without duplicating any genes in the selected subset from the first parent.

        :param first_parent: First individual whose genes will take part in a crossover to produce an offspring.
        :param second_parent: Second individual whose genes will take part in a crossover to produce an offspring.
        :return: Offspring.
        """
        offspring = []

        gene_A = randint(0, len(first_parent.path))
        gene_B = randint(0, len(first_parent.path))

        start_cut = min(gene_A, gene_B)
        end_cut = max(gene_A, gene_B)

        for gene in first_parent.path[start_cut:end_cut]:
            offspring.append(gene)

        for gene in second_parent.path:
            if gene not in offspring:
                offspring.append(gene)

        return Path(offspring)

    def __crossover_population(self, mating_pool):
        """
        Function caryying crossover over the entire selected population.

        :param mating_pool: Population selected for breeding.
        :return: Population for the next generation.
        """
        offsprings = []
        length = len(mating_pool) - self.elitism

        for i in range(self.elitism):
            offsprings.append(mating_pool[i])

        shuffle(mating_pool)

        for i in range(length):
            offspring = self.__crossover(mating_pool[i], mating_pool[len(mating_pool) - i - 1])
            offsprings.append(offspring)

        return offsprings

    def __mutate(self, individual):
        """
        Swap mutation. With specified low probability (mutation rate), two nodes will swap places in our individual. Swap does not include first and last node.

        :param individual: An individual to mutate.
        :return: Mutated individual.
        """
        for node in range(1, len(individual.path) - 2):
            if random() < self.mutation:
                other_node = randint(1, len(individual.path) - 2)
                individual.swap_nodes(node, other_node)
        return individual

    def __mutate_population(self, offsprings):
        """
        Function carrying out mutation over the entire selected population.
        """
        for offspring in range(len(offsprings)):
            offsprings[offspring] = self.__mutate(offsprings[offspring])
        return offsprings

    def __evaluate(self, next_generation):
        """
        Each generation the best individual needs to be found. This function does exactly that.

        :return: Best individual's score (path length) and path.
        """
        best_score = 0
        best_path = None

        for individual in range(len(next_generation)):
            path = next_generation[individual]
            score = next_generation[individual].length

            if individual == 0:
                best_score = score
                best_path = path

            elif score < best_score:
                best_score = score
                best_path = path

        return best_path, best_score

    def fit(self, nodes, popsize=50, gens=500, mode='proportionate', verbose=True, decimal=2):
        """
        Core function of the optimizer. It fits GAO to a specific list of nodes. 

        :param nodes: List of positions (x, y) of the nodes.
        :param popsize: Size of the initial population.
        :param gens: Number of generations.
        :param selection: Type of used selection method - proportionate or tournament.
        :param verbose: If enabled GAO informs you about the progress.
        :param decimal: Number of decimal places. 
        """
        self.__initialize(np.array(nodes), popsize, gens, mode) ; start = time.time()

        if verbose: print(f"{self.num_nodes} nodes were given. Beggining GAO optimization with {gens} generations...\n")

        for gen in range(gens+1):
            start_gen = time.time()

            if mode == 'proportionate': 
                selection_result = self.__proportionate_selection()
            elif mode == 'tournament':
                selection_result = self.__tournament_selection()

            offspring = self.__crossover_population(selection_result)
            next_generation = self.__mutate_population(offspring)
            self.population = next_generation

            best_path, best_score = self.__evaluate(next_generation)
            self.series.append(best_score)

            if gen == 0:
                self.best = best_path.copy()
                self.best_score = best_score

            elif self.best_score > best_score:
                self.best = best_path.copy()
                self.best_score = best_score

            if verbose: print(f"Generation {gen}/{gens} | Score: {round(best_score, decimal)} | Best: {round(self.best_score, decimal)} | {round(time.time() - start_gen, decimal)} s")

        self.fit_time = round(time.time() - start)
        self.fitted = True

        if verbose: print(f"\nGA fitted. Runtime: {self.fit_time // 60} minute(s) | Best score: {round(self.best_score, decimal)} | Mode: {self.mode}\n")
        if verbose: print(f"Best path:\n {self.best}")

    def plot(self, figsize=(10,5), dpi=200):
        """
        Plots performance over generations. If GAO has NOT been fitted returns None.
        """
        if not self.fitted:
            print("Genetic Algorithm Optimizer NOT fitted. There is nothing to plot.")
            return None
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax.plot(
                self.series,
                lw = 0.75,
                color = "darkorange",
            )

            ax.set_xlim(0, self.generations - 1)
            ax.set_xlabel(r'$\bf{Generation}$')
            ax.set_ylabel(r'$\bf{Performance}$')
            plt.show()

    def show_graph(self, fitted=True, figsize=(10,5), dpi=200):
        """
        Shows graph of nodes that GAO is working on.

        :param fitted: If True it shows the best path that the optimizer has found.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)

        if self.fitted and fitted:
            for i in range(len(self.best.path)-1):
                ax.plot(
                    [self.best.path[i].x, self.best.path[i+1].x],
                    [self.best.path[i].y, self.best.path[i+1].y],
                    color = "orange",
                    linewidth = 0.5,
                    zorder=1
                )

        for node in self.nodes:
            ax.scatter(
                x = node.x,
                y = node.y,
                linewidth=0.5,
                marker="o",
                s=8,
                edgecolor="orange",
                c="black",
                zorder=2
            )

        for node, other_node in combinations(self.nodes, r=2):
            ax.plot(
                [node.x, other_node.x],
                [node.y, other_node.y],
                color="blue",
                linewidth=0.2,
                alpha=0.1,
                zorder=0
            )

        if not self.fitted and fitted:
            print("GAO NOT fitted. There is no path to show.")

        ax.set_title(r"$\bf{Graph}$ $\bf{of}$ $\bf{nodes}$")
        ax.axis('off')
        plt.show()

    def get_result(self):
        """
        :return: Tuple consisted of best path, best distance, fit time and list of each generation's best distance.
        """
        return self.best, self.best_score, self.fit_time, self.series 