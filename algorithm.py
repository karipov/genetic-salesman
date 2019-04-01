import numpy as np
print("numpy version", np.__version__)
import matplotlib.pyplot as plt

import copy
import math
import random

# from IPython.display import clear_output



class Field:
    def __init__(self,
                 grid_size=100,
                 num_cities=30,
                 pop_size=200,
                 crossover=0.05,
                 mutation=0.05,
                 max_path_length=10):
        self.grid_size = grid_size
        self.num_cities = num_cities
        self.cities = self._create_points()

        self.pop_size = pop_size
        self.crossover = crossover
        self.mutation = mutation

        self.max_path_length = max_path_length

        if self.num_cities < 3:
            self.num_cities = 3


    def _create_points(self):
        """
        :return: 2D list of points on a grid
        """
        x = np.random.randint(self.grid_size, size=self.num_cities)
        y = np.random.randint(self.grid_size, size=self.num_cities)

        return np.array(list(zip(x, y)))


    def _return_points(self, path):
        """
        :return: two arrays (x, y) that correspond to point
        axes
        """
        return zip(*path[0])


    def _create_path(self):
        """
        :return: list of points
        """
        points = copy.deepcopy(self.cities)
        np.random.shuffle(points)

        return points


    def _create_population(self):
        """
        :return: list of paths
        """
        population = []

        for _ in range(self.pop_size):
            member = self._create_path()
            population.append(member)

        # population with distance
        # distance_pop = self._fitness_population(population)
        # print(distance_pop)

        return population


    def _fitness_path(self, path):
        """
        :param path: list of points that represent a path
        :return: Euclidean distance between all the
        points in the path
        """
        distance = 0

        path_pairs = np.array(list(zip(path[0:], path[1:])))
        path_pairs = np.vstack((path_pairs, [[path[-1], path[0]]]))

        for p0, p1 in path_pairs:
            # print("p0", p0, "p1", p1)
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            distance += d

        return distance


    def re_fitness_path(self, fitness_path):
        """
        :param fitness_path: [[[12, 34], [56, 75]], 123.456]
        :return: correct distance on fitness path
        """
        correct_distance = 0
        points = fitness_path[0] # excluding the ending distance

        for p0, p1 in points:
            d = math.hypot(p1 - p0, p1 - p0)
            correct_distance += d

        return correct_distance


    def _fitness_population(self, population):
        """
        :param population: list of paths e.g. [[[12, 34], [56, 75]], [[89, 10], [13, 24]]]
        :return: list of paths with fitness score attached to each [[[[12, 34], [56, 75]], 123.456], [[[89, 10], [13, 24]], 456.778]]
        """
        fitness_population = []

        for member in population:
            distance = self._fitness_path(member)
            fitness_population.append([member, distance])

        fitness_population = sorted(fitness_population, key=lambda m: m[1])

        return np.array(fitness_population)


    def re_fitness_population(self, fitness_population):
        """
        :param fitness_population: e.g. [[[[12, 34], [56, 75]], 123.456], [[[89, 10], [13, 24]], 456.778]]
        :return: correctly fitnessed population
        """
        re_fitness_population = []

        for member in fitness_population:
            distance = self.re_fitness_path(member)
            re_fitness_population.append([member[0], distance])

        re_fitness_population = sorted(re_fitness_population, key=lambda m: m[1])

        return np.array(re_fitness_population)


    def population(self, fitness_population):
        """
        :param fitness_population: initial population
        :return: population halved with a skewed probability
        """
        bias_weights = np.array(list(reversed([(x / len(fitness_population))**np.e for x in range(len(fitness_population))])))
        probabilities = np.array(bias_weights / np.sum(bias_weights))
        sample_size = int(len(fitness_population)/2)

        choices = np.random.choice(len(fitness_population), size=sample_size, replace=False, p=probabilities)

        new_population = copy.deepcopy([fitness_population[i] for i in choices])
        new_population = np.array(sorted(new_population, key=lambda member: member[1]))

        return new_population


    def _mutate_path(self, path):
        """
        :param path: path to be mutated
        :return: mutated path at a random cross-section
        """
        a, b = np.random.randint(low=0, high=len(path[0]), size=2)
        path[0][a], path[0][b] = path[0][b], path[0][a]

        return path


    def _mutate_population(self, fitness_population):
        mutated_population = []

        for member in fitness_population:
            if np.random.random() < self.mutation:
                 mutated_population.append(self._mutate_path(member))
            else:
                mutated_population.append(member)

        return np.array(mutated_population)


    def _crossover_paths(self, path_a, path_b):
        i = np.random.randint(0, len(path_a))
        path_a[:i], path_b[:i] = path_b[:i].copy(), path_a[:i].copy()

        return random.choice([path_a, path_b])


    def crossover_population(self, fitness_population):
        new_population = []

        while len(new_population) < self.pop_size:
            index1 = np.random.randint(0, len(fitness_population))
            index2 = np.random.randint(0, len(fitness_population))

            child = self._crossover_paths(fitness_population[index1], fitness_population[index2])
            new_population.append(child)

        return np.array(new_population)


    def evolve(self, iterations=1000):
        population = self._create_population()
        population = self._fitness_population(population)

        # TODO: fix
        # self._create_population() produces a list without the distances
        # this cannot be used as the initial list.

        for i in range(iterations):
            population = self.re_fitness_population(population)
            # print(min(population, key=lambda p: p[1])[1])
            mini = min(population, key=lambda p: p[1])

            plt.plot(*list(self._return_points(mini)))
            plt.scatter(*zip(*self.cities))
            plt.draw()
            plt.pause(0.001)
            plt.clf()

            halved = self.population(population)
            mutated = self._mutate_population(halved)

            population = self.crossover_population(mutated)

l = Field()
l.evolve()
