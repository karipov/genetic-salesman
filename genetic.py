import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from IPython.display import clear_output, display


class Points:
    def __init__(self, grid=(100, 100), size=10):
        """
        :param grid: tuple that defines the map size
        :paran size: number of points to be generated
        """
        # generate random points
        self.x = np.random.randint(grid[0], size=size)
        self.y = np.random.randint(grid[1], size=size)

        # turn into a 2d array
        self.points = self._generate_2d_points(self.x, self.y)


    def _generate_2d_points(self, x, y):
        """
        :param x: all the x coordinates
        :param y: all the y cooridnates
        :return: 2D list with each element representing (x, y) coordinate
        """
        return copy.deepcopy(np.array(list(zip(x, y))))




class Route:
    def __init__(self,
                 points: list,
                 shuffle: bool = True,
                 ):
        """
        :param points: list where each element represents (x, y) coordinate
        :param random: whether points are supplied or should be generated
        randomly
        """
        self.points = copy.deepcopy(points)

        if shuffle:
            np.random.shuffle(self.points)

        self.path = self._define_path(self.points)
        self.distance = self._calc_distance(self.path)


    def __repr__(self) -> str:
        return "Route obj. - dist.: ~{:0.2f}\n".format(self.distance)


    def _define_path(self, points: list) -> list:
        """
        :param points: list of points with each element as (x, y) of coordinate
        :return: 3D list of connected points.
        """
        # completing a full path by adding the first point to the end
        points = np.vstack((points, points[0]))
        path = np.array(list(zip(points, points[1:])))

        return path


    def _calc_distance(self, path: list) -> float:
        """
        :param path: 3D list  (2, 2, x) of connected points
        :return: float, indicates the total Euclidean distance between
        all the points in the path
        """
        distance = 0

        for p0, p1 in path:
            distance += math.hypot(p1[0] - p0[0], p1[1] - p0[1])

        return distance


    def mutate(self):
        """
        Mutation is done in place.
        """
        i1 = np.random.randint(0, len(self.points))
        i2 = np.random.randint(0, len(self.points))

        self.points[[i1, i2]] = self.points[[i2, i1]]

        # other parameters are updated
        self.path = self._define_path(copy.deepcopy(self.points))
        self.distance = self._calc_distance(copy.deepcopy(self.path))


    @staticmethod
    def crossover(a, b):
        """
        Adapted from https://stackoverflow.com/a/55426480/8814732 (Paul Panzer)
        Both arrays must have a full intersection (all elements are the same)
        :param a: first parent path
        :param b: second parent path
        :return: child path
        """
        # index shouldn't be 0, otherwise it's just a copy of one parent
        i = np.random.randint(1, len(a))

        sa, sb = map(np.lexsort, (a.T, b.T))
        mb = np.empty(len(a), '?')
        mb[sb] = np.arange(2, dtype='?').repeat((i, len(a)-i))[sa]

        # Fill the b part of c, using mask
        return copy.deepcopy(np.concatenate([a[:i], b[mb]], 0))



class Population:
    def __init__(self,
                 points: list,
                 mutation: float = 0.3,
                 population_size: int = 500):
        """
        :param points: list where each element represents (x, y) coordinates
        :param mutation: chance of path to mutate
        :param population_size: number of paths to have in a population
        """
        self.ARGS = {
            "size": population_size
        }
        # must be at 3 least paths
        if self.ARGS["size"] < 3:
             self.ARGS["size"] = 3

        self.points = copy.deepcopy(points)
        self.x, self.y = list(zip(*self.points))

        self.population = self._initial_population()
        self.fittest = None


    def _initial_population(self):
        """
        Creates an initial population made of Routes
        :return: list of Route objects
        """
        population = [Route(self.points) for _ in range(self.ARGS["size"])]

        return np.array(population)


    def selection(self, survival_size=0.5, weight=np.e):
        """
        Selection is done in place
        :param survival_size: percent of population that survives
        - np.e is used as the power for bias weights
        - 50% sample size is hardcoded
        """
        self.population = np.array(
            sorted(self.population, key=lambda member: member.distance)
        )

        bias_weights = np.array(list(reversed(
            [(x / self.ARGS["size"])**weight for x in range(self.ARGS["size"])]
        )))
        # normalization so probabilities add up to 1
        probabilities = bias_weights / np.sum(bias_weights)
        sample_size = int(self.ARGS["size"] * survival_size)

        indices = np.random.choice(
            self.ARGS["size"], size=sample_size, replace=False, p=probabilities
        )

        new_population = copy.deepcopy(
            [self.population[i] for i in indices]
        )

        self.population = sorted(
            new_population, key=lambda member: member.distance
            )


    def mutate(self, chance=0.35):
        """
        Mutation is done in place.
        :param chance: the chance that a mutation occurs in the path
        - mutation chance works best at ~0.3, due to only a single
        swap of element position
        """
        for i in range(len(self.population)):
            if np.random.random() < chance:
                self.population[i].mutate()


    def crossover(self):
        """
        Creates new points from two parents to fill up the population back
        to full size.
        """
        while len(self.population) < self.ARGS["size"]:
            i1, i2 = np.random.choice(
                len(self.population), size=2, replace=False
                )

            child_points = Route.crossover(
                self.population[i1].points,
                self.population[i2].points
                )

            self.population.append(Route(points=child_points, shuffle=False))


    def plot(self, i, animate=False, jupyter=False):
        """
        Plots the points and the fittest path for current generation
        :param i: iteration; used for on screen graphs
        :param animate: whether the plot is updated manually or automatically
        """
        gen_fittest = min(self.population, key=lambda route: route.distance)

        # check if the new path is the fittest of all population
        if self.fittest == None:
            self.fittest = gen_fittest
        elif gen_fittest.distance < self.fittest.distance:
            self.fittest = gen_fittest

        # create side by side sublplots
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
        fig.tight_layout(pad=2)

        ax1.set_title("All time fittest", fontsize=14)
        ax2.set_title("Fittest of Generation {}".format(i), fontsize=14)

        # ensure last point is included in path (full circle)
        ax1_path_list = list(zip(*np.vstack(
            (self.fittest.points, self.fittest.points[0])
            )))
        ax1_label = "dist.: ~{:0.2f}".format(self.fittest.distance)
        ax1.scatter(self.x, self.y, facecolors='none', edgecolors="k")
        ax1.plot(*ax1_path_list, color="b", label=ax1_label)
        ax1.legend(loc=1, fontsize='x-small')
        ax1.set_aspect('equal')

        # ensure last point is included in path (full circle)
        ax2_path_list = list(zip(*np.vstack(
            (gen_fittest.points, gen_fittest.points[0])
            )))
        ax2_label = "dist.: ~{:0.2f}".format(gen_fittest.distance)
        ax2.scatter(self.x, self.y, facecolors='none', edgecolors="k")
        ax2.plot(*ax2_path_list, color="c", label=ax2_label)
        ax2.legend(loc=1, fontsize='x-small')
        ax2.set_aspect('equal')

        if animate:
            plt.draw()
            plt.pause(0.001)
            plt.close()
        elif jupyter:
            plt.show()
            clear_output(wait=True)
        else:
            # goes forward by clicking
            plt.show()



# -------- testing area --------

np.random.seed(42)

x = np.array(
    [[20, 40],
    [40, 20],
    [60, 20],
    [80, 40],
    [80, 60],
    [60, 80],
    [40, 80],
    [20, 60]])

city = Points(grid=(100, 100), size=15)

routes = Population(city.points)

for i in range(2000):
    routes.selection()
    routes.crossover()
    routes.mutate(chance=0.4)
    routes.plot(i, animate=True)

# settings to use for good results: 15-0.4, 20-0.4, 30-0.45
