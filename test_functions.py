# test_functions.py
# Contains definitions of various 2-input functions that can be used to test optimisers
import numpy as np


class TestFunction:
    def __init__(self):
        """ Generic parent class for test functions. 
        
        Each should define __call__() to get values, a range over which it should be plotted, and the global minima (as a list).
        The range is also used to determine the starting points.
        """
        raise NotImplementedError
        self.range = [-5, 5]
        self.global_minima = [[0, 0]]
    
    def __call__(self, x: float, y: float) -> float:
        """ Expected to return a scalar. """
        raise NotImplementedError
        

class Sphere(TestFunction):
    def __init__(self):
        """ Sphere function z = x**2 (y is ignored). """
        self.range = [-5, 5]
        self.global_minima = [[0, 0]]


    def __call__(self, x: float, y: float) -> float:
        return x**2 + y**2


class Beale(TestFunction):
    def __init__(self):
        """ Beale function. """
        self.range = [-4.5, 4.5]
        self.global_minima = [[3, 0.5]]


    def __call__(self, x: float, y: float) -> float:
        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


class GoldsteinPrice(TestFunction):
    def __init__(self):
        """ Goldstein-Price function. """
        self.range = [-2, 2]
        self.global_minima = [[0, -1]]


    def __call__(self, x: float, y: float) -> float:
        a = (1 + (x + y + 1)**2*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
        b = (30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
        return a*b


class Booth(TestFunction):
    def __init__(self):
        """ Booth function. """
        self.range = [-10, 10]
        self.global_minima = [[1, 3]]


    def __call__(self, x: float, y: float) -> float:
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2


class Himmelblau(TestFunction):
    def __init__(self):
        """ Himmelblau's Function. """
        self.range = [-5, 5]
        self.global_minima = [
            [3, 2],
            [-2.8051118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126]
        ]


    def __call__(self, x: float, y: float) -> float:
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    

class ThreeHumpCamel(TestFunction):
    def __init__(self):
        """ Three-Hump Camel Function. """
        self.range = [-5, 5]
        self.global_minima = [[0, 0]]


    def __call__(self, x: float, y: float) -> float:
        return 2*x**2 - 1.05*x**4 + (x**6 / 6) + x*y + y**2
    

class Rastrigin(TestFunction):
    def __init__(self):
        """ 2D Rastrigin Function. """
        self.range = [-5.12, 5/12]
        self.global_minima = [[0, 0]]


    def __call__(self, x: float, y: float) -> float:
        return 20 + (x**2 - 2 * np.cos(2 * np.pi * x**2)) + (2 * np.cos(2 * np.pi * y**2))
    

class Rosenbrock(TestFunction):
    def __init__(self):
        """ 2D Rosenbrock Function. """
        self.range = [-10, 10]  # Infinite but some bounds make it plottable.
        self.global_minima = [[1, 1]]


    def __call__(self, x: float, y: float) -> float:
        return 100 * (y - x**2)**2 + (1-x)**2
    

class Ackley(TestFunction):
    def __init__(self):
        """ Ackley Function. """
        self.range = [-5, 5]
        self.global_minima = [[0, 0]]


    def __call__(self, x: float, y: float) -> float:
	    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20