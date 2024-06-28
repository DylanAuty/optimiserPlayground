# test_functions.py
# Contains definitions of various 2-input functions that can be used to test optimisers

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