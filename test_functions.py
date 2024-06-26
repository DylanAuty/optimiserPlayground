# test_functions.py
# Contains definitions of various 2-input functions that can be used to test optimisers

def sphere(x, y):
    """ Sphere function z = x**2 (y is ignored). """
    return x**2 + y**2


def beale(x, y):
    """ Beale function. """
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def goldstein_price(x, y):
    """ Goldstein-Price function. """
    a = (1 + (x + y + 1)**2*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    b = (30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return a*b


def booth(x, y):
    """ Booth function """
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2


def himmelblau(x, y):
    """ Himmelblau's Function """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2