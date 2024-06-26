# optim.py
# Contains definitions for different optimizers

import numpy as np

class Optim():
    def __init__(self):
        """ Generic optimiser parent class. """
        raise NotImplementedError


    def step(self):
        raise NotImplementedError
    

    def get_xy_lists(self) -> tuple[list, list]:
        """ Returns tuple of (x_coords, y_coords) from opt.history """
        self.pos_history = np.array(self.pos_history)
        return self.pos_history[:, 0], self.pos_history[:, 1]


class SGD(Optim):
    def __init__(self, lr: float = 1e-5, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements SGD. """
        self.lr = lr
        self.pos = np.asarray(pos, dtype=np.float64)
        self.pos_history = []
        self.pos_history.append(self.pos)
        self.grad = np.asarray([0, 0], dtype=np.float64)


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        self.pos = self.pos - self.lr * self.grad
        self.pos_history.append(self.pos)


class Momentum(Optim):
    def __init__(self, lr: float = 1e-5, decay: float = 1e-3, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements SGD with momentum. """
        self.lr = lr
        self.decay = decay

        self.pos = np.asarray(pos, dtype=np.float64)
        self.pos_history = []
        self.pos_history.append(self.pos)
        self.grad = np.asarray([0, 0], dtype=np.float64)
        self.prev_delta = 0


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        delta = self.lr * self.grad + self.prev_delta * self.decay
        self.pos = self.pos - delta
        self.prev_delta = delta
        self.pos_history.append(self.pos)


# class AdaGrad(Optim):
#     def __init__(self, lr: float = 1e-5, pos: np.ndarray | list = np.asarray([2, 3])):
#         """ Implements AdaGrad. """
#         self.lr = lr
#         self.pos = np.asarray(pos, dtype=np.float64)
#         self.pos_history = []
#         self.pos_history.append(self.pos)
#         self.grad = np.asarray([0, 0], dtype=np.float64)


#     def step(self, grad: np.ndarray | list):
#         self.grad = np.asarray(grad)
#         self.pos = self.pos - self.lr * self.grad
#         self.pos_history.append(self.pos)
