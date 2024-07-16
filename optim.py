# optim.py
# Contains definitions for different optimizers

import numpy as np

class Optim():
    def __init__(self, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Generic optimiser parent class. """
        self.pos = np.asarray(pos, dtype=np.float64)
        self.pos_history = []
        self.loss_history = []
        self.pos_history.append(self.pos)
        self.grad = np.asarray([0, 0], dtype=np.float64)


    def step(self):
        raise NotImplementedError
    

    def get_xy_lists(self) -> tuple[list, list]:
        """ Returns tuple of (x_coords, y_coords) from opt.history """
        self.pos_history = np.array(self.pos_history)
        return self.pos_history[:, 0], self.pos_history[:, 1]


class SGD(Optim):
    def __init__(self, lr: float = 1e-3, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements SGD. 
        
        Finds the direction of steepest descent, and steps a bit in that direction.
        """
        super().__init__(pos=pos)
        self.lr = lr


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        self.pos = self.pos - self.lr * self.grad
        self.pos_history.append(self.pos)


class Momentum(Optim):
    def __init__(self, lr: float = 1e-3, decay: float = 1e-5, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements SGD with momentum. 

        Same as SGD, but also adds on the previous step (but with magnitude decayed).
        """
        super().__init__(pos=pos)
        self.lr = lr
        self.decay = decay
        self.prev_delta = 0


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        delta = self.lr * self.grad + self.prev_delta * self.decay
        self.pos = self.pos - delta
        self.prev_delta = delta
        self.pos_history.append(self.pos)


class NestorovMomentum(Optim):
    def __init__(self, lr: float = 1e-3, decay: float = 1e-5, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements SGD with Nestorov momentum. 

        Nestorov momentum is the same as regular momentum, but it evaluates the gradient after the update instead of before.
        """
        raise NotImplementedError   # Needs API change to know value of function so it can estimate grad again
        super().__init__(pos=pos)
        self.lr = lr
        self.decay = decay
        self.prev_delta = 0


    def step(self, grad: np.ndarray | list):
        raise NotImplementedError   # Needs API change to know value of function so it can estimate grad again
        self.grad = np.asarray(grad)
        delta = self.lr * self.grad + self.prev_delta * self.decay
        self.pos = self.pos - delta
        self.prev_delta = delta
        self.pos_history.append(self.pos)


class AdaGrad(Optim):
    def __init__(self, lr: float = 1e-2, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements AdaGrad. 
        
        Keeps track of (squared) gradients in an accumulator, and use that to emphasise stepping in directions that
        have not been well-explored. I.e., if a particular feature has been updated a lot already, it doesn't need
        to be updated as much.
        """
        super().__init__(pos=pos)
        self.lr = lr
        self.sq_grad_acc = np.asarray([0, 0], dtype=np.float64)


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        self.sq_grad_acc = self.sq_grad_acc + self.grad ** 2
        self.pos = self.pos - self.lr * (self.grad / np.sqrt(self.sq_grad_acc))
        self.pos_history.append(self.pos)


class AdaDelta(Optim):
    def __init__(self, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-06, decay:float = 0, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements AdaDelta. 
        
        Keeps track of the square of the change at each step and the square of the gradients.
        Delta is grad * RMS(decayed past deltas) / RMS(decayed past grads)
        """
        super().__init__(pos=pos)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.decay = decay
        self.sq_grad_acc_1 = np.asarray([0, 0], dtype=np.float64)
        self.sq_delta_acc = np.asarray([0, 0], dtype=np.float64)


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        self.grad = self.grad + self.decay * self.pos

        self.sq_grad_acc_1 = self.sq_grad_acc_1 * self.rho + (self.grad ** 2) * (1 - self.rho)
        delta = (np.sqrt(self.sq_delta_acc + self.eps) / (np.sqrt(self.sq_grad_acc_1 + self.eps))) * self.grad
        self.sq_delta_acc = self.sq_delta_acc * self.rho + (delta ** 2) * (1 - self.rho)
        self.pos = self.pos - self.lr * delta
        self.pos_history.append(self.pos)


class RMSProp(Optim):
    def __init__(self, lr: float = 1e-3, decay: float = 1e-5, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements RMSProp. 
        
        Like AdaGrad, except allows the gradient accumulator to decay (in an effort to speed things up).
        AdaGrad is slow since the updates are only made smaller more aggressively over time.
        """
        super().__init__(pos=pos)
        self.lr = lr
        self.decay = decay
        self.sq_grad_acc = np.asarray([0, 0], dtype=np.float64)


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        self.sq_grad_acc = self.sq_grad_acc * self.decay + (self.grad ** 2) * (1 - self.decay)
        self.pos = self.pos - self.lr * (self.grad / np.sqrt(self.sq_grad_acc))
        self.pos_history.append(self.pos)


class Adam(Optim):
    def __init__(self, lr: float = 1e-2, beta1: float = 0.9, beta2: float = 0.999, pos: np.ndarray | list = np.asarray([2, 3])):
        """ Implements Adam. 
        
        Momentum + RMSProp: keeps both grad and sq_grad accumulators, i.e. it does both momentum and RMSProp.
        """
        super().__init__(pos=pos)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_acc = np.asarray([0, 0], dtype=np.float64)
        self.sq_grad_acc = np.asarray([0, 0], dtype=np.float64)


    def step(self, grad: np.ndarray | list):
        self.grad = np.asarray(grad)
        self.grad_acc = self.grad_acc * self.beta1 + self.grad * (1 - self.beta1)
        self.sq_grad_acc = self.sq_grad_acc * self.beta2 + (self.grad ** 2) * (1 - self.beta2)
        self.pos = self.pos - self.lr * (self.grad_acc / np.sqrt(self.sq_grad_acc))
        self.pos_history.append(self.pos)
