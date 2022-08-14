
import numpy as np

class RLS:
    """
    `beta` is recursively updated
    """
    def __init__(self, Xinit, Yinit) -> None:
        self.P = np.linalg.inv(Xinit.T @ Xinit)
        self.beta = self.P @ Xinit.T @ Yinit
    
    def add_new(self, x, y):

        
        z = x.reshape((x.shape[0], 1))
        
        np.seterr(over="raise")
        try:
            G = (self.P @ z) / (1 + z.T @ self.P @ z)
            self.P = self.P - ((G @ z.T) * self.P)
            self.beta = self.beta + G * (y - self.beta.T @ z)
        except FloatingPointError as e:
            print("[ERROR] FloatingPointError occurred!")
            print(e)
            print(f"{self.P=}")
            print(f"{z=}")
            print(f"{self.beta=}")
            raise FloatingPointError

        return

    def evaluate(self, x):
        x_eval = np.concatenate([x[1:], np.array(x[0]).reshape(1)], axis=0)
        return x_eval.dot(self.beta)