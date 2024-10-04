import numpy as np


class PSpace:
    '''
    Container for the definition of the P space in Iterative_Solver
    '''

    def __init__(self):
        self.size = 0
        self.offsets = [0]
        self.indices = []
        self.coefficients = np.ndarray(0)
        self.simple = True
        pass

    def add_complex(self, indices, coefficients):
        self.simple = self.simple and len(indices) <= 1
        np.append(self.offsets, len(self.coefficients) + len(coefficients))
        np.append(self.indices, indices)
        np.append(self.coefficients, coefficients)
        self.size = self.offsets[-1]

    def add_simple(self, indices):
        for i in indices:
            self.add_complex([i], [1.0])
