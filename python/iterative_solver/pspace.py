import numpy as np


class PSpace:
    '''
    Container for the definition of the P space in Iterative_Solver
    '''

    def __init__(self):
        self.size = 0
        self.offsets=[0]
        self.indices=[]
        self.coefficients=np.ndarray(0)
        self.simple = True
        pass

    def add_complex(self, indices, coefficients):
        self.simple = self.simple and len(indices) <=1
        self.offsets.append(len(self.coefficients)+len(coefficients))
        self.indices.append(indices)
        self.coefficients.append(coefficients)
        self.size = self.offsets[-1]

    def add_simple(self, indices):
        for i in indices:
            self.add_complex([i],[1.0])
