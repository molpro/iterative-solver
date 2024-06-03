import sys

import numpy as np


class Problem:
    '''
    Semi-abstract base class for specifying the problem to be solved by iterative_solver.
    '''
    def __init__(self):
        self.dimension = None
    def residual(self, parameters, residual):
        raise NotImplementedError

    def action(self, parameters, action):
        raise NotImplementedError

    def diagonals(self, diagonals):
        return False

    def precondition(self, residual, shift=None, diagonals=None):
        small = 1e-14
        if len(residual.shape) > 1:
            for i in range(residual.shape[0]):
                assert type(shift) == np.ndarray
                self.precondition(residual[i, :], float(shift[i]) if shift is not None else None, diagonals)
            return
        if diagonals is not None:
            if shift is not None:
                for j in range(residual.size):
                    residual[j] = residual[j] / (diagonals[j] + shift + small)
            else:
                for j in range(residual.size):
                    residual[j] = residual[j] / (diagonals[j] + small)
        else:
            raise NotImplementedError

    def pp_action_matrix(self, pparams):
        return np.array([], dtype=np.double)

    def p_action(self, p_coefficients, pparams, actions):
        raise NotImplementedError('P-space unavailable: unimplemented p_action() in Problem class')
        pass

    def test_parameters(self, instance, parameters):
        return False

    def report(self, iteration, verbosity, errors, value=None, eigenvalues=None):
        # print('in report')
        # print('report errors',errors)
        if (iteration <= 0 and verbosity >= 1) or verbosity >= 2:
            if iteration > 0 and verbosity >= 2:
                print('Iteration', iteration, 'log10(|residual|)=', np.log10(errors))
            elif iteration == 0:
                print('Converged', 'log10(|residual|)=', np.log10(errors+sys.float_info.min))
            else:
                print('Unconverged', 'log10(|residual|)=', np.log10(errors+sys.float_info.min))
            if value is not None:
                print('Objective function value', value)
            if eigenvalues is not None:
                print('Eigenvalues', eigenvalues)
            return True
        else:
            return False
