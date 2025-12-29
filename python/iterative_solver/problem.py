import numpy as np
import sys

from .pspace import PSpace


class Problem:
    '''
    Semi-abstract base class for specifying the problem to be solved by iterative_solver.
    '''

    def __init__(self):
        self.p_space = PSpace()
        self.size = None

    def residual(self, parameters, residual):
        """
        Calculate the residual vector. Used by non-linear solvers (NonLinearEquations,  Optimize) only.

        :return: In the case where the residual is an exact differential, the corresponding function value. Used by Optimize but not NonLinearEquations.
        :rtype: float
        :param parameters: The trial solution for which the residual is to be calculated
        :type parameters: np.ndarray(dtype=float)
        :param residual:  The residual vector
        :type residual: np.ndarray(dtype=float)
        """
        raise NotImplementedError

    def action(self, parameters, action):
        """
        Calculate the action of the kernel matrix on a set of parameters. Used by
        linear solvers, but not by the non-linear solvers (NonLinearEquations, Optimize).

        :param parameters: The trial solutions for which the action is to be calculated
        :type parameters: np.ndarray(dtype=float)
        :param action: The action vectors
        :type action: np.ndarray(dtype=float)
        """
        raise NotImplementedError

    def diagonals(self, diagonals):
        """
        Optionally provide the diagonal elements of the underlying kernel. If
        implemented and returning true, the provided diagonals will be used by
        IterativeSolver for preconditioning (and therefore the precondition() function does
        not need to be implemented), and, in the case of linear problems, for selection of
        the P space. Otherwise, preconditioning will be done with precondition(), and any P
        space has to be provided manually.

        :param diagonals: On exit, contains the diagonal elements
        :type diagonals: np.ndarray(dtype=bool, ndim=1)
        :return: Whether diagonals have been provided.
        :rtype: bool
        """
        return False

    def precondition(self, residual, shift=None, diagonals=None):
        """
        Apply preconditioning to a residual vector in order to predict a step towards
        the solution.

        :param residual: On entry, assumed to be the residual. On exit, the negative of the predicted step.
        :type residual: np.ndarray(dtype=float)
        :param shift: When called from LinearEigensystem, contains the corresponding current
           eigenvalue estimates for each of the parameter vectors in the set. All other solvers
           pass a vector of zeroes.
        :type shift: float
        :param diagonals: Diagonal elements of kernel
        :type diagonals: np.ndarray(dtype=float, ndim=1)
        """
        if diagonals is None:
            raise NotImplementedError
        small = 1e-14
        if shift is None:
            residual /= (diagonals + small)
        else:
            if len(residual.shape) == 1:
                residual /= (diagonals - (shift[0] if isinstance(shift, np.ndarray) else shift) + small)
            else:
                residual /= (diagonals - np.asanyarray(shift)[:, np.newaxis] + small)
    def pp_action_matrix(self):
        """
        Calculate the representation of the kernel matrix in the P space. Implementation required only for linear hermitian problems for which P-space acceleration is wanted.
        """
        if self.p_space.size > 0:
            raise NotImplementedError('P-space unavailable: unimplemented pp_action_matrix() in Problem class')
        return np.array([], dtype=np.double)

    def p_action(self, p_coefficients, actions, ranges):
        """
        Calculate the action of the kernel matrix on a set of vectors in the P space. Implementation required only for linear hermitian problems for which P-space acceleration is wanted.

        :param p_coefficients The projection of the vectors onto to the P space
        :param actions On exit, the computed action has been added to the original contents
        :param range The range of the full space for which actions should be computed.
        """
        if self.p_space.size > 0:
            raise NotImplementedError('P-space unavailable: unimplemented p_action() in Problem class')

    def test_parameters(self, instance: int, parameters) -> bool:
        """
        Provide parameters for testing purposes.

       :param instance: Which distinct parameter set is being requested. The function should expect to be called with values of instance in the sequence 0,1,2,...
       :param parameters: The parameters.

        :return: Whether parameters for this instance are available.
        """
        return False

    def report(self, iteration, verbosity, errors, value=None, eigenvalues=None):
        if (iteration <= 0 and verbosity >= 1) or verbosity >= 2:
            if iteration > 0 and verbosity >= 2:
                print('Iteration', iteration, 'log10(|residual|)=', np.log10(errors))
            elif iteration == 0:
                print('Converged', 'log10(|residual|)=', np.log10(errors + sys.float_info.min))
            else:
                print('Unconverged', 'log10(|residual|)=', np.log10(errors + sys.float_info.min))
            if value is not None:
                print('Objective function value', value)
            if eigenvalues is not None:
                print('Eigenvalues', eigenvalues)
            return True
        else:
            return False

    def test(self, **kwargs):
        if hasattr(self,'size'):
            parameters = np.zeros(self.size)
            for i in range(self.size):
                self.test_parameters(i, parameters)
                test_problem_class(self, parameters, **kwargs)


def test_problem_class(problem_class_instance: Problem, parameters: np.ndarray=None, step=1e-4,
                       tolerance=1e-8, verbosity=0, test_hessian=True) -> float:
    r"""
    Test the analytical derivatives of a problem class instance by numerical differentiation, in the case that the residual is meant to be the derivative of a scalar function.

    :param problem_class_instance: An instance of iterative_solver.Problem.
    :param parameters: Values of the parameters to be tested.
    :param step: Numerical differentiation step size.
    :param tolerance: An exception is raised if the numerical derivatives differ from the analytical ones by more than this tolerance.
    :param verbosity: How much output to produce.
    :param test_hessian:  Whether to test the analytical hessian as well. Ignored if the problem class instance does not contain a method `hessian`.
    :return: The norm of the difference between the numerical and analytical derivatives.
    """
    if parameters is None:
       parameters = np.zeros(problem_class_instance.size)
       if not problem_class_instance.test_parameters(0, parameters): return None
    residual_analytic = np.zeros(len(parameters))
    residual_numerical = np.zeros(len(parameters))
    test_hessian &= hasattr(problem_class_instance, 'hessian')
    if (test_hessian):
        hessian_numerical = np.zeros((len(parameters), len(parameters)))
    residualp1 = np.zeros(len(parameters))
    residualp2 = np.zeros(len(parameters))
    residualm1 = np.zeros(len(parameters))
    residualm2 = np.zeros(len(parameters))
    value0 = problem_class_instance.residual(parameters, residual_analytic)
    for i in range(len(parameters)):
        parameters[i] -= 2 * step
        valuesm2 = problem_class_instance.residual(parameters, residualm2)
        parameters[i] += step
        valuesm1 = problem_class_instance.residual(parameters, residualm1)
        parameters[i] += 2 * step
        valuesp1 = problem_class_instance.residual(parameters, residualp1)
        parameters[i] += step
        valuesp2 = problem_class_instance.residual(parameters, residualp2)
        parameters[i] -= 2 * step
        residual_numerical[i] = (valuesm2 - 8 * valuesm1 + 8 * valuesp1 - valuesp2) / (12 * step)
        if test_hessian:
            hessian_numerical[:, i] = (residualm2 - 8 * residualm1 + 8 * residualp1 - residualp2) / (12 * step)
    if verbosity > 1: print('test_problem_class', parameters, residual_analytic, residual_numerical)
    error = np.linalg.norm(residual_numerical - residual_analytic)
    if test_hessian:
        error += np.linalg.norm(hessian_numerical - problem_class_instance.hessian(parameters))
        if verbosity > 2:
            print('test_problem_class numerical hessian')
            print(hessian_numerical)
            print('test_problem_class analytical hessian')
            print(problem_class_instance.hessian(parameters))
    if verbosity > 0: print('test_problem_class', type(problem_class_instance).__name__,  error)
    if error > tolerance:
        raise ValueError(f'test_problem_class {type(problem_class_instance).__name__} error {error}')
    return float(error)
