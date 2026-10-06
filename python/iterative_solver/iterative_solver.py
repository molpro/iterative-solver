import numpy as np

from .problem import Problem
from .iterative_solver_extension import Optimize, LinearEigensystem, LinearEquations, NonLinearEquations


def Solve_Linear_Eigensystem(parameters, actions, problem: Problem, nroot=1, generate_initial_guess=True,
                             max_iter=1000000, max_p=0, **kwargs):
    solver = LinearEigensystem(parameters.shape[-1], nroot, **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter,
                 max_p=max_p)
    solver.solution(range(min(nroot, parameters.shape[0])), parameters, actions)
    return solver


def Solve_Linear_Equations(parameters, actions, problem: Problem, generate_initial_guess=True, max_iter=1000000,
                           max_p=0, **kwargs):
    """
    Solve linear equations whose right-hand sides are provided by problem.RHS(vector, instance), called for
    instance = 0, 1, ... until it returns False. On exit, parameters holds the solutions.
    """
    if not hasattr(problem, 'RHS'):
        raise TypeError('Solve_Linear_Equations: the problem must provide RHS(vector, instance)')
    rhs = []
    vector = np.empty(parameters.shape[-1])
    while problem.RHS(vector, len(rhs)):
        rhs.append(vector.copy())
    if not rhs:
        raise ValueError('Solve_Linear_Equations: the problem provides no right-hand sides')
    solver = LinearEquations(np.array(rhs), **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter,
                 max_p=max_p)
    solver.solution(range(min(len(rhs), parameters.shape[0])), parameters, actions)
    return solver


def Solve_NonLinear_Equations(parameters, actions, problem: Problem, generate_initial_guess=True, max_iter=1000000,
                              **kwargs):
    solver = NonLinearEquations(parameters.shape[-1], **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter)
    return solver


def Solve_Optimization(parameters, actions, problem: Problem, generate_initial_guess=True, max_iter=1000000, **kwargs):
    solver = Optimize(parameters.shape[-1], **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter)
    return solver
