from .problem import Problem
from .iterative_solver_extension import Optimize, LinearEigensystem, LinearEquations, NonLinearEquations


def Solve_Linear_Eigensystem(parameters, actions, problem: Problem, nroot=1, generate_initial_guess=True, max_iter=1000000, max_p=0, **kwargs):
    for k,v in kwargs.items():
        print(k,v)
    solver = LinearEigensystem(parameters.shape[-1], nroot, **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter)#, max_p=max_p)
    return solver


def Solve_Linear_Equations(parameters, actions, problem: Problem, generate_initial_guess=True, max_iter=1000000, max_p=0, **kwargs):
    solver = LinearEquations(parameters.shape[-1], **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter, max_p=max_p)
    return solver


def Solve_NonLinear_Equations(parameters, actions, problem: Problem, generate_initial_guess=True, max_iter=1000000, **kwargs):
    solver = NonLinearEquations(parameters.shape[-1], **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter)
    return solver


def Solve_Optimization(parameters, actions, problem: Problem, generate_initial_guess=True, max_iter=1000000, **kwargs):
    solver = Optimize(parameters.shape[-1], **kwargs)
    solver.solve(parameters, actions, problem, generate_initial_guess=generate_initial_guess, max_iter=max_iter)
    return solver
