import gc
import unittest

import numpy as np

import iterative_solver


class Diagonal(iterative_solver.Problem):
    '''M_ij = scale * (i+1) delta_ij + rho'''

    def __init__(self, n, scale, rho=0.1):
        super().__init__()
        self.size = n
        self.matrix = rho * np.ones((n, n))
        for i in range(n):
            self.matrix[i, i] += scale * (i + 1)

    def action(self, parameters, actions):
        np.matmul(parameters, self.matrix, out=actions)

    def diagonals(self, diagonals):
        diagonals[:self.size] = np.diag(self.matrix)
        return True

    @property
    def eigenvalues(self):
        return np.linalg.eigvalsh(self.matrix)


class TestMultipleSolvers(unittest.TestCase):
    '''Several solvers alive at once must each act on their own C-side instance'''
    n = 8

    def make(self, scale, nroot):
        # Solvers differ in nroot, so that acting on the wrong C-side instance is detectable
        problem = Diagonal(self.n, scale)
        solver = iterative_solver.LinearEigensystem(self.n, nroot, hermitian=True, thresh=1e-10)
        return problem, solver

    def solve_and_check(self, problem, solver):
        parameters = np.zeros([solver.nroot, self.n])
        actions = np.zeros([solver.nroot, self.n])
        solver.solve(parameters, actions, problem, generate_initial_guess=True)
        self.assertEqual(solver.errors.size, solver.nroot)
        np.testing.assert_allclose(solver.eigenvalues, problem.eigenvalues[:solver.nroot], rtol=1e-8)

    def test_use_older_solver_while_newer_alive(self):
        problem_a, solver_a = self.make(1.0, 1)
        problem_b, solver_b = self.make(3.0, 2)
        self.solve_and_check(problem_a, solver_a)
        self.solve_and_check(problem_b, solver_b)

    def test_delete_older_solver_first(self):
        problem_a, solver_a = self.make(1.0, 1)
        problem_b, solver_b = self.make(3.0, 2)
        del solver_a
        gc.collect()
        self.solve_and_check(problem_b, solver_b)

    def test_delete_newer_solver_first(self):
        problem_a, solver_a = self.make(1.0, 1)
        problem_b, solver_b = self.make(3.0, 2)
        del solver_b
        gc.collect()
        self.solve_and_check(problem_a, solver_a)

    def test_finalised_solver_is_refused(self):
        problem, solver = self.make(1.0, 1)
        solver.__del__()
        with self.assertRaises(RuntimeError):
            solver.errors


if __name__ == '__main__':
    unittest.main()
