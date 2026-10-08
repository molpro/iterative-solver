import unittest
import iterative_solver
import numpy as np


class TestCase(unittest.TestCase):
    class Quadratic(iterative_solver.Problem):
        """f = sign * (x.M.x/2 - sum(x)), stationary where M.x = 1"""

        def __init__(self, n, sign=1, **kwargs):
            super().__init__(**kwargs)
            self.size = n
            self.sign = sign
            i = np.arange(1, n + 1)
            self.matrix = 0.01 * np.cos(i[:, None] + 2 * i[None, :]) + 0.01 * np.cos(i[None, :] + 2 * i[:, None])
            np.fill_diagonal(self.matrix, i)

        def residual(self, parameters, gradient):
            gradient[:] = self.sign * (self.matrix @ parameters - 1)
            return self.sign * (0.5 * parameters @ self.matrix @ parameters - parameters.sum())

        def diagonals(self, diagonals):
            diagonals[:] = self.sign * np.diag(self.matrix)
            return True

    def solve(self, sign, **kwargs):
        problem = TestCase.Quadratic(40, sign)
        parameters = np.zeros(problem.size)
        residual = np.zeros(problem.size)
        solver = iterative_solver.Optimize(problem.size, thresh=1e-8, **kwargs)
        solver.solve(parameters, residual, problem)
        value = solver.solution([0], parameters, residual)
        return solver, problem, parameters, residual, value

    def test_maximize(self):
        # minimize=False maximises: the maximum of -f is the minimum of f, with the value negated (issue #633)
        _, _, _, _, minimum = self.solve(1)
        solver, problem, parameters, residual, value = self.solve(-1, minimize=False)
        self.assertTrue(solver.converged)
        self.assertLess(np.linalg.norm(problem.matrix @ parameters - 1), 1e-7)
        self.assertLess(np.linalg.norm(residual), 1e-7)
        self.assertAlmostEqual(value, -minimum, places=12)


if __name__ == '__main__':
    unittest.main()
