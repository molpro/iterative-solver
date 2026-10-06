import unittest

import numpy as np

import iterative_solver


class TestAugmentedHessian(unittest.TestCase):
    '''With augmented Hessian parameter a, the solution satisfies (H - lambda) x = b with lambda = -a^2 b.x (issue #510)'''

    def test_linear_equations(self):
        n = 20
        H = np.diag(np.arange(1.0, n + 1)) + 0.1
        b = np.random.default_rng(1).standard_normal(n)
        problem = iterative_solver.MatrixProblem()
        problem.attach(H, b.reshape(1, n))
        for a in (0.01, 0.1, 1.0):
            with self.subTest(a=a):
                c = np.zeros((1, n))
                g = np.zeros((1, n))
                solver = iterative_solver.LinearEquations(b.reshape(1, n), aughes=a, thresh=1e-10, hermitian=True)
                self.assertTrue(solver.solve(c, g, problem, generate_initial_guess=True))
                solver.solution([0], c, g)
                x = c[0]
                lam = -a * a * b.dot(x)
                self.assertLess(np.linalg.norm((H - lam * np.eye(n)) @ x - b), 1e-8 * np.linalg.norm(b))


if __name__ == '__main__':
    unittest.main()
