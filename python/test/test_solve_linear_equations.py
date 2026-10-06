import unittest

import numpy as np

import iterative_solver


class TestSolveLinearEquations(unittest.TestCase):
    n = 20

    def setUp(self):
        self.H = np.diag(np.arange(1.0, self.n + 1)) + 0.1
        self.b = np.random.default_rng(2).standard_normal((2, self.n))
        self.problem = iterative_solver.MatrixProblem()
        self.problem.attach(self.H, self.b)

    def test_solve_linear_equations(self):
        c = np.zeros((2, self.n))
        g = np.zeros((2, self.n))
        solver = iterative_solver.Solve_Linear_Equations(c, g, self.problem, thresh=1e-10, hermitian=True)
        self.assertTrue(solver.converged)
        for root in range(2):
            np.testing.assert_allclose(self.H @ c[root], self.b[root], atol=1e-8 * np.linalg.norm(self.b[root]))

    def test_solve_linear_equations_augmented_hessian(self):
        a = 0.1
        c = np.zeros((2, self.n))
        g = np.zeros((2, self.n))
        solver = iterative_solver.Solve_Linear_Equations(c, g, self.problem, augmented_hessian=a, thresh=1e-10,
                                                         hermitian=True)
        self.assertTrue(solver.converged)
        for root in range(2):
            lam = -a * a * self.b[root].dot(c[root])
            residual = (self.H - lam * np.eye(self.n)) @ c[root] - self.b[root]
            self.assertLess(np.linalg.norm(residual), 1e-8 * np.linalg.norm(self.b[root]))

    def test_aughes_is_still_accepted(self):
        a = 0.1
        results = []
        for kwargs in ({'aughes': a}, {'augmented_hessian': a}):
            c = np.zeros((1, self.n))
            g = np.zeros((1, self.n))
            solver = iterative_solver.LinearEquations(self.b[:1], thresh=1e-10, hermitian=True, **kwargs)
            solver.solve(c, g, self.problem, generate_initial_guess=True)
            solver.solution([0], c, g)
            results.append(c[0].copy())
        np.testing.assert_allclose(results[0], results[1], rtol=1e-10)

    def test_problem_without_rhs_is_refused(self):
        with self.assertRaises(TypeError):
            iterative_solver.Solve_Linear_Equations(np.zeros((1, self.n)), np.zeros((1, self.n)),
                                                    iterative_solver.Problem())


if __name__ == '__main__':
    unittest.main()
