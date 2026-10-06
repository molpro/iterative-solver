import unittest

import numpy as np

import iterative_solver


class TestMaxSizeQspace(unittest.TestCase):
    '''Limiting the Q space forces construction of a D space from disk-backed Q vectors (issue #511)'''

    def test_eigensystem_with_small_qspace(self):
        n, nroot = 100, 5
        m = np.array([[1 if i != j else 3 * (n - i) for i in range(n)] for j in range(n)], dtype=float)
        problem = iterative_solver.MatrixProblem()
        problem.attach(m)
        for max_size_qspace in (6, 9):
            with self.subTest(max_size_qspace=max_size_qspace):
                c = np.empty((nroot, n))
                g = np.empty((nroot, n))
                solver = iterative_solver.Solve_Linear_Eigensystem(c, g, problem, nroot, thresh=1e-8, hermitian=True,
                                                                   options='max_size_qspace=%d' % max_size_qspace)
                self.assertTrue(solver.converged)
                np.testing.assert_allclose(solver.eigenvalues, np.linalg.eigvalsh(m)[:nroot], rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
