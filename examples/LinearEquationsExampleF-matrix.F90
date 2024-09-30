!> @examples LinearEquationsExampleF-matrix.F90
!> This is an examples of simplest use of the LinearEquations framework for iterative
!> solution of linear equations
PROGRAM Linear_Equations_Example
  USE Iterative_Solver, only : mpi_init, mpi_finalize, mpi_rank_global, &
      Solve_Linear_Equations, Iterative_Solver_Print_Statistics, Iterative_Solver_Finalize
  USE Iterative_Solver_Matrix_Problem, only : Matrix_Problem
  IMPLICIT NONE
  INTEGER, PARAMETER :: n = 300, nroot = 2
  DOUBLE PRECISION, PARAMETER :: alpha = 300
  DOUBLE PRECISION, DIMENSION(1), PARAMETER :: augmented_hessian_factors = [0.0_8]! issue 510 , .001_8, .01_8, .1_8, 1.0_8]
  DOUBLE PRECISION, DIMENSION (n, n), target :: m
  DOUBLE PRECISION, DIMENSION (n, nroot), target :: rhs
  DOUBLE PRECISION, DIMENSION (n, nroot) :: c, g
  DOUBLE PRECISION :: augmented_hessian
  INTEGER :: iaug
  LOGICAL :: converged
  TYPE(Matrix_Problem) :: problem
  CALL mpi_init
  PRINT *, 'Fortran binding of IterativeSolver'
  IF (mpi_rank_global() .gt. 0) CLOSE(6)
  CALL initialise_matrices
  CALL problem%attach(m, rhs)
  DO iaug = 1, SIZE(augmented_hessian_factors)
    augmented_hessian = augmented_hessian_factors(iaug)
    PRINT *, 'solve linear system with augmented hessian factor ', augmented_hessian
    converged = Solve_Linear_Equations(c, g, problem, augmented_hessian = augmented_hessian, thresh = 1d-11, verbosity = 2, max_p = 30, hermitian = .true.)
    PRINT *, 'convergence?', converged, ', residual length: ', norm2(g)
    !    print *,c
    !    print *,g
    CALL Iterative_Solver_Print_Statistics
    CALL Iterative_Solver_Finalize
  ENDDO
  CALL mpi_finalize
CONTAINS
  SUBROUTINE initialise_matrices
    INTEGER :: i, j
    DO i = 1, n; m(i, i) = alpha * i + 2 * i - 2; DO j = 1, n; IF (i.NE.j) m(i, j) = i + j - 2;
    END DO;
    END DO
    DO i = 1, nroot; DO j = 1, n; rhs(j, i) = 1 / DBLE(j + i - 1);
    END DO;
    END DO
  END SUBROUTINE initialise_matrices
END PROGRAM Linear_Equations_Example
