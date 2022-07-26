!> @examples LinearEigensystemExampleF.F90
!> This is an example of simplest use of the LinearEigensystem framework for iterative
!> finding of the lowest few eigensolutions of a large matrix.
PROGRAM Linear_Eigensystem_Example
  USE Iterative_Solver
  interface
    subroutine mpi_init() BIND (C, name = 'mpi_init')
    end subroutine mpi_init
    subroutine mpi_finalize() BIND (C, name = 'mpi_finalize')
    end subroutine mpi_finalize
    !    function mpi_comm_global() BIND (C, name = 'mpi_comm_global')
    !      use iso_c_binding, only: c_int64_t
    !      integer(c_int64_t) mpi_comm_global
    !    end function mpi_comm_global
  end interface
  INTEGER, PARAMETER :: n = 6, nroot = 3
  DOUBLE PRECISION, DIMENSION (n, n) :: m
  DOUBLE PRECISION, DIMENSION (n, nroot) :: c, g
  DOUBLE PRECISION, DIMENSION (nroot) :: e, error
  INTEGER :: i, j, root
  LOGICAL :: converged
  PRINT *, 'Fortran binding of IterativeSolver'
  m = 1; DO i = 1, n; m(i, i) = 3 * i;
  END DO
  CALL Iterative_Solver_Linear_Eigensystem_Initialize(n, nroot, thresh = 1d-7, verbosity = 1)
  c = 0; DO i = 1, nroot; c(i, i) = 1;
  ENDDO
  DO i = 1, n
    g = MATMUL(m, c)
    IF (Iterative_Solver_Add_Vector(c, g, e)) THEN
      e = Iterative_Solver_Eigenvalues()
      DO root = 1, nroot
        DO j = 1, n
          c(j, root) = c(j, root) - g(j, root) / (m(j, j) - e(root) + 1e-15)
        END DO
      END DO
    END IF
    converged = Iterative_Solver_End_Iteration(c, g, error)
    IF (converged) EXIT
  END DO
  PRINT *, 'error =', error, ' eigenvalue =', e
  CALL Iterative_Solver_Print_Statistics
  CALL Iterative_Solver_Finalize
  CALL mpi_finalize
END PROGRAM Linear_Eigensystem_Example
