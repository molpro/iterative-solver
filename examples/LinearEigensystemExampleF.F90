!> @examples LinearEigensystemExampleF.F90
!> This is an examples of use of the LinearEigensystem framework for iterative
!> finding of the lowest few eigensolutions of a large matrix.
!> Comparison is made of explicitly declaring a (bad) P-space, generating one automatically, or not using one
PROGRAM Linear_Eigensystem_Example
  USE Iterative_Solver, only : mpi_init, mpi_finalize, mpi_rank_global, &
      Solve_Linear_Eigensystem, Iterative_Solver_Print_Statistics, Iterative_Solver_Finalize, &
      Iterative_Solver_Eigenvalues, Iterative_Solver_Converged
  USE Iterative_Solver_Matrix_Problem, only : matrix_problem
  USE iso_fortran_env, only : output_unit
  INTEGER, PARAMETER :: n = 300, nroot = 3
  INTEGER :: nP, max_p
  DOUBLE PRECISION, DIMENSION (:, :), allocatable, target :: m
  INTEGER :: i
  DOUBLE PRECISION, DIMENSION(nroot) :: reference_eigenvalues
  LOGICAL :: have_reference = .false.
  CALL MPI_INIT
  IF (mpi_rank_global() .gt. 0) close(output_unit)
  ALLOCATE(m(n, n))
  m = 1
  DO i = lbound(m,1), ubound(m,1)
    m(i, i) = 3 * (n + 1 - i)
  END DO
  do nP = 0, 50, 50
    do max_p = 0, 50, 10
      if (max_p.gt.0 .and. nP.gt.0) cycle
      WRITE (6, *) 'Explicit P-space=', nP, ', auto P-space=', max_p, ', dimension=', n, ', roots=', nroot
      call solve(m, nP, max_p)
    end do
  end do
  DEALLOCATE(m)
  CALL MPI_Finalize
CONTAINS
  subroutine solve(m, nP, max_P)
    DOUBLE PRECISION, DIMENSION (:, :), INTENT(IN), target :: m
    integer, intent(in) :: nP, max_P
    DOUBLE PRECISION, DIMENSION (ubound(m,1), nroot) :: c, g
    TYPE(Matrix_Problem) :: problem
    DOUBLE PRECISION, DIMENSION(nroot) :: eigenvalues
    INTEGER :: k
    CALL problem%attach(m)
    CALL problem%p_space%add_simple([(i, i = 1, nP)]) ! the first nP components, so not the best
    CALL Solve_Linear_Eigensystem(c, g, problem, nroot, verbosity = 2, thresh = 1d-8, hermitian = .true., max_p = max_p)
    if (mpi_rank_global().eq.0) CALL Iterative_Solver_Print_Statistics
    ! check the solution independently of the solver, and that every P-space choice gives the same eigenvalues
    eigenvalues = Iterative_Solver_Eigenvalues()
    IF (.NOT. Iterative_Solver_Converged()) ERROR STOP 'LinearEigensystemExampleF: not converged'
    DO k = 1, nroot
      IF (norm2(MATMUL(m, c(:, k)) - eigenvalues(k) * c(:, k)) .GT. 1d-6 * norm2(c(:, k))) &
          ERROR STOP 'LinearEigensystemExampleF: wrong eigenpair'
    END DO
    IF (have_reference) THEN
      IF (MAXVAL(ABS(eigenvalues - reference_eigenvalues)) .GT. 1d-8) &
          ERROR STOP 'LinearEigensystemExampleF: eigenvalues depend on the P space'
    ELSE
      reference_eigenvalues = eigenvalues
      have_reference = .true.
    END IF
    CALL Iterative_Solver_Finalize
  end subroutine solve
END PROGRAM Linear_Eigensystem_Example
