PROGRAM Linear_Eigensystem_Benchmark
  USE Iterative_Solver, only : mpi_init, mpi_finalize, mpi_rank_global, &
      Iterative_Solver_Linear_Eigensystem_Initialize, Iterative_Solver_Finalize, &
      Iterative_Solver_Add_Vector, Iterative_Solver_End_Iteration, &
      Iterative_Solver_Working_Set_Eigenvalues, Iterative_Solver_Eigenvalues, Iterative_Solver_Errors
  USE ProfilerF
  IMPLICIT NONE
  TYPE(Profiler) :: p
  INTEGER, PARAMETER :: n = 2000000, nroot = 3, max_iter = 1000
  ! The matrix is m(i, j) = 1 + (3 * i - 1) * delta(i, j), applied without storing it
  DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: c, g
  DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: eigs
  DOUBLE PRECISION :: su
  INTEGER :: iter, j, root, nwork
  CALL mpi_init
  IF (mpi_rank_global() .EQ. 0) PRINT *, 'Fortran binding of IterativeSolver'
  CALL p%construct('Benchmark driver')
  ALLOCATE(c(n, nroot), g(n, nroot))
  CALL Iterative_Solver_Linear_Eigensystem_Initialize(n, nroot, pname = 'Benchmark', thresh = 1d-7, verbosity = 1)
  c = 0
  DO root = 1, nroot
    c(root, root) = 1
  END DO
  DO iter = 1, max_iter
    CALL p%start('action')
    DO root = 1, nroot
      su = SUM(c(:, root))
      DO j = 1, n
        g(j, root) = (3 * j - 1) * c(j, root) + su
      END DO
    END DO
    CALL p%stop('action')
    CALL p%start('Add_Vector')
    nwork = Iterative_Solver_Add_Vector(c, g)
    CALL p%stop('Add_Vector')
    IF (nwork .LE. 0) EXIT
    CALL p%start('precondition')
    eigs = Iterative_Solver_Working_Set_Eigenvalues(nwork)
    DO root = 1, nwork
      DO j = 1, n
        g(j, root) = -g(j, root) / (3 * j - eigs(root) + 1d-12)
      END DO
    END DO
    CALL p%stop('precondition')
    CALL p%start('End_Iteration')
    nwork = Iterative_Solver_End_Iteration(c, g)
    CALL p%stop('End_Iteration')
    IF (nwork .LE. 0) EXIT
  END DO
  IF (mpi_rank_global() .EQ. 0) THEN
    PRINT *, 'error =', Iterative_Solver_Errors(), ' eigenvalue =', Iterative_Solver_Eigenvalues()
  END IF
  CALL p%start('Finalize')
  CALL Iterative_Solver_Finalize
  CALL p%stop('Finalize')
  IF (mpi_rank_global() .EQ. 0) CALL p%print(6)
  CALL p%destroy
  DEALLOCATE(c, g)
  CALL mpi_finalize
END PROGRAM Linear_Eigensystem_Benchmark
