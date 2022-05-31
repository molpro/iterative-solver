!> @examples OptimizeExampleF-problem.F90
module Example_Problem
  USE Iterative_Solver_Problem
  private
  !> @brief objective function is (1/2) * c . m . c - sum(c)  where m(i,j) = 1 + (3*i-1)*delta(i,j)
  type, extends(Problem), public :: problem_t
  contains
    procedure, pass :: action => action
    procedure, pass :: diagonals => diagonals
  end type problem_t

contains

  subroutine action(this, parameters, actions)
    class(problem_t), intent(in) :: this
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: actions
    do i = lbound(actions, 1), ubound(actions, 1)
      actions(i, 1) = sum(parameters(:, 1)) + (3 * i - 1) * parameters(i, 1)
    enddo
  end subroutine action

  logical function diagonals(this, d)
    class(problem_t), intent(in) :: this
    double precision, intent(inout), dimension(:) :: d
    d = [(3 * i, i = 1, size(d))]
    diagonals = .true.
  end function diagonals

end module Example_Problem

PROGRAM QuasiNewton_Example
  USE Iterative_Solver
  USE Example_Problem
  IMPLICIT NONE
  interface
    subroutine mpi_init() BIND (C, name = 'mpi_init')
    end subroutine mpi_init
    subroutine mpi_finalize() BIND (C, name = 'mpi_finalize')
    end subroutine mpi_finalize
  end interface
  INTEGER, PARAMETER :: n = 50, verbosity = 2
  DOUBLE PRECISION, DIMENSION (n,2) :: c, g
  type(problem_t) :: problem

  call mpi_init

  CALL Iterative_Solver_Linear_Eigensystem_Initialize(n, thresh = 1d-6, verbosity = verbosity, &
    options = "max_size_qspace=10", nroot=2)
  c = 0;  c(1,1) = 1; c(2,2)=1
  CALL Iterative_Solver_Solve(c, g, problem)
    print *, 'Optimized eigenvalues ', Iterative_Solver_Eigenvalues()
  if (verbosity.lt.1) then
    print *, 'Error ', Iterative_Solver_Errors()
  end if
  if (verbosity.gt.1) then
    call Iterative_Solver_Solution([1], c, g)
    PRINT *, 'solution ', c(1:MIN(n, 10),:)
  end if
  CALL Iterative_Solver_Finalize

  call mpi_finalize
END PROGRAM QuasiNewton_Example
