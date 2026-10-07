!> @examples NonLinearExampleF.F90
!> This is an example of use of either the Optimize (BFGS) or NonlinearEquations (DIIS) framework for iterative
!> minimisation of a non-linear function using the simplified driver.
!> The first example makes stationary a normalised quadratic form, so is equivalent to finding an eigenvector.
!> The second example makes stationary a quadratic form plus a linear force.
module QuasiNewton_Examples
  USE Iterative_Solver_Problem
  implicit none
  private
  !> @brief objective function is (1/2) * c . m . c - sum(c)  where m(i,j) = 1 + (3*i-1)*delta(i,j)
  type, extends(Problem), public :: forced_t
    integer :: size
  contains
    procedure, pass :: residual => forced_residual
    procedure, pass :: diagonals => forced_diagonals
  end type forced_t

  !> @brief objective function is (1/2) * c . m . c / c . c  where m(i,j) = 1 + (3*i-1)*delta(i,j)
  type, extends(Problem), public :: quadratic_t
  contains
    procedure, pass :: residual => quadratic_residual
    procedure, pass :: diagonals => quadratic_diagonals
  end type quadratic_t

contains

  function forced_residual(this, parameters, residuals, range) result(e)
    class(forced_t), intent(in) :: this
    double precision :: e
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: residuals
    integer, dimension(2), intent(in) :: range
    integer :: i
    ! Residuals are needed only in this process's range, but the function value must be the same on every process,
    ! so compute it from the full parameter vector
    do i = range(1) + 1, range(2); residuals(i, 1) = sum(parameters(:, 1)) + (3 * i - 1) * parameters(i, 1) - 1;
    enddo
    e = 0.5 * (sum(parameters(:, 1))**2 + sum([((3 * i - 1) * parameters(i, 1)**2, i = 1, size(parameters, 1))])) &
        - sum(parameters(:, 1))
  end function forced_residual

  logical function forced_diagonals(this, d)
    class(forced_t), intent(in) :: this
    double precision, intent(inout), dimension(:) :: d
    integer :: i
    d = [(3 * i, i = 1, size(d))]
    forced_diagonals = .true.
  end function forced_diagonals

  function quadratic_residual(this, parameters, residuals, range) result(e)
    class(quadratic_t), intent(in) :: this
    double precision :: e
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: residuals
    integer, dimension(2), intent(in) :: range
    integer :: i
    ! As in forced_residual, the function value is computed from the full parameter vector
    e = (sum(parameters(:, 1))**2 + sum([((3 * i - 1) * parameters(i, 1)**2, i = 1, size(parameters, 1))])) &
        / dot_product(parameters(:, 1), parameters(:, 1))
    do i = range(1) + 1, range(2)
      residuals(i, 1) = (sum(parameters(:, 1)) + (3 * i - 1) * parameters(i, 1) - e * parameters(i, 1)) &
          / dot_product(parameters(:, 1), parameters(:, 1))
    enddo
  end function quadratic_residual

  logical function quadratic_diagonals(this, d)
    class(quadratic_t), intent(in) :: this
    double precision, intent(inout), dimension(:) :: d
    integer :: i
    d = [(3 * i, i = 1, size(d))]
    quadratic_diagonals = .true.
  end function quadratic_diagonals

end module QuasiNewton_Examples

PROGRAM QuasiNewton_Example
  USE Iterative_Solver
  USE QuasiNewton_Examples
  IMPLICIT NONE
  INTEGER, PARAMETER :: n = 100, verbosity = 2
  DOUBLE PRECISION, DIMENSION (n) :: c, g
  ! quadratic_t is the alternative problem: a normalised quadratic form, equivalent to finding an eigenvector
  type(forced_t) :: problem
  call mpi_init
  ! The same problem is solved by minimisation (BFGS) and as non-linear equations (DIIS)
  call Solve_Optimization(c, g, problem, thresh = 1d-6, verbosity = verbosity)
  call report_and_check('Solve_Optimization')
  call Solve_Nonlinear_Equations(c, g, problem, thresh = 1d-6, verbosity = verbosity)
  call report_and_check('Solve_Nonlinear_Equations')
  call mpi_finalize
contains
  !> Check the solution independently of the solver: it is stationary when sum(c) + (3*i-1)*c(i) = 1 for all i
  subroutine report_and_check(driver)
    character(*), intent(in) :: driver
    double precision :: error
    integer :: i
    if (verbosity.gt.1) then
      call Iterative_Solver_Solution([1], c, g)
      PRINT *, 'solution ', c(1:MIN(n, 10))
      PRINT *, 'residual ', g(1:MIN(n, 10))
    end if
    error = maxval([(abs(sum(c) + (3 * i - 1) * c(i) - 1), i = 1, n)])
    print *, driver, ': converged ', Iterative_Solver_Converged(), ', stationarity error ', error
    if (.not. Iterative_Solver_Converged() .or. error .gt. 1d-5) error stop 'NonLinearExampleF: wrong solution'
    CALL Iterative_Solver_Finalize
  end subroutine report_and_check
END PROGRAM QuasiNewton_Example
