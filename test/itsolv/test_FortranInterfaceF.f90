!> Problems used to test the Fortran interface. Actions and residuals are computed only within the process's range,
!> and function values are the same on every process. Callback calls are recorded in module variables, since the
!> callbacks receive their object as intent(in).
module fortran_interface_test_problems
  use Iterative_Solver_Problem, only : Problem
  implicit none
  private
  integer, parameter, public :: n = 40
  integer, public :: n_precondition = 0, n_report = 0, last_report_iteration = huge(1), n_rhs = 0

  !> m(i,j) = i delta(i,j) + small symmetric coupling
  type, extends(Problem), public :: matrix_problem
  contains
    procedure, pass :: action => matrix_action
    procedure, pass :: diagonals => matrix_diagonals
  end type matrix_problem

  !> As matrix_problem, overriding precondition, report and RHS and recording their use
  type, extends(matrix_problem), public :: counting_problem
  contains
    procedure, pass :: precondition => counting_precondition
    procedure, pass :: report => counting_report
    procedure, pass :: RHS => counting_rhs
  end type counting_problem

  !> f(c) = sign * ((1/2) c.m.c - sum(c)), stationary where m.c = 1
  type, extends(Problem), public :: forced_problem
    double precision :: sign = 1
  contains
    procedure, pass :: residual => forced_residual
    procedure, pass :: diagonals => forced_diagonals
  end type forced_problem

  public :: matrix_element, reset_counters, eigen_residual, forced_error

contains

  pure double precision function matrix_element(i, j)
    integer, intent(in) :: i, j
    if (i .eq. j) then
      matrix_element = i
    else
      matrix_element = 0.01d0 * cos(dble(i + 2 * j)) + 0.01d0 * cos(dble(j + 2 * i))
    end if
  end function matrix_element

  subroutine reset_counters
    n_precondition = 0
    n_report = 0
    last_report_iteration = huge(1)
    n_rhs = 0
  end subroutine reset_counters

  subroutine matrix_action(this, parameters, actions, range)
    class(matrix_problem), intent(in) :: this
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: actions
    integer, dimension(2), intent(in) :: range
    integer :: i, j, k
    do k = 1, size(parameters, 2)
      do i = range(1) + 1, range(2)
        actions(i, k) = sum([(matrix_element(i, j) * parameters(j, k), j = 1, n)])
      end do
    end do
  end subroutine matrix_action

  logical function matrix_diagonals(this, d)
    class(matrix_problem), intent(in) :: this
    double precision, intent(inout), dimension(:) :: d
    integer :: i
    d(:n) = [(matrix_element(i, i), i = 1, n)]
    matrix_diagonals = .true.
  end function matrix_diagonals

  subroutine counting_precondition(this, action, shift, diagonals, range)
    class(counting_problem), intent(in) :: this
    double precision, intent(inout), dimension(:, :) :: action
    double precision, intent(in), dimension(:), optional :: shift
    double precision, intent(in), dimension(:), optional :: diagonals
    integer, dimension(2), intent(in) :: range
    integer :: i, k
    double precision :: s
    n_precondition = n_precondition + 1
    do k = 1, size(action, 2)
      s = 0
      if (present(shift)) s = shift(k)
      do i = range(1) + 1, range(2)
        action(i, k) = action(i, k) / (matrix_element(i, i) - s + 1d-12)
      end do
    end do
  end subroutine counting_precondition

  logical function counting_report(this, iteration, verbosity, errors, value, eigenvalues)
    class(counting_problem), intent(in) :: this
    integer, intent(in) :: iteration, verbosity
    double precision, intent(in), dimension(:) :: errors
    double precision, intent(in), optional :: value
    double precision, dimension(:), intent(in), optional :: eigenvalues
    n_report = n_report + 1
    last_report_iteration = iteration
    counting_report = .true. ! suppresses the default report
  end function counting_report

  logical function counting_rhs(this, vector, instance, range)
    class(counting_problem), intent(in) :: this
    double precision, intent(inout), dimension(:) :: vector
    integer, intent(in) :: instance
    integer, dimension(2), intent(in) :: range
    integer :: i
    n_rhs = n_rhs + 1
    counting_rhs = instance .ge. 1 .and. instance .le. 2
    if (counting_rhs) then
      do i = range(1) + 1, range(2)
        vector(i) = 1d0 / dble(i + instance)
      end do
    end if
  end function counting_rhs

  function forced_residual(this, parameters, residuals, range) result(value)
    class(forced_problem), intent(in) :: this
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: residuals
    integer, dimension(2), intent(in) :: range
    double precision :: value
    integer :: i, j
    do i = range(1) + 1, range(2)
      residuals(i, 1) = this%sign * (sum([(matrix_element(i, j) * parameters(j, 1), j = 1, n)]) - 1)
    end do
    value = 0
    do i = 1, n
      value = value + 0.5d0 * parameters(i, 1) * sum([(matrix_element(i, j) * parameters(j, 1), j = 1, n)]) &
          - parameters(i, 1)
    end do
    value = this%sign * value
  end function forced_residual

  logical function forced_diagonals(this, d)
    class(forced_problem), intent(in) :: this
    double precision, intent(inout), dimension(:) :: d
    integer :: i
    d(:n) = [(this%sign * matrix_element(i, i), i = 1, n)]
    forced_diagonals = .true.
  end function forced_diagonals

  !> Largest relative eigenpair residual |m c - e c| / |c| over the columns of c
  double precision function eigen_residual(c, e)
    double precision, intent(in) :: c(:, :), e(:)
    integer :: i, j, k
    double precision :: r(n)
    eigen_residual = 0
    do k = 1, size(c, 2)
      do i = 1, n
        r(i) = sum([(matrix_element(i, j) * c(j, k), j = 1, n)]) - e(k) * c(i, k)
      end do
      eigen_residual = max(eigen_residual, norm2(r) / norm2(c(:, k)))
    end do
  end function eigen_residual

  !> Largest stationarity error |m c - 1| of the forced problem
  double precision function forced_error(c)
    double precision, intent(in) :: c(:)
    integer :: i, j
    forced_error = maxval([(abs(sum([(matrix_element(i, j) * c(j), j = 1, n)]) - 1), i = 1, n)])
  end function forced_error

end module fortran_interface_test_problems

!> Iterative_Solver_Verbosity returns the requested print level (it used to return 2 for any of 0, 1 and 2)
function test_verbosityF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems, only : n
  implicit none
  integer(c_int) :: test_verbosityF
  integer :: v
  test_verbosityF = 1
  do v = 0, 3
    call Iterative_Solver_Linear_Eigensystem_Initialize(n, 1, verbosity = v)
    if (Iterative_Solver_Verbosity() .ne. v) then
      write (6, *) 'test_verbosityF: requested ', v, ' got ', Iterative_Solver_Verbosity()
      test_verbosityF = 0
    end if
    call Iterative_Solver_Finalize
  end do
end function test_verbosityF

!> Iterative_Solver_Finalize_All removes every solver, after which a new one works normally
function test_finalize_allF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems, only : n
  implicit none
  integer(c_int) :: test_finalize_allF
  test_finalize_allF = 1
  call Iterative_Solver_Linear_Eigensystem_Initialize(n, 3)
  call Iterative_Solver_DIIS_Initialize(n)
  call Iterative_Solver_Finalize_All
  call Iterative_Solver_Linear_Eigensystem_Initialize(n, 2)
  if (size(Iterative_Solver_Errors()) .ne. 2) test_finalize_allF = 0
  call Iterative_Solver_Finalize
  call Iterative_Solver_Finalize ! with no solver left, this does nothing
end function test_finalize_allF

!> The compute communicator can be set; with mpicomm_self each process solves the whole problem on its own
function test_mpicommF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems, only : n, matrix_problem, eigen_residual
  implicit none
  integer(c_int) :: test_mpicommF
  integer, parameter :: nroot = 2
  double precision :: c(n, nroot), g(n, nroot)
  type(matrix_problem) :: problem
  integer :: range(2)
  test_mpicommF = 1
  if (mpi_size_global() .lt. 1 .or. mpi_rank_global() .lt. 0 .or. mpi_rank_global() .ge. mpi_size_global()) &
      test_mpicommF = 0
  call set_mpicomm_compute(mpicomm_self())
  if (mpicomm_compute() .ne. mpicomm_self()) test_mpicommF = 0
  range = -1 ! request the default distribution, and receive the range in use
  call Solve_Linear_Eigensystem(c, g, problem, nroot, thresh = 1d-10, hermitian = .true., range = range)
  if (any(range .ne. [0, n])) then
    write (6, *) 'test_mpicommF: with mpicomm_self the range should be the whole space, but is ', range
    test_mpicommF = 0
  end if
  if (eigen_residual(c, Iterative_Solver_Eigenvalues()) .gt. 1d-8) test_mpicommF = 0
  call Iterative_Solver_Finalize
  call set_mpicomm_compute(mpicomm_global())
  if (mpicomm_compute() .ne. mpicomm_global()) test_mpicommF = 0
end function test_mpicommF

!> generate_initial_guess=.false. starts from the supplied vectors, and max_iter limits the iterations
function test_initial_guess_and_max_iterF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems, only : n, matrix_problem, eigen_residual
  implicit none
  integer(c_int) :: test_initial_guess_and_max_iterF
  integer, parameter :: nroot = 2
  double precision :: c(n, nroot), g(n, nroot)
  type(matrix_problem) :: problem
  integer :: iterations_from_scratch
  test_initial_guess_and_max_iterF = 1
  call Solve_Linear_Eigensystem(c, g, problem, nroot, thresh = 1d-10, hermitian = .true.)
  iterations_from_scratch = Iterative_Solver_Iterations
  call Iterative_Solver_Finalize
  ! restart from the converged eigenvectors
  call Solve_Linear_Eigensystem(c, g, problem, nroot, thresh = 1d-10, hermitian = .true., &
      generate_initial_guess = .false.)
  if (.not. Iterative_Solver_Converged() .or. Iterative_Solver_Iterations .ge. iterations_from_scratch &
      .or. eigen_residual(c, Iterative_Solver_Eigenvalues()) .gt. 1d-8) then
    write (6, *) 'test_initial_guess_and_max_iterF: restart took ', Iterative_Solver_Iterations, &
        ' iterations, from scratch ', iterations_from_scratch
    test_initial_guess_and_max_iterF = 0
  end if
  call Iterative_Solver_Finalize
  if (iterations_from_scratch .gt. 2) then
    call Solve_Linear_Eigensystem(c, g, problem, nroot, thresh = 1d-10, hermitian = .true., max_iter = 2)
    if (Iterative_Solver_Converged() .or. Iterative_Solver_Iterations .ne. 2) then
      write (6, *) 'test_initial_guess_and_max_iterF: with max_iter=2, iterations=', Iterative_Solver_Iterations
      test_initial_guess_and_max_iterF = 0
    end if
    call Iterative_Solver_Finalize
  end if
end function test_initial_guess_and_max_iterF

!> options= and algorithm= reach the solver
function test_options_and_algorithmF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems, only : n, forced_problem, forced_error
  implicit none
  integer(c_int) :: test_options_and_algorithmF
  double precision :: c(n), g(n)
  type(forced_problem) :: problem
  integer :: iterations_default, iterations_max_iter_1, iterations_explicit
  test_options_and_algorithmF = 1
  call Solve_Nonlinear_Equations(c, g, problem, thresh = 1d-9)
  iterations_default = Iterative_Solver_Iterations
  if (.not. Iterative_Solver_Converged() .or. forced_error(c) .gt. 1d-7) test_options_and_algorithmF = 0
  call Iterative_Solver_Finalize
  call Solve_Nonlinear_Equations(c, g, problem, thresh = 1d-9, algorithm = 'DIIS')
  iterations_explicit = Iterative_Solver_Iterations
  call Iterative_Solver_Finalize
  ! an option with a deterministic effect: the solver stops after one iteration, unconverged
  call Solve_Nonlinear_Equations(c, g, problem, thresh = 1d-9, options = 'max_iter=1')
  iterations_max_iter_1 = Iterative_Solver_Iterations
  if (Iterative_Solver_Converged()) test_options_and_algorithmF = 0
  call Iterative_Solver_Finalize
  if (iterations_explicit .ne. iterations_default .or. iterations_max_iter_1 .ne. 1 .or. iterations_default .le. 1) then
    write (6, *) 'test_options_and_algorithmF: iterations default ', iterations_default, ', algorithm=DIIS ', &
        iterations_explicit, ', max_iter=1 ', iterations_max_iter_1
    test_options_and_algorithmF = 0
  end if
end function test_options_and_algorithmF

!> Solve_Optimization minimises, by default and with minimize=.true. (the minimize argument used to be passed to C
!> uninitialised when absent; maximisation is not implemented by the library, so is not tested)
function test_minimizeF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems, only : n, forced_problem, forced_error
  implicit none
  integer(c_int) :: test_minimizeF
  double precision :: c(n), g(n), value_default
  double precision, parameter :: tight(2) = [1d-9, 1d-11]
  integer :: i
  type(forced_problem) :: problem
  test_minimizeF = 1
  call Solve_Optimization(c, g, problem, thresh = 1d-6)
  value_default = Iterative_Solver_Value()
  if (.not. Iterative_Solver_Converged() .or. forced_error(c) .gt. 1d-5) then
    write (6, *) 'test_minimizeF: minimisation failed, error ', forced_error(c)
    test_minimizeF = 0
  end if
  call Iterative_Solver_Finalize
  call Solve_Optimization(c, g, problem, thresh = 1d-6, minimize = .true.)
  if (.not. Iterative_Solver_Converged() .or. forced_error(c) .gt. 1d-5 &
      .or. abs(Iterative_Solver_Value() - value_default) .gt. 1d-10) then
    write (6, *) 'test_minimizeF: minimize=.true. differs from the default'
    test_minimizeF = 0
  end if
  call Iterative_Solver_Finalize
  ! at tight thresholds the convergence flag must agree with the solution returned (issue #634)
  do i = 1, size(tight)
    call Solve_Optimization(c, g, problem, thresh = tight(i))
    if (.not. Iterative_Solver_Converged() .or. maxval(Iterative_Solver_Errors()) .gt. tight(i) &
        .or. norm2(g) .gt. tight(i) .or. forced_error(c) .gt. tight(i)) then
      write (6, *) 'test_minimizeF failed: at thresh ', tight(i), ' expected converged with errors, |g| and', &
          ' stationarity error all within thresh; got converged ', Iterative_Solver_Converged(), &
          ', errors ', Iterative_Solver_Errors(), ', |g| ', norm2(g), ', stationarity error ', forced_error(c)
      test_minimizeF = 0
    end if
    call Iterative_Solver_Finalize
  end do
end function test_minimizeF

!> Overridden precondition, report and RHS are used by the Solve drivers
function test_problem_callbacksF() bind(C)
  use iso_c_binding
  use Iterative_Solver
  use fortran_interface_test_problems
  implicit none
  integer(c_int) :: test_problem_callbacksF
  integer, parameter :: nroot = 2
  double precision :: c(n, nroot), g(n, nroot), r(n)
  type(counting_problem) :: problem
  integer :: i, j, k
  test_problem_callbacksF = 1
  call reset_counters
  call Solve_Linear_Eigensystem(c, g, problem, nroot, thresh = 1d-10, hermitian = .true.)
  if (.not. Iterative_Solver_Converged() .or. eigen_residual(c, Iterative_Solver_Eigenvalues()) .gt. 1d-8 &
      .or. n_precondition .lt. 1 .or. n_report .lt. 2 .or. last_report_iteration .ne. 0) then
    write (6, *) 'test_problem_callbacksF (eigensystem): precondition calls ', n_precondition, ', report calls ', &
        n_report, ', last report iteration ', last_report_iteration
    test_problem_callbacksF = 0
  end if
  call Iterative_Solver_Finalize
  call reset_counters
  call Solve_Linear_Equations(c, g, problem, thresh = 1d-10, hermitian = .true.)
  do k = 1, 2
    do i = 1, n
      r(i) = sum([(matrix_element(i, j) * c(j, k), j = 1, n)]) - 1d0 / dble(i + k)
    end do
    if (norm2(r) .gt. 1d-8) test_problem_callbacksF = 0
  end do
  if (.not. Iterative_Solver_Converged() .or. n_rhs .lt. 3 .or. n_precondition .lt. 1 .or. n_report .lt. 2) then
    write (6, *) 'test_problem_callbacksF (linear equations): RHS calls ', n_rhs, ', precondition calls ', &
        n_precondition, ', report calls ', n_report
    test_problem_callbacksF = 0
  end if
  call Iterative_Solver_Finalize
end function test_problem_callbacksF
