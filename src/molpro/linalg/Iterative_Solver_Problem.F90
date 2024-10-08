module Iterative_Solver_Problem
  use Iterative_Solver_Pspace, only : PSpace
  private

  !> @brief Abstract class defining the problem-specific interface for the simplified solver
  !> interface to IterativeSolver
  type, abstract, public :: Problem
!  type, public :: Problem
    type(PSpace) :: p_space
  contains
    procedure, pass :: diagonals
    procedure, pass :: precondition
    procedure, pass :: residual
    procedure, pass :: action
    procedure, pass :: RHS
    procedure, pass :: report
    procedure, pass :: p_action
    procedure, pass :: pp_action_matrix
  end type Problem
contains

  !> @brief Optionally provide the diagonal elements of the underlying kernel. If
  !> implemented and returning true, the provided diagonals will be used by
  !> the base precondition() function for preconditioning (and therefore the precondition() function does
  !> not need to be reimplemented), and, in the case of linear problems, for selection of
  !> the P space. Otherwise, preconditioning will be done with precondition(), and any P
  !> space has to be provided manually.
  !> @param d On exit, contains the diagonal elements
  !> @return Whether diagonals have been provided.
  logical function diagonals(this, d)
    class(Problem), intent(in) :: this
    double precision, intent(inout), dimension(:) :: d
    diagonals = .false.
  end function diagonals

  !> @brief Apply preconditioning to a residual vector in order to predict a step towards
  !> the solution.
  !> This is a complete implementation provided that the diagonals() function is reimplemented, but can be overridden if desired.
  !> @param residual On entry, assumed to be the residual. On exit, the negative of the
  !> predicted step.
  !> @param shift When called from LinearEigensystem, contains the corresponding current
  !> eigenvalue estimates for each of the parameter vectors in the set. All other solvers
  !> should pass a vector of zeroes, which is the default if omitted.
  !> @param diagonals The diagonal elements of the underlying kernel. If passed, they will be used,
  !> otherwise the default preconditioner does nothing.
  !> @param range The range of the space for which actions should be computed. It's OK to provide also the values outside this range (which will happen in a multiprocessing context), but they will be ignored by the solver.
  subroutine precondition(this, action, shift, diagonals, range)
    class(Problem), intent(in) :: this
    double precision, intent(inout), dimension(:, :) :: action
    double precision, intent(in), dimension(:), optional :: shift
    double precision, intent(in), dimension(:), optional :: diagonals
    integer, dimension(2), intent(in) :: range
    double precision, parameter :: small = 1e-14
    if (present(diagonals)) then
      do i = lbound(action, 2), ubound(action, 2)
        if (present(shift)) then
          do j = range(1) + 1, range(2)
            action(j, i) = action(j, i) / (diagonals(j) + shift(i) + small)
          end do
        else
          do j = range(1) + 1, range(2)
            action(j, i) = action(j, i) / (diagonals(j) + small)
          end do
        end if
      end do
    end if
  end subroutine precondition

  !> @brief Calculate the residual vector. Used by non-linear solvers (NonLinearEquations,
  !> Optimize) only.
  !> @param parameters The trial solution for which the residual is to be calculated
  !> @param resid The residual vector
  !> @param range The range of the space for which actions should be computed. It's OK to provide also the values outside this range (which will happen in a multiprocessing context), but they will be ignored by the solver.
  !> @return In the case where the residual is an exact differential, the corresponding
  !> function value. Used by Optimize but not NonLinearEquations.
  function residual(this, parameters, residuals, range) result(value)
    class(Problem), intent(in) :: this
    integer, dimension(2), intent(in) :: range
    double precision :: value
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: residuals
    residuals = 0d0
  end function residual

  !> @brief Calculate the action of the kernel matrix on a set of parameters. Used by
  !> linear solvers, but not by the non-linear solvers (NonLinearEquations, Optimize).
  !> @param parameters The trial solutions for which the action is to be calculated
  !> @param actions The action vectors
  !> @param range The range of the space for which actions should be computed. It's OK to provide also the values outside this range (which will happen in a multiprocessing context), but they will be ignored by the solver.
  subroutine action(this, parameters, actions, range)
    class(Problem), intent(in) :: this
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: actions
    integer, dimension(2), intent(in) :: range
  end subroutine action

  !> @brief Provide the inhomogeneous part of one of the sets of linear equations. Implementation required only for linear equation solver.
  !> @param vector Will contain the requested RHS on exit
  !> @param instance Which equation set for which the RHS should be provided
  !> @param range The range of the space for which actions should be computed. It's OK to provide also the values outside this range (which will happen in a multiprocessing context), but they will be ignored by the solver.
  !> @return Whether the requested instance exists
  logical function RHS(this, vector, instance, range)
    class(Problem), intent(in) :: this
    double precision, intent(inout), dimension(:) :: vector
    integer, dimension(2), intent(in) :: range
    integer, intent(in) :: instance
    RHS = .false.
  end function RHS

  !> @brief Report progress at the end of each iteration, or at the end of the calculation
  !> @return .true. if the information was used, and therefore the caller should be silent. This is a complete implementation that can be used, but can be overridden if desired.
  logical function report(this, iteration, verbosity, errors, value, eigenvalues)
    class(Problem), intent(in) :: this
    integer, intent(in) :: iteration !< The iteration number if positive, or zero indicating successful convergence, or a negative number indicating failure to converge
    integer, intent(in) :: verbosity !< Expected interpretation:
    !< - 0 or less, nothing should be printed
    !< - 1 nothing should be printed if iteration > 0
    !< - 2 or more show progress in each iteration
    double precision, intent(in), dimension(:) :: errors !< The current residual norm for each solution
    double precision, intent(in), optional :: value !< In the case of optimisation only, the current objective function value
    double precision, dimension(:), intent(in), optional :: eigenvalues !< In the case of eigenproblem only, the current eigenvalues
    if ((iteration.le.0 .and. verbosity.ge.1) .or. verbosity.ge.2) then
      if (iteration.gt. 0 .and. verbosity.ge.2) then
        write (6, '(A,I3,1X,A,(T32,10F7.2))') 'Iteration', iteration, 'log10(|residual|)=', log10(errors)
      else if (iteration.eq.0) then
        write (6, '(A,(T32,10F7.2))') 'Converged,   log10(|residual|)=', log10(errors)
      else
        write (6, '(A,(T32,10F7.2))') 'Unconverged, log10(|residual|)=', log10(errors)
      end if
      if (present(value)) then
        write (6, *) 'Objective function value ', value
      end if
      if (present(eigenvalues)) then
        write (6, *) 'Eigenvalues ', eigenvalues
      end if
    end if
    report = .true.
  end function report

  !> @brief Calculate the representation of the kernel matrix in the P space. Implementation required only for linear hermitian problems for which P-space acceleration is wanted.
  function pp_action_matrix(this) result(matrix)
    class(Problem), intent(in) :: this
    double precision, dimension(:, :), allocatable :: matrix
    allocate(matrix(this%p_space%size, this%p_space%size))
    if (this%p_space%size.le.0) return
    error stop 'P-space unavailable: unimplemented pp_action_matrix() in Problem class'
!    do i = 1, this%p_space%size
!      do j = 1, this%p_space%size
!        matrix(i, j) = 0d0
!        do ic = this%p_space%offsets(i - 1) + 1, this%p_space%offsets(i)
!          do jc = this%p_space%offsets(j - 1) + 1, this%p_space%offsets(j)
!            matrix(i, j) = matrix(i, j) + this%matrix(this%p_space%indices(ic), this%p_space%indices(jc)) * this%p_space%coefficients(ic) * this%p_space%coefficients(jc)
!          end do
!        end do
!      end do
!    end do
  end function pp_action_matrix

  !> @brief Calculate the action of the kernel matrix on a set of vectors in the P space. Implementation required only for linear hermitian problems for which P-space acceleration is wanted.
  !> @param p_coefficients The projection of the vectors onto to the P space
  !> @param actions On exit, the computed action has been added to the original contents
  !> @param range The range of the full space for which actions should be computed.
  subroutine p_action(this, p_coefficients, actions, range)
    class(Problem), intent(in) :: this
    double precision, dimension(:, :), intent(in) :: p_coefficients
    double precision, dimension(:, :), intent(inout) :: actions
    integer, dimension(2), intent(in) :: range
    if (this%p_space%size.le.0) return
    error stop 'P-space unavailable: unimplemented p_action() in Problem class'
    !    do i = lbound(actions, 2), ubound(actions, 2)
    !      do k = 1, this%p_space%size
    !        do kc = this%p_space%offsets(k - 1) + 1, this%p_space%offsets(k)
    !          do j = range(1) + 1, range(2)
    !            actions(j, i) = actions(j, i) + this%matrix(j, this%p_space%indices(kc)) * this%p_space%coefficients(kc) * p_coefficients(k, i)
    !          end do
    !        end do
    !      end do
    !    end do
  end subroutine p_action

  !> @brief Provide values of R vectors for testing the problem class.
  !> For use in a non-linear solver, the first vector (instance=0) should be a reference point, and the remainder
  !> (instance>0) should be close to it, such that meaningful numerical differentation can be done to test the residual
  !> function.
  !> @param instance
  !> @param parameters
  !> @return true if a vector has been provided
  logical function test_parameters(instance, parameters)
    integer, intent(in) :: instance
    double precision, dimension(:), intent(inout) :: parameters
    test_parameters = .false.
  end function test_parameters

end module Iterative_Solver_Problem