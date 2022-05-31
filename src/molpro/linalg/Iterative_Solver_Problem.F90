module Iterative_Solver_Problem

  private

  !> @brief Abstract class defining the problem-specific interface for the simplified solver
  !> interface to IterativeSolver
  !type, abstract, public :: Problem
  type, public :: Problem
  contains
    procedure, pass :: diagonals
    procedure, pass :: precondition
    procedure, pass :: residual
    procedure, pass :: action
  end type Problem

contains

  !> @brief Optionally provide the diagonal elements of the underlying kernel. If
  !> implemented and returning true, the provided diagonals will be used by
  !> IterativeSolver for preconditioning (and therefore the precondition() function does
  !> not need to be implemented), and, in the case of linear problems, for selection of
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
  !> the solution
  !> @param residual On entry, assumed to be the residual. On exit, the negative of the
  !> predicted step.
  !> @param shift When called from LinearEigensystem, contains the corresponding current
  !> eigenvalue estimates for each of the parameter vectors in the set. All other solvers
  !> should pass a vector of zeroes, which is the default if omitted.
  !> @param diagonals The diagonal elements of the underlying kernel. If passed, they will be used,
  !> otherwise the default preconditioner does nothing.
  subroutine precondition(this, action, shift, diagonals)
    class(Problem), intent(in) :: this
    double precision, intent(inout), dimension(:, :) :: action
    double precision, intent(in), dimension(:), optional :: shift
    double precision, intent(in), dimension(:), optional :: diagonals
    double precision, parameter :: small = 1e-14
    if (present(diagonals)) then
      do i = lbound(action, 2), ubound(action, 2)
        if (present(shift)) then
          do j = lbound(action, 1), ubound(action, 1)
            action(j, i) = action(j, i) / (diagonals(j) + shift(i) + small)
          end do
        else
          do j = lbound(action, 1), ubound(action, 1)
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
  !> @return In the case where the residual is an exact differential, the corresponding
  !> function value. Used by Optimize but not NonLinearEquations.
  function residual(this, parameters, residuals) result(value)
    class(Problem), intent(in) :: this
    double precision :: value
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: residuals
    residuals = 0d0
  end function residual

  !> @brief Calculate the action of the kernel matrix on a set of parameters. Used by
  !> linear solvers, but not by the non-linear solvers (NonLinearEquations, Optimize).
  !> @param parameters The trial solutions for which the action is to be calculated
  !> @param act The action vectors
  subroutine action(this, parameters, actions)
    class(Problem), intent(in) :: this
    double precision, intent(in), dimension(:, :) :: parameters
    double precision, intent(inout), dimension(:, :) :: actions
  end subroutine action

end module Iterative_Solver_Problem

module try_Iterative_Solver_Problem
  use Iterative_Solver_Problem
  type, extends(Problem) :: my_Problem

  end type my_Problem

contains
  subroutine try
    type(my_Problem) :: thing
  end subroutine try
end module try_Iterative_Solver_Problem