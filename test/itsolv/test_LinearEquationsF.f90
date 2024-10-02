module mod_linear_problem
  use Iterative_Solver_Matrix_Problem
  type, extends(matrix_Problem) :: linear_problem
    double precision, dimension(:,:), pointer :: rhss
    contains
    procedure, pass :: RHS
  end type linear_problem
contains
  logical function RHS(this, vector, instance, range)
    class(linear_problem), intent(in) :: this
    double precision, intent(inout), dimension(:) :: vector
    integer, intent(in) :: instance
    integer, dimension(2), intent(in) :: range
    RHS = .false.
    if (instance.lt.lbound(this%rhss,2).or.instance.gt.ubound(this%rhss,2)) return
    RHS = .true.
    vector(range(1)+1:range(2)) = this%rhss(range(1)+1:range(2),instance)
  end function RHS

end module mod_linear_problem
function test_LinearEquationsF(matrix, rhs, n, np, nroot, hermitian, augmented_hessian) BIND(C)
  use iso_c_binding
  use Iterative_Solver
  implicit none
  integer(c_int) test_LinearEquationsF
  integer(c_size_t), intent(in), value :: n, np, nroot
  integer(c_int), intent(in), value :: hermitian
  double precision, intent(in), dimension(n, n), target :: matrix
  double precision, intent(in), dimension(n, nroot), target :: rhs
  double precision, intent(in), value :: augmented_hessian
  double precision, dimension(n, nroot) :: c, g
  double precision :: error
  integer :: nwork, i, j, k
  integer, dimension(nroot) :: guess
  double precision :: guess_value
  double precision, parameter :: thresh = 1d-10

  if (np .gt. 0) return
  !  write (6, *) 'test_linearEquationsF ', hermitian, n, nroot
  !    write (6, *) 'matrix'
  !    do i = 1, n
  !      write (6, *) matrix(:, i)
  !    end do
  !    write (6, *) 'rhs'
  !    do i = 1, nroot
  !      write (6, *) rhs(:, i)
  !    end do
  !  call flush(6)
  !  options->convergence_threshold = 1.0e-8;
  !  //    options->norm_thresh = 1.0e-14;
  !  //    options->svd_thresh = 1.0e-10;
  !  options->max_size_qspace = std::max(std::min(n, 6 * nroot), std::min(n, std::min(1000, 6 * nroot)) - np);
  !  options->reset_D = 8;
  call Iterative_Solver_Linear_Equations_Initialize(n, nroot, rhs, augmented_hessian = augmented_hessian, &
      hermitian = hermitian.ne.0, &
      thresh = thresh, thresh_value = 1d50)
  nwork = nroot
  c = 0
  do k = 1, nroot
    do i = k, n
      c(i, k) = -rhs(i, k) / matrix(i, i)
    end do
  end do
  !  write (6, *) 'guess'
  !  do i = 1, nroot
  !    write (6, *) c(:, i)
  !  end do
  call flush(6)
  do i = 1, 1000
    g = matmul(matrix, c)
    nwork = Iterative_Solver_Add_Vector(c, g);
    !    write (6, *) 'nwork after add_vector ', nwork
    if (nwork.le.0) exit
    do k = 1, nwork
      do j = 1, n
        g(j, k) = -g(j, k) / matrix(j, j)
      end do
    end do
    nwork = Iterative_Solver_End_Iteration(c, g);
    !    write (6, *) 'nwork after end_iteration ', nwork, Iterative_Solver_Errors(); call flush(6)
    if (nwork.le.0) exit
  end do
  call Iterative_Solver_Solution([(i, i = 1, nroot)], c, g)
  error = 0
  do i = 1, nroot
    !    write (6, *) 'solution ', i, c(:, i)
    !    write (6, *) 'reported residual ', i, g(:, i)
    error = max(error, sqrt(dot_product(g(:, i), g(:, i))))
    !    write (6, *) 'reported residual length ', sqrt(dot_product(g(:, i), g(:, i)))
    if (.true.) then ! TODO really check this, as sometimes it differs
      g(:, i) = matmul(matrix, c(:, i)) - rhs(:, i)
!      write (6, *) 'calculated residual ', i, g(:, i)
      error = max(error, sqrt(dot_product(g(:, i), g(:, i))))
!      write (6, *) 'calculated residual length ', sqrt(dot_product(g(:, i), g(:, i)))
    end if
  end do
  test_LinearEquationsF = 1
  if (error.gt.1d-4) then
    write (6, *) 'test_linearEquationsF has failed ', error
    test_LinearEquationsF = 0
  end if
  call Iterative_Solver_Finalize
  call simplified_solver
  call flush(6)
  return
contains
  subroutine simplified_solver
    use mod_linear_problem, only : linear_problem
    type(linear_problem) :: prob
    call prob%attach(matrix, rhs)
    call Solve_Linear_Equations(c, g, prob, augmented_hessian = augmented_hessian, &
        hermitian = hermitian.ne.0, &
        thresh = thresh, thresh_value = 1d50, verbosity=2)
    if (.not. Iterative_Solver_Converged()) then
      error stop  '!!! failure !!!'
    end if
    call Iterative_Solver_Finalize
  end subroutine simplified_solver
end function test_LinearEquationsF