function test_LinearEigensystemF(matrix, n, np, nroot, hermitian, expected_eigenvalues) BIND(C)
  use iso_c_binding
  use Iterative_Solver
  implicit none
  integer(c_int) :: test_LinearEigensystemF
  integer(c_size_t), intent(in), value :: n, np, nroot
  integer(c_int), intent(in), value :: hermitian
  double precision, intent(in), dimension(n, n) :: matrix
  double precision, intent(in), dimension(nroot) :: expected_eigenvalues

  double precision, dimension(n, nroot) :: c, g
  double precision, allocatable, dimension(:) :: eigs
  double precision :: eigk, error
  integer :: nwork, i, j, k, alloc_stat
  integer, dimension(nroot) :: guess
  double precision :: guess_value
  double precision, parameter :: thresh = 1d-8

  test_LinearEigensystemF = 1
  if (np .gt. 0) return
  write (6, *) 'test_linearEigensystemF ', hermitian
  call Iterative_Solver_Linear_Eigensystem_Initialize(int(n), int(nroot), verbosity = 2, hermitian = hermitian.ne.0, &
      thresh = thresh)
  nwork = nroot
  c = 0
  do i = 1, nroot
    guess_value = 1d50
    guess(i) = 1
    do k = 1, n
      if (matrix(k, k) .lt. guess_value) then
        do j = 1, i - 1
          if (guess(j).eq.k) goto 44
        end do
        guess(i) = k
        guess_value = matrix(k, k)
      end if
      44 continue
    end do
    c(guess(i), i) = 1
!    write (6, *) 'guess ', guess(i), matrix(guess(i), guess(i))
  end do
  do i = 1, 1000
    g = matmul(matrix, c)
    nwork = Iterative_Solver_Add_Vector(c, g);
    !write (6, *) 'errors after Add_Vector', Iterative_Solver_Errors()
    if (nwork.le.0) exit
    allocate(eigs(nwork), stat=alloc_stat)
    eigs = Iterative_Solver_Working_Set_Eigenvalues(nwork)
    !write (6, *) 'working set eigenvalues ', eigs
    do k = 1, nwork
      eigk = eigs(k)
      do j = 1, n
        g(j, k) = -g(j, k) / (matrix(j, j) + 1e-12 - eigk)
      end do
    end do
    deallocate(eigs)
    nwork = Iterative_Solver_End_Iteration(c, g);
    !write (6, *) 'errors after End_Iteration', Iterative_Solver_Errors()
    if (nwork.le.0) exit
  end do
  write (6, *) 'errors ', Iterative_Solver_Errors()
  allocate(eigs(nroot), stat=alloc_stat)
  eigs = Iterative_Solver_Eigenvalues()
  error = sqrt(dot_product(eigs - expected_eigenvalues, eigs - expected_eigenvalues))
  if (error.gt.thresh) then
    write (6, *) 'test_linearEigensystemF eigenvalues ', eigs
    write (6, *) 'test_linearEigensystemF expected eigenvalues ', expected_eigenvalues
    write (6, *) 'difference ', eigs - expected_eigenvalues
    write (6, *) 'difference ', sqrt(dot_product(eigs - expected_eigenvalues, eigs - expected_eigenvalues))
    test_LinearEigensystemF = 0
  end if
  deallocate(eigs)
  call Iterative_Solver_Finalize

end function test_LinearEigensystemF
!> A solver created and finalised while another is active must not disturb the sizes the Fortran interface caches
!> for the enclosing solver
function test_nested_finalizeF() BIND(C)
  use iso_c_binding
  use Iterative_Solver
  implicit none
  integer(c_int) :: test_nested_finalizeF
  integer, parameter :: n_outer = 20, nroot_outer = 3, n_inner = 7
  test_nested_finalizeF = 1
  call Iterative_Solver_Linear_Eigensystem_Initialize(n_outer, nroot_outer)
  call Iterative_Solver_DIIS_Initialize(n_inner)
  call Iterative_Solver_Finalize
  if (size(Iterative_Solver_Errors()) .ne. nroot_outer) test_nested_finalizeF = 0
  if (size(Iterative_Solver_Eigenvalues()) .ne. nroot_outer) test_nested_finalizeF = 0
  call Iterative_Solver_Finalize
end function test_nested_finalizeF

!> Each process requests its own (deliberately uneven) range, and computes actions and preconditioned residuals only
!> inside it, filling the rest with garbage that the solver must ignore
function test_supplied_rangeF(matrix, n, nroot, expected_eigenvalues) BIND(C)
  use iso_c_binding
  use Iterative_Solver
  implicit none
  integer(c_int) :: test_supplied_rangeF
  integer(c_size_t), intent(in), value :: n, nroot
  double precision, intent(in), dimension(n, n) :: matrix
  double precision, intent(in), dimension(nroot) :: expected_eigenvalues
  double precision, dimension(n, nroot) :: c, g
  double precision, dimension(nroot) :: eigs
  double precision, parameter :: garbage = 1d10
  integer :: range(2), requested(2), rank, nproc, nwork, iter, k, j
  test_supplied_rangeF = 1
  rank = int(mpi_rank_global())
  nproc = int(mpi_size_global())
  if (nproc .eq. 1) then
    requested = [0, int(n)]
  else if (rank .eq. 0) then
    requested = [0, int(n) / 5]
  else
    requested(1) = int(n) / 5 + (rank - 1) * ((int(n) - int(n) / 5) / (nproc - 1))
    requested(2) = int(n) / 5 + rank * ((int(n) - int(n) / 5) / (nproc - 1))
    if (rank .eq. nproc - 1) requested(2) = int(n)
  end if
  range = requested
  call Iterative_Solver_Linear_Eigensystem_Initialize(int(n), int(nroot), thresh = 1d-10, hermitian = .true., &
      range = range)
  if (any(range .ne. requested)) then
    write (6, *) 'test_supplied_rangeF: requested range ', requested, ' but got ', range
    test_supplied_rangeF = 0
  end if
  c = 0
  do k = 1, int(nroot)
    c(k, k) = 1
  end do
  do iter = 1, 200
    g = garbage
    g(range(1) + 1:range(2), :) = matmul(matrix(range(1) + 1:range(2), :), c)
    nwork = Iterative_Solver_Add_Vector(c, g)
    if (nwork .le. 0) exit
    eigs(:nwork) = Iterative_Solver_Working_Set_Eigenvalues(nwork)
    do k = 1, nwork
      do j = range(1) + 1, range(2)
        g(j, k) = -g(j, k) / (matrix(j, j) - eigs(k) + 1d-12)
      end do
    end do
    nwork = Iterative_Solver_End_Iteration(c, g)
    if (nwork .le. 0) exit
  end do
  eigs = Iterative_Solver_Eigenvalues()
  if (maxval(abs(eigs - expected_eigenvalues)) .gt. 1d-8) then
    write (6, *) 'test_supplied_rangeF: eigenvalues ', eigs, ' expected ', expected_eigenvalues
    test_supplied_rangeF = 0
  end if
  call Iterative_Solver_Finalize
end function test_supplied_rangeF
