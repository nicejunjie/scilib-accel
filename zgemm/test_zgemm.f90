program zgemm_example

  implicit none

  integer :: m, n, k, niter
  complex*16, parameter :: alpha = (1.0, 0.0), beta = (1.0, 0.0)

  complex*16, allocatable :: A(:,:), B(:,:), C(:,:)
  real(8) :: mysecond
  real(8) :: t
  complex*16 :: check
  CHARACTER(len=32) :: arg

  integer :: i, j, l

  ! Read matrix sizes from command line
  if (command_argument_count() /= 4) then
    write(*,*) "Usage: zgemm_example <m> <n> <k> <niter>"
    stop
  else
    call get_command_argument(1, arg)
    read(arg,*) m
    call get_command_argument(2, arg)
    read(arg,*) n
    call get_command_argument(3, arg)
    read(arg,*) k
    call get_command_argument(4, arg)
    read(arg,*) niter
  end if


  ! Allocate matrices A, B, and C
  allocate(A(m, k), B(k, n), C(m, n))

  ! Initialize matrices A and B
!$omp parallel do private(j)
  do i = 1, m
    do j = 1, k
      A(i,j) = cmplx(2.0, 2.0)
    end do
  end do

!$omp parallel do private(j)
  do i = 1, k
    do j = 1, n
      B(i,j) = cmplx(0.25, 0.25)
    end do
  end do

!$omp parallel do private(j)
  do i = 1, m
    do j = 1, n
      C(i,j) = cmplx(0.0, 0.0)
    end do
  end do

  ! Call zgemm to multiply A and B, storing the result in C
  do l = 1, niter
    t = mysecond()
    call zgemm("N", "N", m, n, k, alpha, A, m, B, k, beta, C, m)
    t = mysecond() - t
    write(*, '(A20,F15.6)') 'zgemm time is', t
  end do
  
  check=0.0

!$omp parallel do reduction(+:check) private(j)
  do i = 1, n
     do j =1, m
         check = check + C(j,i)
     enddo
  enddo
  check = check / m / n / niter

  ! Print the result matrix C
  !do i = 1, 3
  !  write(*,'(6(F15.6 ))') (real(C(i,j)), aimag(C(i,j)), j=1, 3)
  !end do

  write(*,'(A8,2x,"("F15.6","F15.6")",2x,A2,2x,"("F15.6","F15.6")")') "check", check, "vs", cmplx(0.0,k)

  ! Deallocate matrices
  deallocate(A, B, C)

end program zgemm_example

