program test_dgemm
    implicit none

    ! Declare variables
    integer :: m, n, k, lda, ldb, ldc, Niter, i, j
    real(8), allocatable :: a(:,:), b(:,:), c(:,:)
    character :: transa, transb
    real(8) :: alpha, beta, start_time, end_time, total_time, min_time, avg_time
    REAL ::  start, finish, dtime
    real(8) :: check

    integer :: num_args
    character(len=32) :: arg
    logical :: full_input = .false.

    ! Default values
    transa = 'N'
    transb = 'N'
    alpha = 1.0
    beta = 0.0

    ! Check for command-line argument
    num_args = command_argument_count()
    if (num_args > 0) then
        call get_command_argument(1, arg)
        if (arg == "-f") full_input = .true.
    end if

    ! Get input based on full_input flag
    if (full_input) then
        write(*,*) "Enter the input parameters in the order: transa, transb, m, n, k, alpha, lda, ldb, beta, ldc, Niter"
        write(*,*) "e.g.: T N 32 2400 93536 1.0 93536 93536 0.0 32 11"
        read(*,*) transa, transb, m, n, k, alpha, lda, ldb, beta, ldc, Niter
    else
        write(*,*) "Run with -f to if you'd like to enter the full input arguments."
        write(*,*) "Enter m, n, k niter:"
        read(*,*) m, n, k, Niter
        lda = m
        ldb = k
        ldc = m
    end if


    ! Allocate memory for matrices
    if (transa == 'N' .or. transa == 'n') then
      allocate(a(lda,k))
    else
      allocate(a(lda,m))
    end if

    if (transb == 'N' .or. transb == 'n') then
      allocate(b(ldb,n))
    else
      allocate(b(ldb,k))
    end if

    allocate(c(ldc,n))

    ! Initialize matrices with random values
   ! call random_number(a)
   ! call random_number(b)
   ! call random_number(c)
    a = 2.0d0
    b = 0.5d0
    c = 0.0d0

    ! Print the input matrices
    !write(*,*) "Matrix A:"
    !call print_matrix(a, lda, k)
    !write(*,*) "Matrix B:"
    !call print_matrix(b, ldb, n)
    !write(*,*) "Matrix C:"
    !call print_matrix(c, ldc, n)

    total_time = 0.0
    min_time = 1.0e10
    avg_time = 0.0d0

    ! Loop over dgemm call Niter times
    do i = 1, Niter
        call cpu_time(start)
        call dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
          call cpu_time(finish)
          dtime = finish - start
          write(*,'(A20 F20.6)') "dgemm runtime(s):", dtime
          if(min_time > dtime ) min_time = dtime
          if(i>1) avg_time = avg_time + dtime
    end do
    if (Niter>1) avg_time = avg_time / (Niter - 1)
    write(*, '("Min dgemm runtime (s):", F20.6)')  min_time 
    write(*, '("Avg dgemm runtime (s):", F20.6, " (excl. 1st run)")')  avg_time 


    ! Print the result matrix C
!   write(*,*) "Matrix C after dgemm:"
!   call print_matrix(c, ldc, n)

    check=0.0d0
!$omp parallel do reduction(+:check) private(i)
    do j=1, n
      do i=1, m
        check=check+c(i,j)
      enddo
    enddo 
 !$omp end parallel do
    check=check/m/n
    if ( abs(beta)>1.0e-8) check = check / Niter
    WRITE(*,'("# Result check" F15.2 I10)') check, k


    deallocate(a, b, c)

contains

    subroutine print_matrix(a, lda, n)
        real(8), intent(in) :: a(:,:)
        integer, intent(in) :: lda, n
        integer :: i, j

        do i = 1, size(a, 1)
            write(*,*) (a(i,j), j=1,n)
        end do
    end subroutine print_matrix

end program test_dgemm
