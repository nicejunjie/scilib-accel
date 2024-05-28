function mysecond()
  real(8):: mysecond
  mysecond=0.0
end function

PROGRAM blas_test

  implicit none
  
  INTEGER :: M, N, K
  REAL*8, ALLOCATABLE :: A(:,:), B(:,:), C(:,:)
  INTEGER :: i, Niter, j
  REAL*8 :: dtime, mysecond
  REAL*8 :: check
  REAL*8 :: alpha, beta
  
  ! Read matrix dimensions from user input
   READ(*,*) M, N, K, Niter
!  M=20816
!  N=2400
!  K=32
!  Niter=10
  
  ! Allocate memory for the matrices
  ALLOCATE(A(M, K), B(K, N), C(M, N))
  
  ! Initialize matrices A and B (for example)
  A = 2.0d0
  B = 0.5d0
  C = 0.0d0
! call random_number(A)
! call random_number(B)

 alpha=1.0d0
 beta=1.0d0
  
  do i=1, Niter
  dtime = mysecond()
!    print *, "iter=",i
    CALL DGEMM('N', 'N', M, N, K, alpha, A, M, B, K, beta, C, M)
  dtime = mysecond() - dtime
  write(*,'(A10 F20.6)') "runtime(s):", dtime
  end do
!  dtime = dtime / Niter

!$omp parallel do reduction(+:check) private(j)
  do i=1, M
    do j=1, N
      check=check+C(i,j)
    enddo
  enddo 
!$omp end parallel do
  check=check/M/N
  
  
  ! Display the result
  !WRITE(*,*) "Resultant Matrix C:"
  !DO i = 1, M
  !  WRITE(*, '(100F8.2)') (C(i, j), j = 1, N)
  !ENDDO
  WRITE(*,"(A20 F15.2 I10)") "# Result check", check, K*Niter
  
  
  ! Deallocate memory
  DEALLOCATE(A, B, C)
  
END PROGRAM blas_test

