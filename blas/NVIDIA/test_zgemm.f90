program zgemm_submatrix_both
    implicit none
    
    integer, parameter :: n = 4, m = 4, k = 4  ! Dimensions of matrices
    integer :: lda, ldb, ldc, i, j
    complex*16 :: alpha, beta
    complex*16 :: A(n, m), B(m, k), C(n, k)   ! Matrices A, B, and C
! Define the dimensions of the submatrices
    integer :: n_sub = 2, m_sub = 2, k_sub = 2
    integer :: offset_A = 1, offset_B = 1

    
    ! Initialize matrices A and B with some values
    do i = 1, n
        do j = 1, m
            A(i, j) = cmplx(i + j, 0)
        end do
    end do
    
    do i = 1, m
        do j = 1, k
            B(i, j) = cmplx(i - j, 0)
        end do
    end do

    ! Print matrix A
    print *, "Matrix A:"
    do i = 1, n
        write(*, '(5("(",f5.2,",",f5.2,")",:,1x))') (A(i, j), j = 1, m)
    end do
    print *  ! Print a new line after the matrix

    ! Print matrix B
    print *, "Matrix B:"
    do i = 1, m
        write(*, '(5("(",f5.2,",",f5.2,")",:,1x))') (B(i, j), j = 1, k)
    end do
    print *  ! Print a new line after the matrix

    
    ! Set the dimensions of the submatrices
    lda = n
    ldb = m
    ldc = n
    
    ! Set the scalar coefficients
    alpha = (1.0, 0.0)
    beta = (0.0, 0.0)
    
        
    ! Call zgemm to multiply the submatrices
    call zgemm('N', 'N', n_sub, k_sub, m_sub, alpha, A(2,2), &
               lda, B(2,2), ldb, beta, C(2,2), ldc)
    
    ! Print the result matrix C
    print *, "Result matrix C:"
    do i = 1, n
        write(*, '(5("(",f6.2,",",f6.2,")",:,1x))') (C(i, j), j = 1, k)
    end do
    print *  ! Print a new line after the matrix

    
end program zgemm_submatrix_both

