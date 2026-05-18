! Proxy benchmark for scilib-accel data-movement work.
!
! Reproduces three behaviours of quick-test/run2.sh that make the dgemm
! single-shot micro-test a bad predictor:
!   1. multi-rank GPU contention (launch via mpirun -np N)
!   2. alloc/BLAS/dealloc cycle so each iteration sees fresh DRAM pages
!   3. mix of zgemm + ztrsm with the same buffer-size families as MuST/LSMS
!
! Default sizes mirror what was profiled out of MuST CoCrFeMnNi:
!     zgemm: m=n=1978, k=256, lda=ldb=ldc=6750  -> A:26MB B,C:204MB
!     ztrsm: m=2234, n=4516, lda=ldb=6750       -> A:241MB B:487MB
!
! Usage:
!   echo "<niter>" | mpirun --mca coll ^hcoll -np 28 -map-by node:PE=2 ./proxy.x
! or rely on defaults:
!   mpirun --mca coll ^hcoll -np 28 -map-by node:PE=2 ./proxy.x
!
! Each rank prints its wall time; the run script aggregates and prints the
! max-over-ranks (which is what app wall-clock looks like).

program proxy
    implicit none
    integer, parameter :: dp = kind(0.0d0)

    ! sizes from MuST profile
    integer, parameter :: lda      = 6750
    integer, parameter :: k_gemm   = 256
    integer, parameter :: m_gemm   = 1978
    integer, parameter :: n_gemm   = 1978
    integer, parameter :: m_trsm   = 2234
    integer, parameter :: n_trsm   = 4516

    complex(dp), allocatable :: A(:,:), B(:,:), C(:,:)
    complex(dp), allocatable :: At(:,:), Bt(:,:)
    complex(dp) :: alpha, beta
    integer :: niter, iter, i, j, ios
    integer(8) :: c0, c1, crate
    real(dp) :: total

    alpha = (1.0_dp, 0.0_dp)
    beta  = (0.0_dp, 0.0_dp)

    ! Read niter from stdin, default 3 (matches MuST's 3 outer iterations)
    niter = 3
    read(*, *, iostat=ios) niter
    if (ios /= 0) niter = 3

    call system_clock(c0, crate)

    do iter = 1, niter
        ! ---- zgemm pattern: fresh small/medium buffers ----
        allocate(A(lda, k_gemm))
        allocate(B(lda, n_gemm))
        allocate(C(lda, n_gemm))
        A = (2.0_dp, 0.5_dp)
        B = (0.5_dp, 0.1_dp)
        C = (0.0_dp, 0.0_dp)
        call zgemm('N', 'N', m_gemm, n_gemm, k_gemm, &
                   alpha, A, lda, B, lda, beta, C, lda)
        deallocate(A, B, C)

        ! ---- ztrsm pattern: fresh large buffers (the dominant cost) ----
        allocate(At(lda, m_trsm))
        allocate(Bt(lda, n_trsm))
        ! Make At a non-singular upper-triangular block (with strong diag)
        do j = 1, m_trsm
            do i = 1, m_trsm
                if (i < j) then
                    At(i, j) = (0.5_dp, 0.1_dp)
                else if (i == j) then
                    At(i, j) = (2.0_dp, 0.0_dp)
                else
                    At(i, j) = (0.0_dp, 0.0_dp)
                end if
            end do
        end do
        Bt = (1.0_dp, 0.3_dp)
        call ztrsm('L', 'U', 'N', 'N', m_trsm, n_trsm, &
                   alpha, At, lda, Bt, lda)
        deallocate(At, Bt)
    end do

    call system_clock(c1)
    total = real(c1 - c0, dp) / real(crate, dp)

    write(*, '("proxy rank time (s): ", F8.3)') total
end program proxy
