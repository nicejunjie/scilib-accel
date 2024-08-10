
#define func_group "BLAS"
#include "global.h"
#include "complex.h"
#include <math.h>
#include <stdlib.h>
#include <init.h>
#include "utils.h"
#include "stdio.h"


void mysgbmv(const char *trans, const int *m, const int *n, const int *kl, const int *ku, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = sgbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydgbmv(const char *trans, const int *m, const int *n, const int *kl, const int *ku, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = dgbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mycgbmv(const char *trans, const int *m, const int *n, const int *kl, const int *ku, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = cgbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzgbmv(const char *trans, const int *m, const int *n, const int *kl, const int *ku, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = zgbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mysgemv(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = sgemv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydgemv(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = dgemv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mycgemv(const char *trans, const int *m, const int *n, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = cgemv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzgemv(const char *trans, const int *m, const int *n, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = zgemv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mysger(const int *m, const int *n, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = sger;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(m, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydger(const int *m, const int *n, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = dger;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(m, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mycgerc(const int *m, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = cgerc;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(m, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzgerc(const int *m, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = zgerc;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(m, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mycgeru(const int *m, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = cgeru;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(m, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzgeru(const int *m, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = zgeru;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(m, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mychbmv(const char *uplo, const int *n, const int *k, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = chbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzhbmv(const char *uplo, const int *n, const int *k, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = zhbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mychemv(const char *uplo, const int *n, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = chemv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzhemv(const char *uplo, const int *n, const void *alpha, const void *a, const int *lda, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = zhemv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mycher(const char *uplo, const int *n, const float *alpha, const void *x, const int *incx, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = cher;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzher(const char *uplo, const int *n, const float *alpha, const void *x, const int *incx, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = zher;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mycher2(const char *uplo, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = cher2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzher2(const char *uplo, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = zher2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mychpmv(const char *uplo, const int *n, const void *alpha, const void *ap, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = chpmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, ap, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzhpmv(const char *uplo, const int *n, const void *alpha, const void *ap, const void *x, const int *incx, const void *beta, void *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = zhpmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, ap, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mychpr(const char *uplo, const int *n, const float *alpha, const void *x, const int *incx, void *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = chpr;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzhpr(const char *uplo, const int *n, const float *alpha, const void *x, const int *incx, void *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = zhpr;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mychpr2(const char *uplo, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = chpr2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myzhpr2(const char *uplo, const int *n, const void *alpha, const void *x, const int *incx, const void *y, const int *incy, void *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = zhpr2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myssbmv(const char *uplo, const int *n, const int *k, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = ssbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydsbmv(const char *uplo, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = dsbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mysspmv(const char *uplo, const int *n, const float *alpha, const float *ap, const float *x, const int *incx, const float *beta, float *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = sspmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, ap, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydspmv(const char *uplo, const int *n, const double *alpha, const double *ap, const double *x, const int *incx, const double *beta, double *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = dspmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, ap, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mysspr(const char *uplo, const int *n, const float *alpha, const float *x, const int *incx, float *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = sspr;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydspr(const char *uplo, const int *n, const double *alpha, const double *x, const int *incx, double *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = dspr;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mysspr2(const char *uplo, const int *n, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = sspr2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydspr2(const char *uplo, const int *n, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *ap)
{
    void (*orig_f)() = NULL;
    enum findex fi = dspr2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, ap);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myssymv(const char *uplo, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = ssymv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydsymv(const char *uplo, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy)
{
    void (*orig_f)() = NULL;
    enum findex fi = dsymv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myssyr(const char *uplo, const int *n, const float *alpha, const float *x, const int *incx, float *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = ssyr;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydsyr(const char *uplo, const int *n, const double *alpha, const double *x, const int *incx, double *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = dsyr;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myssyr2(const char *uplo, const int *n, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = ssyr2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydsyr2(const char *uplo, const int *n, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *a, const int *lda)
{
    void (*orig_f)() = NULL;
    enum findex fi = dsyr2;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, n, alpha, x, incx, y, incy, a, lda);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mystbmv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const float *a, const int *lda, float *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = stbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydtbmv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const double *a, const int *lda, double *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = dtbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myctbmv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ctbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myztbmv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ztbmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mystbsv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const float *a, const int *lda, float *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = stbsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydtbsv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const double *a, const int *lda, double *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = dtbsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myctbsv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ctbsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myztbsv(const char *uplo, const char *trans, const char *diag, const int *n, const int *k, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ztbsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, k, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mystpmv(const char *uplo, const char *trans, const char *diag, const int *n, const float *ap, float *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = stpmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydtpmv(const char *uplo, const char *trans, const char *diag, const int *n, const double *ap, double *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = dtpmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myctpmv(const char *uplo, const char *trans, const char *diag, const int *n, const void *ap, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ctpmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myztpmv(const char *uplo, const char *trans, const char *diag, const int *n, const void *ap, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ztpmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mystpsv(const char *uplo, const char *trans, const char *diag, const int *n, const float *ap, float *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = stpsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydtpsv(const char *uplo, const char *trans, const char *diag, const int *n, const double *ap, double *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = dtpsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myctpsv(const char *uplo, const char *trans, const char *diag, const int *n, const void *ap, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ctpsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myztpsv(const char *uplo, const char *trans, const char *diag, const int *n, const void *ap, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ztpsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, ap, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mystrmv(const char *uplo, const char *trans, const char *diag, const int *n, const float *a, const int *lda, float *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = strmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydtrmv(const char *uplo, const char *trans, const char *diag, const int *n, const double *a, const int *lda, double *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = dtrmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myctrmv(const char *uplo, const char *trans, const char *diag, const int *n, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ctrmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myztrmv(const char *uplo, const char *trans, const char *diag, const int *n, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ztrmv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mystrsv(const char *uplo, const char *trans, const char *diag, const int *n, const float *a, const int *lda, float *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = strsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void mydtrsv(const char *uplo, const char *trans, const char *diag, const int *n, const double *a, const int *lda, double *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = dtrsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myctrsv(const char *uplo, const char *trans, const char *diag, const int *n, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ctrsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

void myztrsv(const char *uplo, const char *trans, const char *diag, const int *n, const void *a, const int *lda, void *x, const int *incx)
{
    void (*orig_f)() = NULL;
    enum findex fi = ztrsv;
    double local_time=0.0;
    if (!orig_f) orig_f = farray[fi].fptr;
    local_time=mysecond();
    orig_f(uplo, trans, diag, n, a, lda, x, incx);
    local_time=mysecond()-local_time;
    farray[fi].t0 += local_time;
    farray[fi].t1 += local_time;

    return;
}

