/***   gcc -O0 -pg baseline.c -o baseline    -lm
 *     gcc -O2 -pg baseline.c -o baseline_O2 -lm
 *     gcc -O3 -pg baseline.c -o baseline_O3 -lm
 *
 * Run:
 *     ./baseline                 # default: N = 512, runs naive + blocked
 *     ./baseline 1024 naive      # naive only
 *     ./baseline 1024 blocked    # blocked only
 *     ./baseline 512  blocksweep # blocked, sweeping block size 8..256
 *     ./baseline scaling         # N = 128,256,512,768,1024
 */

 #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_N     512
#define DEFAULT_BLOCK 64
#define SEED          42


// this fun is for time and this CLOCK_MONOTONIC is not affected by the system
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}




//allocation and initialisation    
static double *alloc_matrix(int n)
{
    double *m = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (m == NULL) {
        fprintf(stderr, "ERROR: could not allocate %d x %d matrix\n", n, n);
        exit(EXIT_FAILURE);
    }
    return m;
}
static void fill_random(double *m, int n)
{
    int i;
    int total = n * n;
    for (i = 0; i < total; i++) {
        /* values in [0,1) - small and bounded so the sums stay well behaved */
        m[i] = (double)rand() / (double)RAND_MAX;
    }
}

static void fill_zero(double *m, int n)
{
    memset(m, 0, (size_t)n * (size_t)n * sizeof(double));
}




//naive triple loop (i, j, k)  TC=O(n^3)
//  C= A*B
static void matmul_naive(const double *A, const double *B, double *C, int n)
{
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}




// blocked / tiled multiply 

static void multiply_block(const double *A, const double *B, double *C,
                           int n, int ii, int jj, int kk, int bs)
{
    int i, j, k;
    int i_max = (ii + bs < n) ? (ii + bs) : n;
    int j_max = (jj + bs < n) ? (jj + bs) : n;
    int k_max = (kk + bs < n) ? (kk + bs) : n;

    for (i = ii; i < i_max; i++) {
        for (j = jj; j < j_max; j++) {
            double sum = C[i * n + j];
            for (k = kk; k < k_max; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}


static void matmul_blocked(const double *A, const double *B, double *C,
                           int n, int bs)
{
    int ii, jj, kk;

    for (ii = 0; ii < n; ii += bs) {
        for (jj = 0; jj < n; jj += bs) {
            for (kk = 0; kk < n; kk += bs) {
                multiply_block(A, B, C, n, ii, jj, kk, bs);
            }
        }
    }
}



/* verification                                                        */

/* The two kernels must produce the same answer. Floating point addition
 * is not associative, so the order of summation differs slightly between
 * naive and blocked. We allow a small tolerance instead of exact equality.
 */
static int verify(const double *C1, const double *C2, int n)
{
    int i;
    int total = n * n;
    double worst = 0.0;

    for (i = 0; i < total; i++) {
        double d = C1[i] - C2[i];
        if (d < 0.0) d = -d;
        if (d > worst) worst = d;
    }

    printf("  verification: largest difference = %.3e  ->  %s\n",
           worst, (worst < 1e-9) ? "MATCH" : "MISMATCH");

    return (worst < 1e-9);
}

/* ------------------------------------------------------------------ */
/* results.csv                                                         */
/* ------------------------------------------------------------------ */

static void log_result(const char *kernel, int n, int bs,
                       double seconds, double gflops)
{
    FILE *f;
    int write_header = 0;

    f = fopen("results.csv", "r");
    if (f == NULL) {
        write_header = 1;
    } else {
        fclose(f);
    }

    f = fopen("results.csv", "a");
    if (f == NULL) {
        fprintf(stderr, "WARNING: could not open results.csv for writing\n");
        return;
    }

    if (write_header) {
        fprintf(f, "kernel,n,block_size,threads,seconds,gflops\n");
    }

    /* threads is always 1 here. The parallel version will write the same
     * columns so one plotting script handles both files. */
    fprintf(f, "%s,%d,%d,1,%.6f,%.4f\n", kernel, n, bs, seconds, gflops);
    fclose(f);
}

/* Matrix multiply does 2*n^3 floating point operations:
 * n^3 multiplies and n^3 additions. */
static double compute_gflops(int n, double seconds)
{
    double ops = 2.0 * (double)n * (double)n * (double)n;
    if (seconds <= 0.0) return 0.0;
    return ops / seconds / 1e9;
}


/* drivers                                                             */


static double run_naive(int n, double *A, double *B, double *C)
{
    double t0, t1, elapsed;

    fill_zero(C, n);
    t0 = now_seconds();
    matmul_naive(A, B, C, n);
    t1 = now_seconds();
    elapsed = t1 - t0;

    printf("  naive    n=%4d              %10.6f s   %7.3f GFLOP/s\n",
           n, elapsed, compute_gflops(n, elapsed));
    log_result("naive", n, 0, elapsed, compute_gflops(n, elapsed));

    return elapsed;
}

static double run_blocked(int n, int bs, double *A, double *B, double *C)
{
    double t0, t1, elapsed;

    fill_zero(C, n);
    t0 = now_seconds();
    matmul_blocked(A, B, C, n, bs);
    t1 = now_seconds();
    elapsed = t1 - t0;

    printf("  blocked  n=%4d  block=%4d  %10.6f s   %7.3f GFLOP/s\n",
           n, bs, elapsed, compute_gflops(n, elapsed));
    log_result("blocked", n, bs, elapsed, compute_gflops(n, elapsed));

    return elapsed;
}

static void mode_both(int n, int bs)
{
    double *A, *B, *C1, *C2;
    double t_naive, t_blocked;

    A  = alloc_matrix(n);
    B  = alloc_matrix(n);
    C1 = alloc_matrix(n);
    C2 = alloc_matrix(n);

    srand(SEED);
    fill_random(A, n);
    fill_random(B, n);

    printf("\nSequential matrix multiplication, n = %d\n", n);
    printf("-----------------------------------------------------------\n");

    /* naive first, into C1 */
    fill_zero(C1, n);
    t_naive = now_seconds();
    matmul_naive(A, B, C1, n);
    t_naive = now_seconds() - t_naive;
    printf("  naive    n=%4d              %10.6f s   %7.3f GFLOP/s\n",
           n, t_naive, compute_gflops(n, t_naive));
    log_result("naive", n, 0, t_naive, compute_gflops(n, t_naive));

    /* blocked second, into C2 */
    fill_zero(C2, n);
    t_blocked = now_seconds();
    matmul_blocked(A, B, C2, n, bs);
    t_blocked = now_seconds() - t_blocked;
    printf("  blocked  n=%4d  block=%4d  %10.6f s   %7.3f GFLOP/s\n",
           n, bs, t_blocked, compute_gflops(n, t_blocked));
    log_result("blocked", n, bs, t_blocked, compute_gflops(n, t_blocked));

    verify(C1, C2, n);

    if (t_blocked > 0.0) {
        printf("  blocking speedup: %.2fx  (same arithmetic, better locality)\n",
               t_naive / t_blocked);
    }

    free(A); free(B); free(C1); free(C2);
}

static void mode_single(int n, int bs, const char *which)
{
    double *A, *B, *C;

    A = alloc_matrix(n);
    B = alloc_matrix(n);
    C = alloc_matrix(n);

    srand(SEED);
    fill_random(A, n);
    fill_random(B, n);

    printf("\nSequential matrix multiplication, n = %d\n", n);
    printf("-----------------------------------------------------------\n");

    if (strcmp(which, "naive") == 0) {
        run_naive(n, A, B, C);
    } else {
        run_blocked(n, bs, A, B, C);
    }

    free(A); free(B); free(C);
}

/* Sweep the block size at fixed n. This graph alone is worth a slide:
 * the arithmetic never changes, only the block size, and the runtime
 * still swings by a large factor. That is pure locality. */
static void mode_blocksweep(int n)
{
    int sizes[] = {8, 16, 32, 64, 128, 256};
    int count = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int idx;
    double *A, *B, *C;

    A = alloc_matrix(n);
    B = alloc_matrix(n);
    C = alloc_matrix(n);

    srand(SEED);
    fill_random(A, n);
    fill_random(B, n);

    printf("\nBlock size sweep, n = %d\n", n);
    printf("-----------------------------------------------------------\n");

    for (idx = 0; idx < count; idx++) {
        if (sizes[idx] > n) continue;
        run_blocked(n, sizes[idx], A, B, C);
    }

    free(A); free(B); free(C);
}

/* Input size study, same idea as the scaling mode in Tutorial 0. */
static void mode_scaling(int bs)
{
    int sizes[] = {128, 256, 512, 768, 1024};
    int count = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int idx;

    printf("\nInput size scaling study (block size = %d)\n", bs);
    printf("-----------------------------------------------------------\n");

    for (idx = 0; idx < count; idx++) {
        int n = sizes[idx];
        double *A = alloc_matrix(n);
        double *B = alloc_matrix(n);
        double *C = alloc_matrix(n);

        srand(SEED);
        fill_random(A, n);
        fill_random(B, n);

        run_naive(n, A, B, C);
        run_blocked(n, bs, A, B, C);

        free(A); free(B); free(C);
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    printf("usage:\n");
    printf("  %s                      n=%d, naive + blocked\n", prog, DEFAULT_N);
    printf("  %s <n>                  n given, naive + blocked\n", prog);
    printf("  %s <n> naive            naive only\n", prog);
    printf("  %s <n> blocked [bs]     blocked only, optional block size\n", prog);
    printf("  %s <n> blocksweep       blocked, bs = 8..256\n", prog);
    printf("  %s scaling              n = 128..1024\n", prog);
}

int main(int argc, char **argv)
{
    int n  = DEFAULT_N;
    int bs = DEFAULT_BLOCK;

    if (argc >= 2 && strcmp(argv[1], "scaling") == 0) {
        mode_scaling(bs);
        return 0;
    }

    if (argc >= 2 && (strcmp(argv[1], "-h") == 0 ||
                      strcmp(argv[1], "--help") == 0)) {
        usage(argv[0]);
        return 0;
    }

    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "ERROR: n must be a positive integer\n");
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 4) {
        bs = atoi(argv[3]);
        if (bs <= 0) bs = DEFAULT_BLOCK;
    }

    if (argc >= 3) {
        if (strcmp(argv[2], "blocksweep") == 0) {
            mode_blocksweep(n);
        } else if (strcmp(argv[2], "naive") == 0 ||
                   strcmp(argv[2], "blocked") == 0) {
            mode_single(n, bs, argv[2]);
        } else {
            fprintf(stderr, "ERROR: unknown mode '%s'\n", argv[2]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    } else {
        mode_both(n, bs);
    }

    printf("\nresults appended to results.csv\n\n");
    return 0;
}
