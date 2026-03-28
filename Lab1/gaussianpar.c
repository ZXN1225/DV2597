/***************************************************************************
 *
 * Parallel version of Gaussian elimination using pthreads
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_SIZE    4096
#define MAX_THREADS 64   /* upper limit, just to be safe */

typedef double matrix[MAX_SIZE][MAX_SIZE];

int     N;              /* matrix size        */
int     maxnum;         /* max number of element*/
char    *Init;          /* matrix init type   */
int     PRINT;          /* print switch       */
matrix  A;              /* matrix A           */
double  b[MAX_SIZE];    /* vector b           */
double  y[MAX_SIZE];    /* vector y           */

/* pthread-related globals */
static int num_threads = 16;      /* default, can be changed by -t */
static pthread_barrier_t barrier;

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int  Read_Options(int, char **);

/* ---------- Thread worker ---------- */

static void
gaussian_worker(int tid)
{
    int k, i, j;

    for (k = 0; k < N; k++) {

        /* --- Division step: done by thread 0 only --- */
        if (tid == 0) {
            for (j = k+1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            y[k] = b[k] / A[k][k];
            A[k][k] = 1.0;
        }

        /* wait until division step for this k is done */
        pthread_barrier_wait(&barrier);

        /* --- Elimination step: parallel over rows i > k --- */

        int rows = N - (k + 1);  /* total number of rows to update */

        if (rows > 0) {
            /* block distribution of rows among threads */
            int base = rows / num_threads;
            int rem  = rows % num_threads;

            int my_rows, start_i, end_i;

            if (tid < rem) {
                my_rows = base + 1;
                start_i = (k + 1) + tid * my_rows;
            } else {
                my_rows = base;
                start_i = (k + 1) + rem * (base + 1) + (tid - rem) * base;
            }
            end_i = start_i + my_rows;

            /* If there are more threads than rows, some will have my_rows=0 */
            if (my_rows > 0) {
                for (i = start_i; i < end_i; i++) {
                    double factor = A[i][k];
                    if (factor != 0.0) {
                        for (j = k+1; j < N; j++)
                            A[i][j] = A[i][j] - factor * A[k][j];
                        b[i] = b[i] - factor * y[k];
                        A[i][k] = 0.0;
                    } else {
                        A[i][k] = 0.0;
                    }
                }
            }
        }

        /* wait until all threads have finished elimination for this k */
        pthread_barrier_wait(&barrier);
    }
}

static void *
thread_func(void *arg)
{
    int tid = *(int *)arg;
    gaussian_worker(tid);
    return NULL;
}

/* ---------- Parallel work() ---------- */

void
work(void)
{
    int t;

    /* clamp thread count to a safe range */
    if (num_threads < 1)
        num_threads = 1;
    if (num_threads > MAX_THREADS)
        num_threads = MAX_THREADS;

    /* Initialize barrier for num_threads participants (including main) */
    pthread_barrier_init(&barrier, NULL, num_threads);

    pthread_t *threads   = NULL;
    int       *thread_id = NULL;

    if (num_threads > 1) {
        threads = (pthread_t *)malloc((num_threads-1) * sizeof(pthread_t));
    }
    thread_id = (int *)malloc(num_threads * sizeof(int));

    /* Create worker threads with ids 1..num_threads-1 */
    for (t = 1; t < num_threads; t++) {
        thread_id[t] = t;
        pthread_create(&threads[t-1], NULL, thread_func, (void *)&thread_id[t]);
    }

    /* Main thread is worker 0 */
    thread_id[0] = 0;
    gaussian_worker(0);

    /* Wait for all other threads to finish */
    for (t = 1; t < num_threads; t++) {
        pthread_join(threads[t-1], NULL);
    }

    pthread_barrier_destroy(&barrier);
    free(thread_id);
    if (threads)
        free(threads);
}

/* ---------- Initialization & I/O (unchanged) ---------- */

void
Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init      = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init,"rand") == 0) {
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init,"fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    printf("Vector b:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", b[j]);
    printf("]\n");
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = (char *)"rand";
    maxnum = 15.0;
    PRINT = 0;
    num_threads = 16;    /* default: 16 threads for the lab */
}

int
Read_Options(int argc, char **argv)
{
    char *prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++*argv) {
                case 'n':
                    --argc;
                    N = atoi(*++argv);
                    break;
                case 'h':
                    printf("\nHELP: try gaussian -u \n\n");
                    exit(0);
                    break;
                case 'u':
                    printf("\nUsage: gaussian [-n problemsize]\n");
                    printf("           [-D] show default values \n");
                    printf("           [-h] help \n");
                    printf("           [-I init_type] fast/rand \n");
                    printf("           [-m maxnum] max random no \n");
                    printf("           [-P print_switch] 0/1 \n");
                    printf("           [-t num_threads] \n");
                    exit(0);
                    break;
                case 'D':
                    printf("\nDefault:  n         = %d ", N);
                    printf("\n          Init      = rand" );
                    printf("\n          maxnum    = 15 ");
                    printf("\n          P         = 0 ");
                    printf("\n          t         = 16 \n\n");
                    exit(0);
                    break;
                case 'I':
                    --argc;
                    Init = *++argv;
                    break;
                case 'm':
                    --argc;
                    maxnum = atoi(*++argv);
                    break;
                case 'P':
                    --argc;
                    PRINT = atoi(*++argv);
                    break;
                case 't':
                    --argc;
                    num_threads = atoi(*++argv);
                    if (num_threads < 1)
                        num_threads = 1;
                    if (num_threads > MAX_THREADS)
                        num_threads = MAX_THREADS;
                    break;
                default:
                    printf("%s: ignored option: -%s\n", prog, *argv);
                    printf("HELP: try %s -u \n\n", prog);
                    break;
            }
    return 0;
}

/* ---------- main ---------- */

int
main(int argc, char **argv)
{
    Init_Default();          /* Init default values   */
    Read_Options(argc,argv); /* Read arguments        */
    Init_Matrix();           /* Init the matrix       */
    work();                  /* Parallel Gaussian elimination */
    if (PRINT == 1)
       Print_Matrix();
    return 0;
}
