/***************************************************************************
 *
 * CUDA version of Gauss-Jordan row reduction
 * Task 3 - Parallel Implementation
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_SIZE 4096
typedef double matrix[MAX_SIZE][MAX_SIZE];

/* Global variables - same as sequential */
int   N;
int   maxnum;
char* Init;
int   PRINT;

matrix A;
double b[MAX_SIZE];
double y[MAX_SIZE];

/* Forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int  Read_Options(int, char**);

/*
 * Macro to check CUDA errors
 * If any CUDA function fails, print error message and exit
 */
static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* ==================== CUDA KERNELS ==================== */

/*
 * Kernel 1 - Normalize pivot row
 * Each thread divides one element A[k][j] by the pivot A[k][k]
 * This is the division step from the sequential algorithm
 */
__global__ void normalize_row(double* A_d, int N, int k)
{
    double pivot = A_d[k * N + k];                    // Get pivot value at diagonal A[k][k]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
    int j = k + 1 + tid;                              // Column index to the right of pivot (j > k)

    if (j < N) {
        A_d[k * N + j] /= pivot;                      // Divide A[k][j] element by pivot value
    }
}

/*
 * Kernel 2 - Set pivot to 1.0 and compute y[k]
 * Only one thread does this work because we update only two values
 * We launch with <<<1, 1>>> which means 1 block with 1 thread
 */
__global__ void set_pivot_y(double* A_d, const double* b_d, double* y_d, int N, int k)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {        // Only first thread works
        double pivot = A_d[k * N + k];                // Get pivot value at diagonal A[k][k]
        y_d[k] = b_d[k] / pivot;                      // Compute partial solution y[k]
        A_d[k * N + k] = 1.0;                         // Set pivot to 1 (identity matrix needs 1s on diagonal)
    }
}

/*
 * Kernel 3 - Eliminate elements below pivot (matrix update)
 * Uses 2D thread blocks: each thread updates one element A[i][j]
 * where i > k (rows below pivot)
 */
__global__ void eliminate_below_matrix(double* A_d, int N, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID for column
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Global thread ID for row

    int j = k + 1 + col;                        // Column index to the right of pivot
    int i = k + 1 + row;                        // Row index below pivot

    if (i < N && j < N) {
        double aik = A_d[i * N + k];            // Get element at column k (the factor for elimination)
        double akj = A_d[k * N + j];            /// Get pivot row element at column j
        A_d[i * N + j] -= aik * akj;            // Update A[i][j]: eliminate below pivot
    }
}

/*
 * Kernel 4 - Update vector b and zero out column below pivot
 * Each thread processes one row below the pivot
 */
__global__ void eliminate_below_vector(double* A_d, double* b_d, const double* y_d, int N, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
    int i = k + 1 + tid;                              // Row index below pivot

    if (i < N) {
        double aik = A_d[i * N + k];            // Get element at column k
        b_d[i] -= aik * y_d[k];                 // Update b vector for rows below pivot
        A_d[i * N + k] = 0.0;                   // Set A[i][k] to zero
    }
}


/*
 * Kernel 5 - Eliminate elements above pivot (Gauss-Jordan extension)
 * Uses 2D thread blocks: each thread updates one element A[i][j]
 * where i < k (rows above pivot)
 */
__global__ void eliminate_above_matrix(double* A_d, int N, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID for column
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Global thread ID for row

    int j = k + 1 + col;                              // Column index to the right of pivot (j > k)
    int i = row;                                      // Row index above pivot (i < k)

    if (i < k && j < N) {
        double aik = A_d[i * N + k];                  // Get element at column k (the factor for elimination)
        double akj = A_d[k * N + j];                  // Get pivot row element at column j
        A_d[i * N + j] -= aik * akj;                  // Update A[i][j]: eliminate above pivot
    }
}

/*
 * Kernel 6 - Update vector y and set column above pivot to zero
 * Gauss-Jordan extension: updates y instead of b for rows above pivot
 */
__global__ void eliminate_above_vector(double* A_d, double* y_d, int N, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
    int i = tid;                                      // Row index above pivot (i < k)

    if (i < k) {
        double aik = A_d[i * N + k];                  // Get element at column k (the factor for elimination)
        y_d[i] -= aik * y_d[k];                       // Update y[i] for rows above pivot (y[i] already computed)
        A_d[i * N + k] = 0.0;                         // Set A[i][k] to zero (elimination complete)
    }
}

/* ==================== HELPER FUNCTIONS ==================== */

/*
 * Convert 2D matrix to 1D array for GPU
 * GPU works better with contiguous memory (1D array)
 * A[i][j] becomes A_h[i * N + j] (row major order)
 */
static double* pack_A(int N) {
    size_t bytesA = (size_t)N * (size_t)N * sizeof(double);
    double* A_h = (double*)malloc(bytesA);
    if (!A_h) {
        perror("malloc A_h");
        exit(1);
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A_h[i * N + j] = A[i][j];

    return A_h;
}

/*
 * Convert 1D array back to 2D matrix
 * After GPU computation, copy results back to original 2D format
 */
static void unpack_A(const double* A_h, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = A_h[i * N + j];
}

/* ==================== MAIN WORK FUNCTION ==================== */

/*
 * Parallel version of work() function
 *
 * The outer loop (k = 0 to N-1) must run sequentially because each
 * iteration depends on the previous one.
 *
 * For a 2048x2048 matrix, we launch 6 kernels per iteration,
 * which means 6 * 2048 = 12,288 kernel launches total.
 */
void work(void)
{
    double *A_d = NULL, *b_d = NULL, *y_d = NULL;  // Device pointers

    size_t bytesA = (size_t)N * (size_t)N * sizeof(double);  // Matrix size in bytes
    size_t bytesV = (size_t)N * sizeof(double);              // Vector size in bytes

    /* Pack 2D matrix into 1D array for GPU */
    double* A_h = pack_A(N);


    /* Allocate memory on GPU using cudaMalloc. */
    gpuErrchk(cudaMalloc((void**)&A_d, bytesA));
    gpuErrchk(cudaMalloc((void**)&b_d, bytesV));
    gpuErrchk(cudaMalloc((void**)&y_d, bytesV));

    /* Copy data from CPU to GPU */
    gpuErrchk(cudaMemcpy(A_d, A_h, bytesA, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_d, b, bytesV, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(y_d, y, bytesV, cudaMemcpyHostToDevice));


    /*
     * Set block sizes for kernel launches.
     * 256 is a multiple of 32 (warp size), which gives better performance.
     */
    const int VEC_BLOCK = 256;      // 256 threads for 1D kernels (vector operations).
    dim3 MAT_BLOCK(16, 16);         // 16x16 = 256 threads for 2D kernels (matrix operations).

    /* Main loop runs on CPU same as sequential, but inner operations run on GPU */
    for (int k = 0; k < N; k++) {

        /* Step 1: Normalize pivot row - parallel division */
        int cols = N - (k + 1);                         // Number of columns to the right of pivot
        if (cols > 0) {
            /* Kernel 1: Divide pivot row elements by pivot value - uses 1D thread layout */
            int normGrid = (cols + VEC_BLOCK - 1) / VEC_BLOCK;
            normalize_row<<<normGrid, VEC_BLOCK>>>(A_d, N, k);

            /*
             * cudaGetLastError() checks if the kernel launch failed.
             * Kernel launches are asynchronous, so errors don't appear immediately.
             * This helps us find bugs quickly during development.
             */
            gpuErrchk(cudaGetLastError());

            /*
             * cudaDeviceSynchronize() waits for the kernel to finish.
             * Each kernel depends on results of the previous one.
             */
            gpuErrchk(cudaDeviceSynchronize());
        }

        /* Step 2: Set pivot to 1.0 and compute y[k] */
        /* Kernel 2: Only one thread needed - launch with <<<1, 1>>> */
        set_pivot_y<<<1, 1>>>(A_d, b_d, y_d, N, k);
        gpuErrchk(cudaGetLastError());                  // Check if kernel launch failed
        gpuErrchk(cudaDeviceSynchronize());             // Wait for kernel to finish

        /* Step 3: Eliminate below pivot - parallel elimination */
        int belowRows = N - (k + 1);                    // Number of rows below pivot (k+1 to N-1)
        int belowCols = N - (k + 1);                    // Number of columns to the right of pivot

        if (belowRows > 0 && belowCols > 0) {
            /* Kernel 3: Update matrix A below pivot - uses 2D thread layout */
            dim3 grid((belowCols + MAT_BLOCK.x - 1) / MAT_BLOCK.x,
                      (belowRows + MAT_BLOCK.y - 1) / MAT_BLOCK.y);
            eliminate_below_matrix<<<grid, MAT_BLOCK>>>(A_d, N, k);
            gpuErrchk(cudaGetLastError());              // Check if kernel launch failed
            gpuErrchk(cudaDeviceSynchronize());         // Wait for kernel to finish

            /* Kernel 4: Update vector b below pivot - uses 1D thread layout */
            int vecGrid = (belowRows + VEC_BLOCK - 1) / VEC_BLOCK;
            eliminate_below_vector<<<vecGrid, VEC_BLOCK>>>(A_d, b_d, y_d, N, k);
            gpuErrchk(cudaGetLastError());              // Check if kernel launch failed
            gpuErrchk(cudaDeviceSynchronize());         // Wait for kernel to finish
        }

        /* Step 4: Eliminate above pivot - Gauss-Jordan extension */
        int aboveRows = k;                              // Number of rows above pivot (0 to k-1)
        int aboveCols = N - (k + 1);                    // Number of columns to the right of pivot

        if (aboveRows > 0) {
            /* Kernel 5: Update matrix A above pivot - uses 2D thread layout */
            if (aboveCols > 0) {
                dim3 grid2((aboveCols + MAT_BLOCK.x - 1) / MAT_BLOCK.x,
                           (aboveRows + MAT_BLOCK.y - 1) / MAT_BLOCK.y);
                eliminate_above_matrix<<<grid2, MAT_BLOCK>>>(A_d, N, k);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }

            /* Kernel 6: Update vector y above pivot - uses 1D thread layout */
            int vecGrid2 = (aboveRows + VEC_BLOCK - 1) / VEC_BLOCK;
            eliminate_above_vector<<<vecGrid2, VEC_BLOCK>>>(A_d, y_d, N, k);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }

    /* Copy results from GPU (device) back to CPU (host) */
    gpuErrchk(cudaMemcpy(A_h, A_d, bytesA, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(b, b_d, bytesV, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(y, y_d, bytesV, cudaMemcpyDeviceToHost));

    /* Unpack 1D array back to 2D matrix */
    unpack_A(A_h, N);

    /* Free all allocated memory */
    free(A_h);
    cudaFree(A_d);
    cudaFree(b_d);
    cudaFree(y_d);
}

/* ==================== MAIN ==================== */
/* Same as sequential, just different print message */

int main(int argc, char** argv)
{
    printf("Gauss Jordan (CUDA Parallel)\n");

    Init_Default();
    Read_Options(argc, argv);
    Init_Matrix();

    work();

    if (PRINT == 1)
        Print_Matrix();

    return 0;
}

/* ==================== INITIALIZATION & I/O ==================== */

void Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init\t  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init, "rand") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j)
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }

    if (strcmp(Init, "fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j)
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }

    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");

    printf("\n\n");
}

void Init_Default()
{
    N = 2048;
    Init = (char*)"fast";
    maxnum = 15;
    PRINT = 0;
}

int Read_Options(int argc, char** argv)
{
    char* prog = *argv;

    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++ * argv) {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
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
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }

    return 0;
}
