#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath> 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << err << "(" << cudaGetErrorString(err) << ") from " << func << std::endl;
        exit(1);
    }
}

 void print_sort_status(const std::vector<int>& numbers) {
     std::cerr << "The output is sorted?: "
               << (std::is_sorted(numbers.begin(), numbers.end()) ? "True" : "False") << std::endl;
 }

// CUDA Kernel: Odd-even Sort (multiple kernel)
__global__ void oddeven_sort_kernel(int* d_numbers, int s, int phase) {
    // phase == 0 for Odd phase, phase == 1 for Even phase
    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_index;
    if (phase == 0) { // Indices being compared: (1, 2), (3, 4), ...
        start_index = 1;
    } else { // Even Phase:  (0, 1), (2, 3), ...
        start_index = 0;
    }
    
    int j = start_index + 2 * global_id; 

    if (j < s - 1) { 
        if (d_numbers[j] > d_numbers[j + 1]) {
            // swap
            int temp = d_numbers[j];
            d_numbers[j] = d_numbers[j + 1];
            d_numbers[j + 1] = temp;
        }
    }
}

double oddeven_sort_parallel_run(std::vector<int>& numbers, int threads_per_block) {
    int s = numbers.size();
    int* d_numbers;
    
    // Each thread processes one comparison pair.。
    int num_comparisons = s / 2;
    int blocks_per_grid = (int)std::ceil((double)num_comparisons / threads_per_block);
    
    int total_phases = s; 

    // Use CUDA event to keep track of time
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_numbers, s * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_numbers, numbers.data(), s * sizeof(int), cudaMemcpyHostToDevice));

    // start timekeeping
    CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));

    // start the kernel
    for (int i = 0; i < total_phases; i++) {
        int phase = i % 2; // 0 for Odd, 1 for Even
        
        oddeven_sort_kernel<<<blocks_per_grid, threads_per_block>>>(d_numbers, s, phase);

        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

    // calculate time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    double elapsed_time = milliseconds / 1000.0; 

    CHECK_CUDA_ERROR(cudaMemcpy(numbers.data(), d_numbers, s * sizeof(int), cudaMemcpyDeviceToHost)); 

    CHECK_CUDA_ERROR(cudaFree(d_numbers));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));

    std::cout << "Elapsed time = " << elapsed_time << " sec" << std::endl;
    
    std::cerr << "Parallel Config: Blocks=" << blocks_per_grid << ", Threads/Block=" << threads_per_block << std::endl;

    return elapsed_time;
}


int main() {

    constexpr unsigned int SIZE = 524288; 
    const int THREADS_PER_BLOCK = 1024; 

    std::vector<int> numbers_par(SIZE);
    srand(time(0));
    std::generate(numbers_par.begin(), numbers_par.end(), rand); 

    std::cerr << "--- Task 2: Multiple Kernels, Multi Blocks ---" << std::endl;
    std::cerr << "Input Size: " << SIZE << std::endl;

    print_sort_status(numbers_par);

    oddeven_sort_parallel_run(numbers_par, THREADS_PER_BLOCK);

    print_sort_status(numbers_par);

    return 0;
}
