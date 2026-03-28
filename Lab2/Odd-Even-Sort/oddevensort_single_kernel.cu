#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void print_sort_status(const std::vector<int>& numbers) {
    // std::is_sorted returns true (non-zero) if sorted.
    std::cout << "The output is sorted?: "
              << (std::is_sorted(numbers.begin(), numbers.end()) ? "True" : "False") << std::endl;
}

// CUDA Kernel: Odd-even Sort (Single Block Only)
__global__ void oddeven_sort_singlekernel(int*d_numbers, int s){
    int tid = threadIdx.x;
    // Outer loop for all phases (i=1 to s)
    for (int i = 1; i <= s; i++) {
    // --- Odd Phase ---
        if (i % 2 == 1) { 
            // Indices being compared: (1, 2), (3, 4), ...
            for (int j = 1 + 2 * tid; j < s - 1; j += 2 * blockDim.x) {
                // Ensure index j is within the array bounds and is odd
                if (j < s && j > 0) {
                    if (d_numbers[j] > d_numbers[j + 1]) {
                        int temp = d_numbers[j];
                        d_numbers[j] = d_numbers[j + 1];
                        d_numbers[j + 1] = temp;
                    }
                }
          }
    }
    __syncthreads();
    
    // --- Even Phase ---
        if (i % 2 == 0) { 
            // Indices being compared: (0, 1), (2, 3), ...
            for (int j = 2 * tid; j < s - 1; j += 2 * blockDim.x) {
                // Ensure index j is within the array bounds and is even
                if (j < s) {
                    if (d_numbers[j] > d_numbers[j + 1]) {
                        int temp = d_numbers[j];
                        d_numbers[j] = d_numbers[j + 1];
                        d_numbers[j + 1] = temp;
                    }
                }
            }
        }
        
        __syncthreads();
  }
}

void oddeven_sort_run(std::vector<int>& numbers, int threads_per_block){
    int s = numbers.size();
    int* d_numbers;
    
    auto start = std::chrono::steady_clock::now(); 

    cudaMalloc((void**)&d_numbers, s * sizeof(int));
    
    cudaMemcpy(d_numbers, numbers.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the single-block kernel 
    oddeven_sort_singlekernel<<<1, threads_per_block>>>(d_numbers, s);
    
    cudaDeviceSynchronize(); 

    cudaMemcpy(numbers.data(), d_numbers, s * sizeof(int), cudaMemcpyDeviceToHost); 

    // --- 2. End Timer ---
    auto end = std::chrono::steady_clock::now();

    cudaFree(d_numbers);

    std::cout << "Elapsed time = " << std::chrono::duration<double>(end - start).count() << " sec\n";
}

int main(){
    constexpr unsigned int SIZE = 524288; // 2^19
    const int THREADS_PER_BLOCK = 1024; // Example block size (adjust based on hardware)

    // Initialize vector
    std::vector<int> numbers(SIZE);
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    std::cout << "--- Task 1: Single Kernel Launch ---" << std::endl;
    std::cout << "Input Size: " << SIZE << std::endl;

    print_sort_status(numbers);

    // Perform parallel sort and measure time
    oddeven_sort_run(numbers, THREADS_PER_BLOCK);

    print_sort_status(numbers);

    return 0;
}
