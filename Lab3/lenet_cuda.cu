/*
 * CUDA Implementation of LeNet-5 Forward Propagation
 * 
 * This file implements GPU-accelerated forward propagation for the LeNet-5
 * neural network. The backpropagation is performed on CPU.
 * 
 * Parallelization Strategy:
 * - Each image in the batch is processed in parallel (blockIdx.x)
 * - Each output channel is processed in parallel (blockIdx.y)
 * - Each output pixel is processed by a thread (threadIdx.x, threadIdx.y)
 */

#include "lenet.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// GPU Kernels
// ============================================================================

// ReLU activation function for GPU
__device__ double gpu_relu(double x) { 
    return x > 0 ? x : 0; 
}

/*
 * Convolution Forward Kernel
 * 
 * Grid: (batch_size, out_channels)
 * Block: (out_side, out_side)
 * 
 * Each thread computes one output pixel for one image and one output channel.
 */
__global__ void conv_forward_batch_kernel(
    const double* input, 
    double* output, 
    const double* weight, 
    const double* bias,
    int batch_size, 
    int in_ch, 
    int out_ch, 
    int in_side, 
    int out_side) 
{
    int img_idx = blockIdx.x;   // Which image in batch
    int oc = blockIdx.y;        // Which output channel
    int r = threadIdx.y;        // Output row
    int c = threadIdx.x;        // Output column
    
    if (img_idx < batch_size && oc < out_ch && r < out_side && c < out_side) {
        double sum = 0.0;
        const double* img_in = input + img_idx * (in_ch * in_side * in_side);
        
        // Convolve over all input channels and 5x5 kernel
        for (int ic = 0; ic < in_ch; ic++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    sum += img_in[ic * in_side * in_side + (r + i) * in_side + (c + j)] *
                           weight[(ic * out_ch + oc) * 25 + i * 5 + j];
                }
            }
        }
        
        // Apply bias and ReLU activation
        output[img_idx * (out_ch * out_side * out_side) + oc * out_side * out_side + r * out_side + c] = 
            gpu_relu(sum + bias[oc]);
    }
}

/*
 * Max Pooling Forward Kernel (2x2 pooling)
 * 
 * Grid: (batch_size, channels)
 * Block: (out_side, out_side)
 */
__global__ void maxpool_forward_batch_kernel(
    const double* input, 
    double* output, 
    int batch_size, 
    int ch, 
    int in_side, 
    int out_side) 
{
    int img_idx = blockIdx.x;   // Which image in batch
    int c = blockIdx.y;         // Which channel
    int r = threadIdx.y;        // Output row
    int col = threadIdx.x;      // Output column
    
    if (img_idx < batch_size && c < ch && r < out_side && col < out_side) {
        double max_val = -1e10;
        const double* img_in = input + img_idx * (ch * in_side * in_side);
        
        // Find max in 2x2 window
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double v = img_in[c * in_side * in_side + (r * 2 + i) * in_side + (col * 2 + j)];
                if (v > max_val) max_val = v;
            }
        }
        
        output[img_idx * (ch * out_side * out_side) + c * out_side * out_side + r * out_side + col] = max_val;
    }
}

/*
 * Fully Connected Forward Kernel
 * 
 * Grid: (batch_size)
 * Block: (out_len)
 */
__global__ void fc_forward_batch_kernel(
    const double* input, 
    double* output, 
    const double* weight, 
    const double* bias, 
    int batch_size, 
    int in_len, 
    int out_len) 
{
    int img_idx = blockIdx.x;   // Which image in batch
    int j = threadIdx.x;        // Which output neuron
    
    if (img_idx < batch_size && j < out_len) {
        double sum = 0.0;
        const double* img_in = input + img_idx * in_len;
        
        // Dot product
        for (int i = 0; i < in_len; i++) {
            sum += img_in[i] * weight[i * out_len + j];
        }
        
        output[img_idx * out_len + j] = gpu_relu(sum + bias[j]);
    }
}

// ============================================================================
// GPU Memory Management
// ============================================================================

// Device pointers for layer activations
static double *d_in, *d_l1, *d_l2, *d_l3, *d_l4, *d_l5, *d_out;

// Device pointers for weights and biases
static double *d_w01, *d_w23, *d_w45, *d_w56;
static double *d_b01, *d_b23, *d_b45, *d_b56;

// Track allocated batch size to avoid reallocation
static int current_max_batch = 0;

/*
 * Copy weights from host LeNet5 structure to GPU
 */
void sync_weights_to_gpu(LeNet5 *h_lenet) {
    cudaMemcpy(d_w01, h_lenet->weight0_1, sizeof(h_lenet->weight0_1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w23, h_lenet->weight2_3, sizeof(h_lenet->weight2_3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w45, h_lenet->weight4_5, sizeof(h_lenet->weight4_5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w56, h_lenet->weight5_6, sizeof(h_lenet->weight5_6), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b01, h_lenet->bias0_1, sizeof(h_lenet->bias0_1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b23, h_lenet->bias2_3, sizeof(h_lenet->bias2_3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b45, h_lenet->bias4_5, sizeof(h_lenet->bias4_5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b56, h_lenet->bias5_6, sizeof(h_lenet->bias5_6), cudaMemcpyHostToDevice);
}

/*
 * Allocate GPU memory for batch processing
 */
void allocate_gpu_memory(int batch_size) {
    if (batch_size <= current_max_batch && current_max_batch > 0) return;
    
    // Free old allocations if expanding
    if (current_max_batch > 0) {
        cudaFree(d_in); cudaFree(d_l1); cudaFree(d_l2); cudaFree(d_l3);
        cudaFree(d_l4); cudaFree(d_l5); cudaFree(d_out);
    }
    
    // Allocate layer activations for entire batch
    // Layer sizes: input=32x32, l1=6x28x28, l2=6x14x14, l3=16x10x10, l4=16x5x5, l5=120, out=10
    cudaMalloc(&d_in,  batch_size * 1 * 32 * 32 * sizeof(double));
    cudaMalloc(&d_l1,  batch_size * 6 * 28 * 28 * sizeof(double));
    cudaMalloc(&d_l2,  batch_size * 6 * 14 * 14 * sizeof(double));
    cudaMalloc(&d_l3,  batch_size * 16 * 10 * 10 * sizeof(double));
    cudaMalloc(&d_l4,  batch_size * 16 * 5 * 5 * sizeof(double));
    cudaMalloc(&d_l5,  batch_size * 120 * sizeof(double));
    cudaMalloc(&d_out, batch_size * 10 * sizeof(double));
    
    // Allocate weights only once
    if (current_max_batch == 0) {
        cudaMalloc(&d_w01, sizeof(double) * 1 * 6 * 25);      // weight0_1
        cudaMalloc(&d_w23, sizeof(double) * 6 * 16 * 25);     // weight2_3
        cudaMalloc(&d_w45, sizeof(double) * 16 * 120 * 25);   // weight4_5
        cudaMalloc(&d_w56, sizeof(double) * 120 * 10);        // weight5_6
        cudaMalloc(&d_b01, sizeof(double) * 6);
        cudaMalloc(&d_b23, sizeof(double) * 16);
        cudaMalloc(&d_b45, sizeof(double) * 120);
        cudaMalloc(&d_b56, sizeof(double) * 10);
    }
    
    current_max_batch = batch_size;
}

// ============================================================================
// Preprocessing
// ============================================================================

/*
 * Preprocess input image: normalize and add padding
 * 
 * IMPORTANT: Padding pixels must be 0, not (0-mean)/std
 * This was the bug in the original implementation.
 */
void gpu_preprocess(image input, double* output_ptr) {
    double mean = 0, std = 0;
    const int sz = 28 * 28;
    
    // Calculate mean and std
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            mean += input[i][j];
            std += (double)input[i][j] * input[i][j];
        }
    }
    mean /= sz;
    std = sqrt(std / sz - mean * mean);
    
    // Initialize entire 32x32 to 0 (padding)
    for (int i = 0; i < 32 * 32; i++) {
        output_ptr[i] = 0.0;
    }
    
    // Fill in the 28x28 image data with PADDING=2 offset
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            output_ptr[(i + PADDING) * 32 + (j + PADDING)] = ((double)input[i][j] - mean) / std;
        }
    }
}

// ============================================================================
// Main CUDA Functions (called from main.c)
// ============================================================================

/*
 * Train a batch of images using GPU for forward propagation
 * and CPU for backpropagation
 */
extern "C" void TrainBatch_CUDA(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize) {
    allocate_gpu_memory(batchSize);
    sync_weights_to_gpu(lenet);

    // Preprocess all images on CPU and copy to GPU
    double *h_batch_in = (double*)malloc(batchSize * 1024 * sizeof(double));
    for (int i = 0; i < batchSize; i++) {
        gpu_preprocess(inputs[i], h_batch_in + i * 1024);
    }
    cudaMemcpy(d_in, h_batch_in, batchSize * 1024 * sizeof(double), cudaMemcpyHostToDevice);

    // Forward propagation on GPU for entire batch
    // Layer 1: Conv 32x32 -> 28x28, 6 output channels
    conv_forward_batch_kernel<<<dim3(batchSize, 6), dim3(28, 28)>>>(
        d_in, d_l1, d_w01, d_b01, batchSize, 1, 6, 32, 28);
    
    // Layer 2: MaxPool 28x28 -> 14x14
    maxpool_forward_batch_kernel<<<dim3(batchSize, 6), dim3(14, 14)>>>(
        d_l1, d_l2, batchSize, 6, 28, 14);
    
    // Layer 3: Conv 14x14 -> 10x10, 16 output channels
    conv_forward_batch_kernel<<<dim3(batchSize, 16), dim3(10, 10)>>>(
        d_l2, d_l3, d_w23, d_b23, batchSize, 6, 16, 14, 10);
    
    // Layer 4: MaxPool 10x10 -> 5x5
    maxpool_forward_batch_kernel<<<dim3(batchSize, 16), dim3(5, 5)>>>(
        d_l3, d_l4, batchSize, 16, 10, 5);
    
    // Layer 5: Conv 5x5 -> 1x1, 120 output channels
    conv_forward_batch_kernel<<<dim3(batchSize, 120), dim3(1, 1)>>>(
        d_l4, d_l5, d_w45, d_b45, batchSize, 16, 120, 5, 1);
    
    // Output: Fully connected 120 -> 10
    fc_forward_batch_kernel<<<batchSize, 10>>>(
        d_l5, d_out, d_w56, d_b56, batchSize, 120, 10);

    // Single sync point after all kernels complete
    cudaDeviceSynchronize();
    
    // Check for any kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Copy all layer outputs back to CPU for backpropagation
    double *h_all_l1 = (double*)malloc(batchSize * 6 * 28 * 28 * sizeof(double));
    double *h_all_l2 = (double*)malloc(batchSize * 6 * 14 * 14 * sizeof(double));
    double *h_all_l3 = (double*)malloc(batchSize * 16 * 10 * 10 * sizeof(double));
    double *h_all_l4 = (double*)malloc(batchSize * 16 * 5 * 5 * sizeof(double));
    double *h_all_l5 = (double*)malloc(batchSize * 120 * sizeof(double));
    double *h_all_out = (double*)malloc(batchSize * 10 * sizeof(double));

    cudaMemcpy(h_all_l1, d_l1, batchSize * 6 * 28 * 28 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_l2, d_l2, batchSize * 6 * 14 * 14 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_l3, d_l3, batchSize * 16 * 10 * 10 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_l4, d_l4, batchSize * 16 * 5 * 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_l5, d_l5, batchSize * 120 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_out, d_out, batchSize * 10 * sizeof(double), cudaMemcpyDeviceToHost);

    // Backpropagation on CPU (accumulate gradients)
    double buffer[sizeof(LeNet5)/sizeof(double)] = {0};
    
    for (int i = 0; i < batchSize; i++) {
        Feature f = {0}; 
        Feature err = {0}; 
        LeNet5 d = {0};
        
        // Reconstruct feature structure from GPU outputs
        memcpy(f.input, h_batch_in + i * 1024, 1024 * sizeof(double));
        memcpy(f.layer1, h_all_l1 + i * 6 * 28 * 28, 6 * 28 * 28 * sizeof(double));
        memcpy(f.layer2, h_all_l2 + i * 6 * 14 * 14, 6 * 14 * 14 * sizeof(double));
        memcpy(f.layer3, h_all_l3 + i * 16 * 10 * 10, 16 * 10 * 10 * sizeof(double));
        memcpy(f.layer4, h_all_l4 + i * 16 * 5 * 5, 16 * 5 * 5 * sizeof(double));
        memcpy(f.layer5, h_all_l5 + i * 120, 120 * sizeof(double));
        memcpy(f.output, h_all_out + i * 10, 10 * sizeof(double));
        
        // Compute loss and backpropagate
        load_target(&f, &err, labels[i]);
        backward(lenet, &d, &err, &f, relugrad);
        
        // Accumulate gradients
        double* d_ptr = (double*)&d;
        for (int j = 0; j < sizeof(LeNet5)/sizeof(double); j++) {
            buffer[j] += d_ptr[j];
        }
    }
    
    // Update weights
    double k = ALPHA / batchSize;
    double* l_ptr = (double*)lenet;
    for (int i = 0; i < sizeof(LeNet5)/sizeof(double); i++) {
        l_ptr[i] += k * buffer[i];
    }

    // Cleanup
    free(h_batch_in); 
    free(h_all_l1); free(h_all_l2); free(h_all_l3);
    free(h_all_l4); free(h_all_l5); free(h_all_out);
}

/*
 * Predict labels for a batch of images using GPU
 */
extern "C" void Predict_Batch_CUDA(LeNet5 *lenet, image *inputs, uint8 *results, int batchSize) {
    allocate_gpu_memory(batchSize);
    sync_weights_to_gpu(lenet);

    // Preprocess and copy to GPU
    double *h_batch_in = (double*)malloc(batchSize * 1024 * sizeof(double));
    for (int i = 0; i < batchSize; i++) {
        gpu_preprocess(inputs[i], h_batch_in + i * 1024);
    }
    cudaMemcpy(d_in, h_batch_in, batchSize * 1024 * sizeof(double), cudaMemcpyHostToDevice);

    // Forward propagation on GPU
    conv_forward_batch_kernel<<<dim3(batchSize, 6), dim3(28, 28)>>>(
        d_in, d_l1, d_w01, d_b01, batchSize, 1, 6, 32, 28);
    maxpool_forward_batch_kernel<<<dim3(batchSize, 6), dim3(14, 14)>>>(
        d_l1, d_l2, batchSize, 6, 28, 14);
    conv_forward_batch_kernel<<<dim3(batchSize, 16), dim3(10, 10)>>>(
        d_l2, d_l3, d_w23, d_b23, batchSize, 6, 16, 14, 10);
    maxpool_forward_batch_kernel<<<dim3(batchSize, 16), dim3(5, 5)>>>(
        d_l3, d_l4, batchSize, 16, 10, 5);
    conv_forward_batch_kernel<<<dim3(batchSize, 120), dim3(1, 1)>>>(
        d_l4, d_l5, d_w45, d_b45, batchSize, 16, 120, 5, 1);
    fc_forward_batch_kernel<<<batchSize, 10>>>(
        d_l5, d_out, d_w56, d_b56, batchSize, 120, 10);

    cudaDeviceSynchronize();
    
    // Copy output and find predictions
    double *h_all_out = (double*)malloc(batchSize * 10 * sizeof(double));
    cudaMemcpy(h_all_out, d_out, batchSize * 10 * sizeof(double), cudaMemcpyDeviceToHost);

    // Find argmax for each image
    for (int i = 0; i < batchSize; i++) {
        double *prob = h_all_out + i * 10;
        uint8 res = 0; 
        double maxv = prob[0];
        for (uint8 j = 1; j < 10; j++) { 
            if (prob[j] > maxv) { 
                maxv = prob[j]; 
                res = j; 
            } 
        }
        results[i] = res;
    }
    
    free(h_batch_in); 
    free(h_all_out);
}
