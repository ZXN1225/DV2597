#!/bin/bash
# run_all.sh - Compile and run both CPU and GPU versions, then calculate speedup
#
# Usage:
#   Google Colab T4:  ./run_all.sh t4
#   Local Laptop:     ./run_all.sh local

echo "============================================================"
echo "LeNet-5 CUDA vs CPU Benchmark"
echo "============================================================"
echo ""

# Select Makefile based on argument
if [ "$1" == "t4" ]; then
    MAKEFILE="Makefile.t4"
    echo "Using: Makefile.t4 (Google Colab T4 GPU)"
elif [ "$1" == "local" ]; then
    MAKEFILE="Makefile.local"
    echo "Using: Makefile.local (Local Laptop GPU)"
else
    echo "Usage: ./run_all.sh [t4|local]"
    echo "  t4    - For Google Colab T4 GPU"
    echo "  local - For local laptop GPU"
    exit 1
fi

echo ""

# Clean and compile
echo "Compiling..."
make -f $MAKEFILE clean
make -f $MAKEFILE all

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo ""
echo "Compilation successful!"
echo ""

# Make executables runnable
chmod +x lenet_cpu lenet_gpu

# Run CPU version and capture output
echo "============================================================"
echo "Running CPU version..."
echo "============================================================"
CPU_OUTPUT=$(./lenet_cpu 2>&1)
echo "$CPU_OUTPUT"

# Extract CPU times using grep and awk
CPU_TRAIN_TIME=$(echo "$CPU_OUTPUT" | grep "CPU Training Time:" | tail -1 | awk '{print $4}')
CPU_TEST_TIME=$(echo "$CPU_OUTPUT" | grep "CPU Testing Time:" | awk '{print $4}')

echo ""
echo "============================================================"
echo "Running GPU version..."
echo "============================================================"
GPU_OUTPUT=$(./lenet_gpu 2>&1)
echo "$GPU_OUTPUT"

# Extract GPU times
GPU_TRAIN_TIME=$(echo "$GPU_OUTPUT" | grep "GPU Training Time:" | tail -1 | awk '{print $4}')
GPU_TEST_TIME=$(echo "$GPU_OUTPUT" | grep "GPU Testing Time:" | awk '{print $4}')

# Calculate speedups
echo ""
echo "============================================================"
echo "SPEEDUP ANALYSIS"
echo "============================================================"
echo ""
echo "CPU Training Time: $CPU_TRAIN_TIME seconds"
echo "GPU Training Time: $GPU_TRAIN_TIME seconds"
echo "CPU Testing Time:  $CPU_TEST_TIME seconds"
echo "GPU Testing Time:  $GPU_TEST_TIME seconds"
echo ""

# Use bc for floating point division
if command -v bc &> /dev/null; then
    TRAIN_SPEEDUP=$(echo "scale=2; $CPU_TRAIN_TIME / $GPU_TRAIN_TIME" | bc)
    TEST_SPEEDUP=$(echo "scale=2; $CPU_TEST_TIME / $GPU_TEST_TIME" | bc)
    echo "Training Speedup: ${TRAIN_SPEEDUP}x"
    echo "Testing Speedup:  ${TEST_SPEEDUP}x"
else
    # Fallback using awk if bc not available
    TRAIN_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $CPU_TRAIN_TIME / $GPU_TRAIN_TIME}")
    TEST_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $CPU_TEST_TIME / $GPU_TEST_TIME}")
    echo "Training Speedup: ${TRAIN_SPEEDUP}x"
    echo "Testing Speedup:  ${TEST_SPEEDUP}x"
fi

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "============================================================"
