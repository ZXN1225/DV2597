#!/bin/bash

# Configuration
ARRAY_SIZE=524288 
# Threads per Block 
THREADS_PER_BLOCK=1024 

# Executables names
SEQ_EXEC="./oddevensort_seq"
TASK1_EXEC="./oddevensort_task1"
TASK2_EXEC="./oddevensort_task2"

# --- Start of output redirection block ---
{

echo "--- Odd-Even Sort Performance Benchmark ---"
echo "Array Size: ${ARRAY_SIZE} elements"
echo "Threads per Block (Task 1 & 2): ${THREADS_PER_BLOCK}"
echo ""

# Function to run an executable and extract the time
# IMPORTANT: This function redirects progress/diagnostics to stderr (>&2) and ONLY prints the final time value to stdout.
run_and_measure() {
    local exec_path=$1
    echo "Running ${exec_path}..." >&2 
    
    # Run the program, capture stdout, and extract the time value
    TIME=$(${exec_path} 2>&1 | grep "Elapsed time =" | awk '{print $4}')
    
    if [ -z "${TIME}" ]; then
        echo "Error: Could not extract time from ${exec_path} output." >&2
        echo "Please ensure the executable is compiled and runs correctly." >&2
        TIME=0
    else
        echo "Raw Time: ${TIME} sec" >&2 
    fi
    
    # ONLY output the time value to stdout for the calling variable capture
    echo "${TIME}"
}

# --- Start Benchmark ---

## 1. Sequential Version (Reference Time)
echo "## 1. Sequential Version"
SEQ_TIME=$(run_and_measure "${SEQ_EXEC}") 
echo "Sequential Time (T_seq): ${SEQ_TIME} sec"
echo "---"

## 2. Task 1 (Single Kernel, Single Block)
echo "## 2. Task 1 (Single Kernel, Single Block)"
TASK1_TIME=$(run_and_measure "${TASK1_EXEC}")
echo "Task 1 Time (T_task1): ${TASK1_TIME} sec"

# Calculate Speedup (Task 1 vs Sequential)
if (( $(echo "${SEQ_TIME} > 0 && ${TASK1_TIME} > 0" | bc -l) )); then
    SPEEDUP_TASK1=$(echo "${SEQ_TIME} / ${TASK1_TIME}" | bc -l)
    printf "Speedup (Task 1 vs Seq): %.4fx\n" "${SPEEDUP_TASK1}"
else
    echo "Cannot calculate Speedup: One of the times is zero."
fi
echo "---"

## 3. Task 2 (Multiple Kernels, Multi Blocks)
echo "## 3. Task 2 (Multiple Kernels, Multi Blocks)"
TASK2_TIME=$(run_and_measure "${TASK2_EXEC}")
echo "Task 2 Time (T_task2): ${TASK2_TIME} sec"

# Calculate Speedup (Task 2 vs Sequential)
if (( $(echo "${SEQ_TIME} > 0 && ${TASK2_TIME} > 0" | bc -l) )); then
    SPEEDUP_TASK2=$(echo "${SEQ_TIME} / ${TASK2_TIME}" | bc -l)
    printf "Speedup (Task 2 vs Seq): %.4fx\n" "${SPEEDUP_TASK2}"
else
    echo "Cannot calculate Speedup: One of the times is zero."
fi
echo "---"

## 4. Task 2 vs Task 1 Speedup
# Calculate Speedup (Task 2 vs Task 1)
if (( $(echo "${TASK1_TIME} > 0 && ${TASK2_TIME} > 0" | bc -l) )); then
    SPEEDUP_T2_T1=$(echo "${TASK1_TIME} / ${TASK2_TIME}" | bc -l)
    printf "Speedup (Task 2 vs Task 1): %.4fx\n" "${SPEEDUP_T2_T1}"
else
    echo "Cannot calculate Speedup: One of the times is zero."
fi

echo "--- Benchmark Complete ---"

# --- End of output redirection block ---
} | tee results.txt 
