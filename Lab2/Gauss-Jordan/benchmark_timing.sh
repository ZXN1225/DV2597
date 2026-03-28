#!/bin/bash

GREEN="\033[0;32m"
RED="\033[0;31m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
END="\033[0m"

echo -e "${BLUE}=========================================${END}"
echo -e "${BLUE}Task 3: Gauss-Jordan Performance Test${END}"
echo -e "${BLUE}=========================================${END}"
echo ""

# Ensure executables are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sequential_executable> <parallel_executable>"
    exit 1
fi

seq_executable=$1
par_executable=$2

# Test with 2048x2048 matrix as required by Task 3
n=2048

echo -e "${YELLOW}Testing with n=$n (2048x2048 matrix)${END}"
echo ""

# Run Sequential Version and measure time
echo -e "${BLUE}Running Sequential Version...${END}"
seq_start=$(date +%s.%N)
seq_result=$($seq_executable -n $n -P 0 -I fast)
seq_end=$(date +%s.%N)
seq_time=$(echo "$seq_end - $seq_start" | bc | sed 's/^\./0./')

echo "$seq_result"
echo -e "${GREEN}Sequential Execution Time: $seq_time seconds${END}"
echo ""

# Run Parallel Version and measure time
echo -e "${BLUE}Running Parallel GPU Version...${END}"
par_start=$(date +%s.%N)
par_result=$($par_executable -n $n -P 0 -I fast)
par_end=$(date +%s.%N)
par_time=$(echo "$par_end - $par_start" | bc | sed 's/^\./0./')

echo "$par_result"
echo -e "${GREEN}Parallel Execution Time: $par_time seconds${END}"
echo ""

# Calculate Speedup
speedup=$(echo "scale=2; $seq_time / $par_time" | bc)

echo -e "${BLUE}=========================================${END}"
echo -e "${YELLOW}RESULTS SUMMARY${END}"
echo -e "${BLUE}=========================================${END}"
echo -e "Sequential Time:  ${GREEN}$seq_time seconds${END}"
echo -e "Parallel Time:    ${GREEN}$par_time seconds${END}"
echo -e "Speedup:          ${GREEN}${speedup}x${END}"
echo -e "${BLUE}=========================================${END}"
echo ""

# Check if speedup is positive
if (( $(echo "$speedup > 1" | bc -l) )); then
    echo -e "${GREEN}✓ Task 3 Requirement MET: Positive speedup achieved!${END}"
else
    echo -e "${RED}✗ Warning: Speedup is less than 1x${END}"
fi