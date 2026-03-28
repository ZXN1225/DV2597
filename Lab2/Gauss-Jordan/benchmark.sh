#!/bin/bash

GREEN="\033[0;32m"
RED="\033[0;31m"
END="\033[0m"

echo "Testing accuracy test with Gaussian elimination"
echo "---------------------------------"

# Ensure executables are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sequential_executable> <parallel_executable>"
    exit 1
fi

seq_executable=$1
par_executable=$2

# Run tests
for n in 2 4 8 16 32 64 128 256 512 1024 2048; do
    echo "Testing with n=$n"

    # Run the sequential and parallel versions
    seq_result=$($seq_executable -n $n -P 1 -I fast)
    par_result=$($par_executable -n $n -P 1 -I fast)

    # Extract "Matrix A"
    seq_matrix_a=$(echo "$seq_result" | grep -A $n "Matrix A:" | grep -v "Matrix A:")
    par_matrix_a=$(echo "$par_result" | grep -A $n "Matrix A:" | grep -v "Matrix A:")
    
    # Extract "Vector b"
    seq_vector_b=$(echo "$seq_result" | grep -A 1 "Vector b:" | grep -v "Vector b:")
    par_vector_b=$(echo "$par_result" | grep -A 1 "Vector b:" | grep -v "Vector b:")
    

    # Extract "Vector y"
    seq_vector_y=$(echo "$seq_result" | grep -A 1 "Vector y:" | grep -v "Vector y:")
    par_vector_y=$(echo "$par_result" | grep -A 1 "Vector y:" | grep -v "Vector y:")
    

    # Compare results
    matrix_a_result=$(diff <(echo "$seq_matrix_a") <(echo "$par_matrix_a") > /dev/null && echo "Passed" || echo "Failed")
    vector_b_result=$(diff <(echo "$seq_vector_b") <(echo "$par_vector_b") > /dev/null && echo "Passed" || echo "Failed")
    vector_y_result=$(diff <(echo "$seq_vector_y") <(echo "$par_vector_y") > /dev/null && echo "Passed" || echo "Failed")

    # Summary
    echo "Comparison results for n=$n:"
    if [ "$matrix_a_result" == "Passed" ]; then
    echo -e "- Matrix A: ${GREEN}$matrix_a_result${END}"
    else
    echo -e "- Matrix A: ${RED}$matrix_a_result${END}"
    fi

    if [ "$vector_b_result" == "Passed" ]; then
    echo -e "- Vector b: ${GREEN}$vector_b_result${END}"
    else
    echo -e "- Vector b: ${RED}$vector_b_result${END}"
    fi

    if [ "$vector_y_result" == "Passed" ]; then
    echo -e "- Vector y: ${GREEN}$vector_y_result${END}"
    else
    echo -e "- Vector y: ${RED}$vector_y_result${END}"
    fi
done
