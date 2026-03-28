#!/usr/bin/env bash
set -euo pipefail

rm -f lab1_results.txt

{
  echo "===== Gaussian (Task 10) Sequential and Parallel ====="
  TASKSET="taskset -c 0-15" ./bench_gauss.sh 2048 "1 2 4 8 16"
  echo
  echo "===== Quicksort (Task 11) – pthread version ====="
  ./bench_qsort.sh "1 2 4 8 16"
  echo
} >> lab1_results.txt


echo "Wrote lab1_results.txt"
