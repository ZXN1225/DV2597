#!/usr/bin/env bash
set -euo pipefail

SEQ=./qsort_seq          # sequential quicksort (qsortseq.c)
PAR=./qsort_par          # pthread quicksort (qsortpthread.c)

THREADS=${1:-"1 2 4 8 16 32"}   # which thread counts to test

echo "===== Parallel Quicksort Bench (Task 11) ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPU(s): $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
echo "GCC: $($(command -v gcc) --version | head -n1)"
echo "Problem Size: 64 * 2^20 items"
echo

# Sequential baseline
SEQ_TIME=$(/usr/bin/time -f "%e" "$SEQ" 2>&1 >/dev/null)
echo "Sequential time (s): $SEQ_TIME"
echo
echo "Threads, Time(s), Speedup_vs_seq, Speedup_vs_t1"

T1_TIME=""

for t in $THREADS; do
  CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 999)
  if [[ "$t" -gt "$CORES" ]]; then
    continue
  fi

  # IMPORTANT: pass -t to parallel version
  TIME=$(/usr/bin/time -f "%e" "$PAR" -t "$t" 2>&1 >/dev/null)

  # t=1 defines our "parallel(1)" baseline for second speedup column
  if [[ "$t" -eq 1 ]]; then
    T1_TIME="$TIME"
  fi
  [[ -z "$T1_TIME" ]] && T1_TIME="$TIME"

  python3 - "$t" "$TIME" "$SEQ_TIME" "$T1_TIME" << 'PY'
import sys
t, p, s, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
speedup_vs_seq = s / p      # T_seq / T_par
speedup_vs_t1  = t1 / p     # T_par(1) / T_par(t)
print(f"{t}, {p:.3f}, {speedup_vs_seq:.3f}, {speedup_vs_t1:.3f}")
PY
done

echo
