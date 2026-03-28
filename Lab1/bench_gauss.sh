#!/usr/bin/env bash

set -euo pipefail

SEQ=./gaussian_seq
PAR=./gaussian_par

N=${1:-2048}
THREADS=${2:-"1 2 4 8 16"}
INIT=${3:-fast}

# Optional: prefix commands with taskset (or nothing if unset)
TASKSET=${TASKSET:-}

echo "===== Gaussian Elimination Bench (Task 10) ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPU(s): $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
echo "GCC: $($(command -v gcc) --version | head -n1)"
echo "Matrix Size: ${N}x${N}"
echo "Init: $INIT"
echo

# Sequential baseline (one time)
SEQ_TIME=$(/usr/bin/time -f "%e" $TASKSET "$SEQ" -n "$N" -I "$INIT" -P 0 2>&1 >/dev/null)
echo "Sequential time (s): $SEQ_TIME"
echo

echo "Threads, Time(s), Speedup_vs_seq, Speedup_vs_t1"

T1_TIME=""

for t in $THREADS; do
  CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 999)
  if [[ "$t" -gt "$CORES" ]]; then
    continue
  fi

  # Parallel time with t threads
  TIME=$(/usr/bin/time -f "%e" $TASKSET "$PAR" -n "$N" -I "$INIT" -P 0 -t "$t" 2>&1 >/dev/null)

  # record time for t=1 as baseline
  [[ -z "$T1_TIME" ]] && T1_TIME="$TIME"

  python3 - "$t" "$TIME" "$SEQ_TIME" "$T1_TIME" << 'PY'
import sys
t, p, s, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
speedup_vs_seq = s / p   # seq_time / par_time
speedup_vs_t1  = t1 / p  # par_time_t1 / par_time_t
print(f"{t}, {p:.3f}, {speedup_vs_seq:.3f}, {speedup_vs_t1:.3f}")
PY

done

echo
