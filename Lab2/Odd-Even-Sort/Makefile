THREADS_PER_BLOCK = 1024

SEQ_EXEC = oddevensort_seq
TASK1_EXEC = oddevensort_task1
TASK2_EXEC = oddevensort_task2
BENCH_SCRIPT = benchmark_oddeven.sh

SEQ_SRC = oddevensort.cpp
TASK1_SRC = oddevensort_single_kernel.cu
TASK2_SRC = oddevensort_multi_kernel.cu

# Compilers and Flags
CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++0x -O2
 
NVCCFLAGS = -O2


.PHONY: all clean seq task1 task2 bench

all: $(SEQ_EXEC) $(TASK1_EXEC) $(TASK2_EXEC)

$(SEQ_EXEC): $(SEQ_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Target for Task 1
$(TASK1_EXEC): $(TASK1_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Target for Task 2
$(TASK2_EXEC): $(TASK2_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

bench: all
	@echo "--- Running Benchmark ---"
	@./$(BENCH_SCRIPT)

clean:
	@echo "Cleaning up..."
	@rm -f $(SEQ_EXEC) $(TASK1_EXEC) $(TASK2_EXEC)
