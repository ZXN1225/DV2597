# Makefile for Gauss-Jordan Row Reduction
# Task 3 - Lab 2
# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -O2
NVCCFLAGS = -O2
# Target executables
SEQ_TARGET = gauss_sequential
PAR_TARGET = gauss_parallel
# Source files
SEQ_SOURCE = gaussjordanseq.c
PAR_SOURCE = gaussjordanparallel.cu
# Default target
all: $(SEQ_TARGET) $(PAR_TARGET)
# Sequential version
$(SEQ_TARGET): $(SEQ_SOURCE)
	$(CC) $(CFLAGS) -o $(SEQ_TARGET) $(SEQ_SOURCE) -lm
# Parallel CUDA version
$(PAR_TARGET): $(PAR_SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(PAR_TARGET) $(PAR_SOURCE)
# Test with benchmark script
test: all
	chmod +x benchmark.sh
	./benchmark.sh ./$(SEQ_TARGET) ./$(PAR_TARGET)
# Run individual tests
run_seq: $(SEQ_TARGET)
	./$(SEQ_TARGET) -n 2048 -I fast -P 0
run_par: $(PAR_TARGET)
	./$(PAR_TARGET) -n 2048 -I fast -P 0
# Clean build artifacts
clean:
	rm -f $(SEQ_TARGET) $(PAR_TARGET) *.o
# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build both sequential and parallel versions"
	@echo "  test      - Run benchmark tests"
	@echo "  run_seq   - Run sequential version"
	@echo "  run_par   - Run parallel version"
	@echo "  clean     - Remove executables and object files"
	@echo "  help      - Show this help message"
.PHONY: all test run_seq run_par clean help