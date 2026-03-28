CC       := gcc
CFLAGS   := -O3 -march=native -flto -Wall
PTHREAD  := -pthread

# Build everything you need for the lab
all: gaussian_seq gaussian_par qsort_seq qsort_par matmul_seq \
     bankaccount falsesharing pthreadcreate1 pthreadcreate2 pthreadcreate3

# ----- Task 10 ------
gaussian_seq: gaussianseq.c
	$(CC) $(CFLAGS) $< -o $@

gaussian_par: gaussianpar.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

# ----- Task 11 ----
qsort_seq: qsortseq.c
	$(CC) $(CFLAGS) $< -o $@

qsort_par: qsortpthread.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

# ----- Other files -----
matmul_seq: matmulseq.c
	$(CC) $(CFLAGS) $< -o $@

bankaccount: bankaccount.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

falsesharing: falsesharing.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

pthreadcreate1: pthreadcreate.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

pthreadcreate2: pthreadcreate2.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

pthreadcreate3: pthreadcreate3.c
	$(CC) $(CFLAGS) $(PTHREAD) $< -o $@

clean:
	rm -f gaussian_seq gaussian_par qsort_seq qsort_par \
	      matmul_seq bankaccount falsesharing pthreadcreate1 pthreadcreate2 pthreadcreate3
