/* qsortpthread.c — Task 11: pthread quicksort (work-first, spawn-on-large) */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#define KILO (1024U)
#define MEGA (1024U*1024U)
#define N_ITEMS (64U*MEGA)          /* problem size required by the lab */

static int *A;                      /* array */
static int  MAX_THREADS = 16;       /* set with -t N */

/* Cutoffs */
static const unsigned SEQ_CUTOFF = 64;       /* insertion sort threshold */
static const unsigned PAR_CUTOFF = 1u<<17;   /* spawn only if range >= 131072 */

/* ------------ small helpers ------------ */
static inline void isort(int *a, unsigned lo, unsigned hi) {
    for (unsigned i = lo + 1; i <= hi; ++i) {
        int key = a[i];
        unsigned j = i;
        while (j > lo && a[j-1] > key) { a[j] = a[j-1]; --j; }
        a[j] = key;
    }
}

static inline unsigned median3_index(int *a, unsigned lo, unsigned hi) {
    unsigned mid = lo + ((hi - lo) >> 1);
    int x = a[lo], y = a[mid], z = a[hi];
    return (x < y) ? ((y < z) ? mid : (x < z ? hi : lo))
                   : ((x < z) ? lo  : (y < z ? hi : mid));
}

static inline unsigned partition_m3(int *v, unsigned lo, unsigned hi) {
    unsigned pidx = median3_index(v, lo, hi);
    int pv = v[pidx];
    int tmp; tmp = v[lo]; v[lo] = v[pidx]; v[pidx] = tmp; /* pivot to lo */
    unsigned i = lo + 1, j = hi;
    while (1) {
        while (i <= hi && v[i] <= pv) ++i;
        while (v[j] > pv)            --j;
        if (i >= j) break;
        tmp = v[i]; v[i] = v[j]; v[j] = tmp;
    }
    tmp = v[lo]; v[lo] = v[j]; v[j] = tmp; /* place pivot */
    return j;
}

/* sequential quicksort, tail-recursive, with insertion cutoff */
static void qsort_seq_opt(int *v, unsigned lo, unsigned hi) {
    while (hi > lo) {
        unsigned n = hi - lo + 1;
        if (n <= SEQ_CUTOFF) { isort(v, lo, hi); return; }
        unsigned p = partition_m3(v, lo, hi);
        /* recurse on smaller half to keep stack small */
        if (p - lo < hi - p) { if (p) qsort_seq_opt(v, lo, p - 1); lo = p + 1; }
        else                  { if (p+1 <= hi) qsort_seq_opt(v, p + 1, hi); hi = p - 1; }
    }
}

/* ------------ pthread quicksort ------------ */
typedef struct { unsigned lo, hi, used; int *a; int max; int from_heap;} QArgs;

static void* qsort_parallel(void *arg) {
    QArgs *qa = (QArgs*)arg;
    int *v = qa->a;
    unsigned lo = qa->lo, hi = qa->hi;
    unsigned used = qa->used;
    int max = qa->max;

    /* tail-recursive parallel quicksort */
    while (hi > lo) {
        unsigned n = hi - lo + 1;
        if (n <= SEQ_CUTOFF) { isort(v, lo, hi); break; }

        unsigned p = partition_m3(v, lo, hi);
        unsigned l_lo = lo,     l_hi = (p ? p - 1 : p);
        unsigned r_lo = p + 1,  r_hi = hi;
        unsigned l_n  = (l_hi >= l_lo) ? (l_hi - l_lo + 1) : 0;
        unsigned r_n  = (r_hi >= r_lo) ? (r_hi - r_lo + 1) : 0;

        /* choose larger half */
        unsigned Llo=l_lo, Lhi=l_hi, Slo=r_lo, Shi=r_hi; unsigned Ln=l_n, Sn=r_n;
        if (l_n < r_n) { Llo=r_lo; Lhi=r_hi; Ln=r_n; Slo=l_lo; Shi=l_hi; Sn=l_n; }

        int spawned = 0;
        pthread_t tid;

        /* spawn on larger half only if big enough and we have thread budget */
        if (Ln >= PAR_CUTOFF && used < (unsigned)max) {
            QArgs *child = (QArgs*)malloc(sizeof(QArgs));
            if (child) {
                child->a = v; child->lo = Llo; child->hi = Lhi;
                child->used = used + 1; child->max = max;
                child->from_heap = 1;
                if (pthread_create(&tid, NULL, qsort_parallel, child) == 0)
                    spawned = 1;
                else
                    free(child);
            }
        }

            if (spawned) {
            /* We spawned a child on the larger half (Llo..Lhi).
               Now handle the smaller half *also* with the parallel algorithm,
               so it can spawn further if needed. */
            if (Sn) {
                QArgs sub = {
                    .lo = Slo,
                    .hi = Shi,
                    .used = used,
                    .a = v,
                    .max = max,
                    .from_heap = 0
                };
                qsort_parallel(&sub);
            }
            pthread_join(tid, NULL);
            break;  /* both halves sorted, we're done with this range */
        } else {
            /* No spawn (Ln too small or thread budget used):
               fall back to classic tail-recursive sequential quicksort:
               sort smaller half, then loop on larger half. */
            if (Sn) qsort_seq_opt(v, Slo, Shi);
            lo = Llo;
            hi = Lhi;
        }

    }
    if (qa->from_heap) free(arg); /* free only heap children */
    return NULL;
}

/* ---------- driver, CLI, and checks ---------- */
static void init_array(void) {
    A = (int*)malloc((size_t)N_ITEMS * sizeof(int));
    if (!A) { perror("malloc"); exit(1); }
    for (unsigned i = 0; i < N_ITEMS; ++i) A[i] = rand();
}

static int check_sorted(int *a, unsigned n) {
    for (unsigned i=1;i<n;++i) if (a[i-1] > a[i]) return 0;
    return 1;
}

static void usage(const char* prog){
    fprintf(stderr, "Usage: %s [-t threads]\n", prog);
    exit(1);
}

int main(int argc, char **argv) {
    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i], "-t") && i+1<argc) { MAX_THREADS = atoi(argv[++i]); }
        else usage(argv[0]);
    }
    if (MAX_THREADS < 1) MAX_THREADS = 1;
    srand((unsigned)time(NULL));
    init_array();

    if (MAX_THREADS == 1) {
        qsort_seq_opt(A, 0, N_ITEMS-1);
    } else {
        QArgs root = { .lo=0, .hi=N_ITEMS-1, .used=1, .a=A, .max=MAX_THREADS, .from_heap=0 };
        qsort_parallel(&root);
    }

    if (!check_sorted(A, N_ITEMS)) { fprintf(stderr, "NOT SORTED\n"); free(A); return 1; }
    free(A);
    return 0;
}
