/* src/spmv_mpi_omp.c
 * MPI SpMV with 1D modulo-cyclic row distribution (owner(i)=i mod P).
 * Option A: replicate x on all ranks via MPI_Bcast (no ghost exchange).
 *
 * Output (rank 0):
 *  - Read/Generated matrix line
 *  - [BONUS] NNZ per rank min/avg/max
 *  - Iter k: SpMV time (max over ranks): <sec> s
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------ Utilities ------------------------ */

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static int owner_mod(int global_index, int P) { return global_index % P; }

/* For modulo-cyclic mapping: indices owned by rank are {rank + q*P}. */
static int local_len_mod(int n, int rank, int P) {
    if (rank >= n) return 0;
    return (n - rank + P - 1) / P; /* ceil((n-rank)/P) */
}
static int local_pos_mod(int global_index, int rank, int P) {
    /* valid only if global_index % P == rank */
    return (global_index - rank) / P;
}

/* ------------------------ CSR ------------------------ */

typedef struct {
    int nrows_global, ncols_global;
    int nrows_local;     /* number of rows owned by this rank */
    int nnz_local;
    int *row_ptr;        /* size nrows_local + 1 */
    int *col_idx;        /* size nnz_local (global column indices) */
    double *values;      /* size nnz_local */
} CSRLocal;

static CSRLocal coo_to_local_csr_modrows(int M, int N, int nnz_local,
                                        const int *coo_r, const int *coo_c, const double *coo_v,
                                        int rank, int P)
{
    CSRLocal A;
    A.nrows_global = M;
    A.ncols_global = N;
    A.nrows_local  = local_len_mod(M, rank, P);
    A.nnz_local    = nnz_local;

    A.row_ptr = (int*)calloc((size_t)A.nrows_local + 1, sizeof(int));
    A.col_idx = (int*)malloc((size_t)nnz_local * sizeof(int));
    A.values  = (double*)malloc((size_t)nnz_local * sizeof(double));
    if (!A.row_ptr || (!A.col_idx && nnz_local > 0) || (!A.values && nnz_local > 0))
        die("Allocation failed (CSR).");

    /* Count nnz per local row */
    for (int k = 0; k < nnz_local; k++) {
        int gi = coo_r[k];
        int li = local_pos_mod(gi, rank, P); /* row belongs to rank by construction */
        if (li < 0 || li >= A.nrows_local) die("Bad local row mapping.");
        A.row_ptr[li + 1]++;
    }

    /* Prefix sum */
    for (int i = 0; i < A.nrows_local; i++) A.row_ptr[i + 1] += A.row_ptr[i];

    int *offset = (int*)calloc((size_t)A.nrows_local, sizeof(int));
    if (!offset) die("Allocation failed (offset).");

    /* Fill */
    for (int k = 0; k < nnz_local; k++) {
        int gi = coo_r[k];
        int li = local_pos_mod(gi, rank, P);
        int dest = A.row_ptr[li] + offset[li]++;
        A.col_idx[dest] = coo_c[k]; /* global col */
        A.values[dest]  = coo_v[k];
    }
    free(offset);
    return A;
}

static void free_csr(CSRLocal *A) {
    free(A->row_ptr);
    free(A->col_idx);
    free(A->values);
    A->row_ptr = NULL; A->col_idx = NULL; A->values = NULL;
}

/* ------------------------ Matrix Market (rank 0) ------------------------ */

static void read_matrix_market_rank0(const char *filename, int *M, int *N, int *NNZ,
                                    int **row, int **col, double **val)
{
    FILE *f = fopen(filename, "r");
    if (!f) die("Error: cannot open Matrix Market file.");

    char line[256];
    do {
        if (!fgets(line, sizeof(line), f)) die("Error: cannot read Matrix Market header.");
    } while (line[0] == '%');

    if (sscanf(line, "%d %d %d", M, N, NNZ) != 3) die("Error: bad Matrix Market size line.");

    *row = (int*)malloc((size_t)(*NNZ) * sizeof(int));
    *col = (int*)malloc((size_t)(*NNZ) * sizeof(int));
    *val = (double*)malloc((size_t)(*NNZ) * sizeof(double));
    if (!*row || !*col || !*val) die("Allocation failed (COO global).");

    for (int i = 0; i < *NNZ; i++) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) die("Error: bad Matrix Market entry.");
        (*row)[i] = r - 1; /* 0-based */
        (*col)[i] = c - 1;
        (*val)[i] = v;
    }
    fclose(f);
}

/* ------------------------ Distribute COO (mod rows) ------------------------ */

static void distribute_coo_modrows(int M, int N, int NNZ,
                                  int *row_g, int *col_g, double *val_g,
                                  int rank, int P,
                                  int *nnz_local_out, int **row_l, int **col_l, double **val_l)
{
    (void)M; (void)N;

    int *counts = (int*)calloc((size_t)P, sizeof(int));
    int *displs = (int*)calloc((size_t)P, sizeof(int));
    if (!counts || !displs) die("Allocation failed (counts/displs).");

    if (rank == 0) {
        for (int k = 0; k < NNZ; k++) {
            int dest = owner_mod(row_g[k], P);
            counts[dest]++;
        }
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r - 1] + counts[r - 1];
    }

    int nnz_local = 0;
    MPI_Scatter(counts, 1, MPI_INT, &nnz_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *nnz_local_out = nnz_local;
    *row_l = (int*)malloc((size_t)nnz_local * sizeof(int));
    *col_l = (int*)malloc((size_t)nnz_local * sizeof(int));
    *val_l = (double*)malloc((size_t)nnz_local * sizeof(double));
    if ((nnz_local > 0) && (!*row_l || !*col_l || !*val_l)) die("Allocation failed (COO local).");

    int *row_pack = NULL, *col_pack = NULL;
    double *val_pack = NULL;

    if (rank == 0) {
        row_pack = (int*)malloc((size_t)NNZ * sizeof(int));
        col_pack = (int*)malloc((size_t)NNZ * sizeof(int));
        val_pack = (double*)malloc((size_t)NNZ * sizeof(double));
        if (!row_pack || !col_pack || !val_pack) die("Allocation failed (packed COO).");

        int *cursor = (int*)calloc((size_t)P, sizeof(int));
        if (!cursor) die("Allocation failed (cursor).");

        for (int k = 0; k < NNZ; k++) {
            int dest = owner_mod(row_g[k], P);
            int pos  = displs[dest] + cursor[dest]++;
            row_pack[pos] = row_g[k];
            col_pack[pos] = col_g[k];
            val_pack[pos] = val_g[k];
        }
        free(cursor);
    }

    MPI_Scatterv(row_pack, counts, displs, MPI_INT, *row_l, nnz_local, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(col_pack, counts, displs, MPI_INT, *col_l, nnz_local, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(val_pack, counts, displs, MPI_DOUBLE, *val_l, nnz_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(row_pack); free(col_pack); free(val_pack);
    }
    free(counts); free(displs);
}

/* ------------------------ Synthetic matrix (rank 0) ------------------------ */
/* Weak scaling generator: M = rows_per_rank * P, NNZ ~ nnz_per_row * M */
static void gen_random_coo_rank0(int rows_per_rank, int nnz_per_row, int P,
                                 int seed,
                                 int *M, int *N, int *NNZ,
                                 int **row, int **col, double **val)
{
    *M = rows_per_rank * P;
    *N = *M;
    *NNZ = (*M) * nnz_per_row;

    *row = (int*)malloc((size_t)(*NNZ) * sizeof(int));
    *col = (int*)malloc((size_t)(*NNZ) * sizeof(int));
    *val = (double*)malloc((size_t)(*NNZ) * sizeof(double));
    if (!*row || !*col || !*val) die("Allocation failed (weak COO).");

    srand(seed);
    int k = 0;
    for (int i = 0; i < *M; i++) {
        (*row)[k] = i;
        (*col)[k] = i;
        (*val)[k] = 10.0;
        k++;

        for (int t = 1; t < nnz_per_row; t++) {
            int j = rand() % (*N);
            (*row)[k] = i;
            (*col)[k] = j;
            (*val)[k] = 1.0;
            k++;
        }
    }
}

/* ------------------------ Local SpMV (uses replicated x) ------------------------ */

static void spmv_local_replx(const CSRLocal *A, const double *x_full, double *y_local) {
    for (int li = 0; li < A->nrows_local; li++) {
        double sum = 0.0;
        for (int kk = A->row_ptr[li]; kk < A->row_ptr[li + 1]; kk++) {
            int j = A->col_idx[kk];
            sum += A->values[kk] * x_full[j];
        }
        y_local[li] = sum;
    }
}

/* ------------------------ Main ------------------------ */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int weak_mode = 0;
    int rows_per_rank = 0;
    int nnz_per_row = 0;
    int seed = 42;

    if (argc == 2) {
        weak_mode = 0;
    } else if (argc >= 4 && strcmp(argv[1], "--weak") == 0) {
        weak_mode = 1;
        rows_per_rank = atoi(argv[2]);
        nnz_per_row   = atoi(argv[3]);
        if (argc >= 5) seed = atoi(argv[4]);
        if (rows_per_rank <= 0 || nnz_per_row <= 0) die("Bad weak params.");
    } else {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage:\n  %s <matrix.mtx>\n  %s --weak rows_per_rank nnz_per_row [seed]\n",
                    argv[0], argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int M = 0, N = 0, NNZ = 0;
    int *row_g = NULL, *col_g = NULL;
    double *val_g = NULL;

    if (rank == 0) {
        if (!weak_mode) {
            read_matrix_market_rank0(argv[1], &M, &N, &NNZ, &row_g, &col_g, &val_g);
            printf("Read global matrix %dx%d with %d nnz\n", M, N, NNZ);
        } else {
            gen_random_coo_rank0(rows_per_rank, nnz_per_row, P, seed,
                                 &M, &N, &NNZ, &row_g, &col_g, &val_g);
            printf("Generated WEAK matrix %dx%d with %d nnz (rows_per_rank=%d, nnz_per_row=%d)\n",
                   M, N, NNZ, rows_per_rank, nnz_per_row);
        }
        fflush(stdout);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NNZ, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int nnz_local = 0;
    int *row_l = NULL, *col_l = NULL;
    double *val_l = NULL;
    distribute_coo_modrows(M, N, NNZ, row_g, col_g, val_g, rank, P,
                           &nnz_local, &row_l, &col_l, &val_l);

    if (rank == 0) { free(row_g); free(col_g); free(val_g); }

    CSRLocal A = coo_to_local_csr_modrows(M, N, nnz_local, row_l, col_l, val_l, rank, P);
    free(row_l); free(col_l); free(val_l);

    /* Build x on rank 0, replicate via Bcast (not timed) */
    double *x_full = (double*)malloc((size_t)N * sizeof(double));
    if (N > 0 && !x_full) die("Allocation failed (x_full).");

    if (rank == 0) {
        srand(42);
        for (int j = 0; j < N; j++)
            x_full[j] = -1000.0 + 2000.0 * ((double)rand() / RAND_MAX);
    }
    MPI_Bcast(x_full, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Bonus: nnz per rank min/avg/max */
    int nnz_min = 0, nnz_max = 0, nnz_sum = 0;
    MPI_Reduce(&A.nnz_local, &nnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&A.nnz_local, &nnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&A.nnz_local, &nnz_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double nnz_avg = (double)nnz_sum / (double)P;
        printf("[BONUS] NNZ per rank min/avg/max: %d / %.2f / %d\n", nnz_min, nnz_avg, nnz_max);
        fflush(stdout);
    }

    double *y_local = (double*)malloc((size_t)A.nrows_local * sizeof(double));
    if (A.nrows_local > 0 && !y_local) die("Allocation failed (y_local).");

    int L = 1;
    const char *Lenv = getenv("L");
    if (Lenv) L = atoi(Lenv);
    if (L < 1) L = 1;

    /* Warmup (not timed) */
    spmv_local_replx(&A, x_full, y_local);

    for (int it = 1; it <= L; it++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        spmv_local_replx(&A, x_full, y_local);

        double t1 = MPI_Wtime();

        double local_time = t1 - t0;
        double global_time = 0.0;
        MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Iter %d: SpMV time (max over ranks): %.6f s\n", it, global_time);
            fflush(stdout);
        }
    }

    /* Optional gather y (not timed) */
    int *counts_rows = NULL, *displs_rows = NULL;
    double *y_packed = NULL;

    if (rank == 0) {
        counts_rows = (int*)malloc((size_t)P * sizeof(int));
        displs_rows = (int*)malloc((size_t)P * sizeof(int));
        if (!counts_rows || !displs_rows) die("Allocation failed (y gather meta).");
        int total = 0;
        for (int r = 0; r < P; r++) {
            counts_rows[r] = local_len_mod(M, r, P);
            displs_rows[r] = total;
            total += counts_rows[r];
        }
        y_packed = (double*)malloc((size_t)total * sizeof(double));
        if (total > 0 && !y_packed) die("Allocation failed (y_packed).");
    }

    MPI_Gatherv(y_local, A.nrows_local, MPI_DOUBLE,
                y_packed, counts_rows, displs_rows, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double *y_global = (double*)malloc((size_t)M * sizeof(double));
        if (M > 0 && !y_global) die("Allocation failed (y_global).");

        for (int i = 0; i < M; i++) {
            int r = owner_mod(i, P);
            int li = local_pos_mod(i, r, P);
            y_global[i] = y_packed[displs_rows[r] + li];
        }

        if (M <= 10) {
            printf("y (first %d rows):\n", M);
            for (int i = 0; i < M; i++) printf("y[%d]=%.6e\n", i, y_global[i]);
        }

        free(y_global);
        free(y_packed);
        free(counts_rows);
        free(displs_rows);
    }

    free(y_local);
    free(x_full);
    free_csr(&A);

    MPI_Finalize();
    return 0;
}

