/* src/spmv_mpi_omp.c
 *
 * MPI + OpenMP SpMV with 1D modulo-cyclic row distribution:
 *    owner(i) = i mod P
 *
 * REQUIRED by deliverable:
 *  - Rank0 reads MatrixMarket and distributes entries (baseline).
 *  - 1D modulo row partitioning.
 *  - Communication to fetch remote x_j (ghost entries) before SpMV.
 *
 * UNIVERSAL + STABLE scaling solution:
 *  - Two vector exchange strategies:
 *      (A) HALO (sparse): build ghost plan once, exchange values via MPI_Alltoallv each iter
 *      (B) ALLGATHER (dense): MPI_Allgatherv each iter to replicate full x
 *    AUTO picks the cheaper in estimated communication bytes (once).
 *
 * Strong scaling:
 *   mpirun -np P ./spmv_mpi_omp matrix.mtx
 *
 * Weak scaling (synthetic):
 *   mpirun -np P ./spmv_mpi_omp --weak rows_per_rank nnz_per_row [seed]
 *
 * Env:
 *   L          = timed iterations (default 1)
 *   XMODE      = AUTO | HALO | ALLGATHER   (default AUTO)
 *   OMP_NUM_THREADS (usual)
 *
 * Output (rank0):
 *   RESULT total_ms=... comm_ms=... comp_ms=... gflops=... bytes_per_iter=...
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------ Utilities ------------------------ */

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static int owner_mod(int idx, int P) {
    int o = idx % P;
    return (o < 0) ? (o + P) : o;
}

/* For modulo-cyclic mapping: indices owned by rank are {rank + q*P}. */
static int local_len_mod(int n, int rank, int P) {
    if (n <= 0) return 0;
    if (rank < 0) return 0;
    if (rank >= n) return 0;
    return (n - rank + P - 1) / P; /* ceil((n-rank)/P) */
}
static int local_pos_mod(int global_index, int rank, int P) {
    /* valid only if owner_mod(global_index,P)==rank */
    return (global_index - rank) / P;
}

/* ------------------------ CSR ------------------------ */

typedef struct {
    int nrows_global, ncols_global;
    int nrows_local;
    int nnz_local;
    int *row_ptr;   /* size nrows_local + 1 */
    int *col_idx;   /* size nnz_local (GLOBAL col indices) */
    double *values; /* size nnz_local */
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
    if (!A.row_ptr || (nnz_local > 0 && (!A.col_idx || !A.values)))
        die("Allocation failed (CSR).");

    for (int k = 0; k < nnz_local; k++) {
        int gi = coo_r[k];
        int li = local_pos_mod(gi, rank, P);
        if (li < 0 || li >= A.nrows_local) die("Bad local row mapping.");
        A.row_ptr[li + 1]++;
    }
    for (int i = 0; i < A.nrows_local; i++) A.row_ptr[i + 1] += A.row_ptr[i];

    int *offset = (int*)calloc((size_t)A.nrows_local, sizeof(int));
    if (!offset) die("Allocation failed (offset).");

    for (int k = 0; k < nnz_local; k++) {
        int gi = coo_r[k];
        int li = local_pos_mod(gi, rank, P);
        int dest = A.row_ptr[li] + offset[li]++;
        A.col_idx[dest] = coo_c[k];
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
        (*row)[i] = r - 1;
        (*col)[i] = c - 1;
        (*val)[i] = v;
    }
    fclose(f);
}

/* ------------------------ Distribute COO (mod rows) ------------------------ */

static void distribute_coo_modrows(int NNZ,
                                   int *row_g, int *col_g, double *val_g,
                                   int rank, int P,
                                   int *nnz_local_out,
                                   int **row_l, int **col_l, double **val_l)
{
    int *counts = (int*)calloc((size_t)P, sizeof(int));
    int *displs = (int*)calloc((size_t)P, sizeof(int));
    if (!counts || !displs) die("Allocation failed (counts/displs).");

    if (rank == 0) {
        for (int k = 0; k < NNZ; k++) counts[owner_mod(row_g[k], P)]++;
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r - 1] + counts[r - 1];
    }

    int nnz_local = 0;
    MPI_Scatter(counts, 1, MPI_INT, &nnz_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *nnz_local_out = nnz_local;
    *row_l = (int*)malloc((size_t)nnz_local * sizeof(int));
    *col_l = (int*)malloc((size_t)nnz_local * sizeof(int));
    *val_l = (double*)malloc((size_t)nnz_local * sizeof(double));
    if (nnz_local > 0 && (!*row_l || !*col_l || !*val_l))
        die("Allocation failed (COO local).");

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

    if (rank == 0) { free(row_pack); free(col_pack); free(val_pack); }
    free(counts); free(displs);
}

/* ------------------------ Weak matrix (LOCAL generation) ------------------------ */
/* FIXED weak generator: stable communication (no all-to-all explosion).
 *
 * Old issue: choosing col = rand()%N makes each rank talk to ~all ranks (Alltoallv dense),
 * and comm_ms explodes as P grows (especially beyond 32 ranks).
 *
 * New pattern:
 *  - keep nnz_per_row constant
 *  - keep a constant set of remote neighbor ranks (ring ±1) -> constant peer-count
 *  - keep a constant mix local/remote -> constant ghost volume per rank
 */
static void gen_random_coo_local(int rows_per_rank, int nnz_per_row,
                                 int M, int N,
                                 int rank, int P,
                                 int seed,
                                 int *nnz_local_out,
                                 int **row_l, int **col_l, double **val_l)
{
    int my_rows = local_len_mod(M, rank, P); /* should equal rows_per_rank when M=rows_per_rank*P */
    (void)rows_per_rank;

    int nnz_local = my_rows * nnz_per_row;

    *nnz_local_out = nnz_local;
    *row_l = (int*)malloc((size_t)nnz_local * sizeof(int));
    *col_l = (int*)malloc((size_t)nnz_local * sizeof(int));
    *val_l = (double*)malloc((size_t)nnz_local * sizeof(double));
    if (nnz_local > 0 && (!*row_l || !*col_l || !*val_l))
        die("Allocation failed (weak COO local).");

    /* rank-dependent deterministic RNG */
    unsigned int s = (unsigned int)seed ^ (unsigned int)(0x9e3779b9u * (unsigned int)(rank + 1));
    srand(s);

    /* constant (w.r.t. P) number of neighbor ranks used for remote cols */
    const int use_neighbors = (P >= 2) ? 1 : 0;

    /* stable local/remote split (about 50% local) */
    const int local_quota = (nnz_per_row >= 4) ? (nnz_per_row / 2) : 1;

    int k = 0;
    for (int li = 0; li < my_rows; li++) {
        int gi = rank + li * P;  /* global row id owned by this rank */

        /* Strong diagonal (always present) */
        int diag_col = (gi < N) ? gi : (gi % N);
        (*row_l)[k] = gi;
        (*col_l)[k] = diag_col;
        (*val_l)[k] = 10.0;
        k++;

        for (int t = 1; t < nnz_per_row; t++) {
            int col;

            if (P == 1) {
                col = rand() % N;
            } else if (t <= local_quota) {
                /* LOCAL: pick a column owned by this rank (owner(col)=rank) */
                int qmax = (N + P - 1) / P;
                int q = (qmax > 0) ? (rand() % qmax) : 0;
                long long cand = (long long)rank + (long long)q * (long long)P;
                col = (cand < N) ? (int)cand : (int)(cand % N);

                /* enforce exact ownership */
                col = col - owner_mod(col, P) + rank;
                if (col < 0) col += P;
                if (col >= N) col %= N;
                if (owner_mod(col, P) != rank) col = rank % N;
            } else {
                /* REMOTE: restrict to neighbor ranks only (ring) */
                int neighbor_rank;
                if (!use_neighbors) {
                    neighbor_rank = rank;
                } else {
                    /* alternate between +1 and -1 to keep 2 peers max */
                    neighbor_rank = (t & 1) ? ((rank + 1) % P) : ((rank - 1 + P) % P);
                }

                int qmax = (N + P - 1) / P;
                int q = (qmax > 0) ? (rand() % qmax) : 0;
                long long cand = (long long)neighbor_rank + (long long)q * (long long)P;
                col = (cand < N) ? (int)cand : (int)(cand % N);

                /* enforce exact ownership */
                col = col - owner_mod(col, P) + neighbor_rank;
                if (col < 0) col += P;
                if (col >= N) col %= N;
                if (owner_mod(col, P) != neighbor_rank) col = neighbor_rank % N;
            }

            (*row_l)[k] = gi;
            (*col_l)[k] = col;
            (*val_l)[k] = 1.0;
            k++;
        }
    }

    if (k != nnz_local) die("Internal error: weak generator nnz mismatch.");
}

/* ------------------------ HALO (sparse) ghost plan ------------------------ */

typedef struct {
    int n_ghost;
    int *ghost_cols;     /* sorted unique ghost global cols */
    double *ghost_vals;  /* size n_ghost */

    int *send_counts;    /* cols we request from each rank */
    int *recv_counts;    /* requests received from each rank */
    int *sdispls;
    int *rdispls;

    int total_send;
    int total_recv;

    int *req_cols_packed; /* size total_send */
    int *req_pos_packed;  /* size total_send */
    int *recv_req_cols;   /* size total_recv */

    int *col_map;         /* size nnz_local: >=0 local x_local index, <0 ghost index */
    long long bytes_indices_once;

    double *send_vals;    /* size total_recv */
    double *recv_vals;    /* size total_send */
} GhostPlan;

static int cmp_int(const void *a, const void *b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x < y) ? -1 : (x > y);
}
static int lower_bound_int(const int *a, int n, int key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static void free_ghost_plan(GhostPlan *gp) {
    free(gp->ghost_cols);
    free(gp->ghost_vals);
    free(gp->send_counts);
    free(gp->recv_counts);
    free(gp->sdispls);
    free(gp->rdispls);
    free(gp->req_cols_packed);
    free(gp->req_pos_packed);
    free(gp->recv_req_cols);
    free(gp->col_map);
    free(gp->send_vals);
    free(gp->recv_vals);
    memset(gp, 0, sizeof(*gp));
}

static void build_ghost_plan(const CSRLocal *A, int rank, int P, GhostPlan *gp)
{
    memset(gp, 0, sizeof(*gp));

    int *tmp = NULL;
    int n = 0;
    if (A->nnz_local > 0) {
        tmp = (int*)malloc((size_t)A->nnz_local * sizeof(int));
        if (!tmp) die("Allocation failed (tmp ghost).");
        for (int k = 0; k < A->nnz_local; k++) {
            int c = A->col_idx[k];
            if (owner_mod(c, P) != rank) tmp[n++] = c;
        }
    }

    gp->send_counts = (int*)calloc((size_t)P, sizeof(int));
    gp->recv_counts = (int*)calloc((size_t)P, sizeof(int));
    gp->sdispls     = (int*)calloc((size_t)P, sizeof(int));
    gp->rdispls     = (int*)calloc((size_t)P, sizeof(int));
    gp->col_map     = (int*)malloc((size_t)A->nnz_local * sizeof(int));
    if (!gp->send_counts || !gp->recv_counts || !gp->sdispls || !gp->rdispls || (A->nnz_local > 0 && !gp->col_map))
        die("Allocation failed (ghost plan base).");

    if (n == 0) {
        for (int k = 0; k < A->nnz_local; k++) {
            int c = A->col_idx[k];
            gp->col_map[k] = local_pos_mod(c, rank, P);
        }
        free(tmp);
        return;
    }

    qsort(tmp, (size_t)n, sizeof(int), cmp_int);

    int uniq = 1;
    for (int i = 1; i < n; i++) if (tmp[i] != tmp[i-1]) uniq++;

    gp->n_ghost = uniq;
    gp->ghost_cols = (int*)malloc((size_t)uniq * sizeof(int));
    gp->ghost_vals = (double*)malloc((size_t)uniq * sizeof(double));
    if (!gp->ghost_cols || !gp->ghost_vals) die("Allocation failed (ghost arrays).");

    int w = 0;
    gp->ghost_cols[w++] = tmp[0];
    for (int i = 1; i < n; i++) if (tmp[i] != tmp[i-1]) gp->ghost_cols[w++] = tmp[i];
    free(tmp);

    for (int i = 0; i < gp->n_ghost; i++) gp->ghost_vals[i] = 0.0;

    for (int i = 0; i < gp->n_ghost; i++) gp->send_counts[owner_mod(gp->ghost_cols[i], P)]++;
    gp->send_counts[rank] = 0;

    MPI_Alltoall(gp->send_counts, 1, MPI_INT, gp->recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    for (int p = 1; p < P; p++) {
        gp->sdispls[p] = gp->sdispls[p-1] + gp->send_counts[p-1];
        gp->rdispls[p] = gp->rdispls[p-1] + gp->recv_counts[p-1];
    }
    gp->total_send = gp->sdispls[P-1] + gp->send_counts[P-1];
    gp->total_recv = gp->rdispls[P-1] + gp->recv_counts[P-1];

    gp->req_cols_packed = (int*)malloc((size_t)gp->total_send * sizeof(int));
    gp->req_pos_packed  = (int*)malloc((size_t)gp->total_send * sizeof(int));
    gp->recv_req_cols   = (int*)malloc((size_t)gp->total_recv * sizeof(int));
    if ((gp->total_send > 0 && (!gp->req_cols_packed || !gp->req_pos_packed)) ||
        (gp->total_recv > 0 && !gp->recv_req_cols))
        die("Allocation failed (ghost packed).");

    int *cursor = (int*)malloc((size_t)P * sizeof(int));
    if (!cursor) die("Allocation failed (ghost cursor).");
    for (int p = 0; p < P; p++) cursor[p] = gp->sdispls[p];

    for (int pos = 0; pos < gp->n_ghost; pos++) {
        int c = gp->ghost_cols[pos];
        int o = owner_mod(c, P);
        if (o == rank) continue;
        int idx = cursor[o]++;
        gp->req_cols_packed[idx] = c;
        gp->req_pos_packed[idx]  = pos;
    }
    free(cursor);

    MPI_Alltoallv(gp->req_cols_packed, gp->send_counts, gp->sdispls, MPI_INT,
                  gp->recv_req_cols,  gp->recv_counts, gp->rdispls, MPI_INT,
                  MPI_COMM_WORLD);

    for (int k = 0; k < A->nnz_local; k++) {
        int c = A->col_idx[k];
        int o = owner_mod(c, P);
        if (o == rank) {
            gp->col_map[k] = local_pos_mod(c, rank, P);
        } else {
            int id = lower_bound_int(gp->ghost_cols, gp->n_ghost, c);
            if (id >= gp->n_ghost || gp->ghost_cols[id] != c) die("Ghost col not found.");
            gp->col_map[k] = -(id + 1);
        }
    }

    gp->bytes_indices_once =
        (long long)(gp->total_send + gp->total_recv) * (long long)sizeof(int);

    gp->send_vals = (gp->total_recv > 0) ? (double*)malloc((size_t)gp->total_recv * sizeof(double)) : NULL;
    gp->recv_vals = (gp->total_send > 0) ? (double*)malloc((size_t)gp->total_send * sizeof(double)) : NULL;
    if ((gp->total_recv > 0 && !gp->send_vals) || (gp->total_send > 0 && !gp->recv_vals))
        die("Allocation failed (ghost val buffers).");
}

static void exchange_ghost_values(GhostPlan *gp, const double *x_local, int rank, int P)
{
    if (gp->total_send == 0 && gp->total_recv == 0) return;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < gp->total_recv; i++) {
        int col = gp->recv_req_cols[i];
        int o = owner_mod(col, P);
        if (o != rank) die("Received request for non-owned col.");
        int loc = local_pos_mod(col, rank, P);
        gp->send_vals[i] = x_local[loc];
    }

    MPI_Alltoallv(gp->send_vals, gp->recv_counts, gp->rdispls, MPI_DOUBLE,
                  gp->recv_vals, gp->send_counts, gp->sdispls, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    for (int i = 0; i < gp->total_send; i++) {
        int pos = gp->req_pos_packed[i];
        gp->ghost_vals[pos] = gp->recv_vals[i];
    }
}

/* ------------------------ SpMV kernels ------------------------ */

static void spmv_local_xfull(const CSRLocal *A, const double *x_full, double *y_local)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int li = 0; li < A->nrows_local; li++) {
        double sum = 0.0;
        for (int kk = A->row_ptr[li]; kk < A->row_ptr[li + 1]; kk++) {
            int j = A->col_idx[kk];
            sum += A->values[kk] * x_full[j];
        }
        y_local[li] = sum;
    }
}

static void spmv_local_halo(const CSRLocal *A, const GhostPlan *gp,
                           const double *x_local, double *y_local)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int li = 0; li < A->nrows_local; li++) {
        double sum = 0.0;
        for (int kk = A->row_ptr[li]; kk < A->row_ptr[li + 1]; kk++) {
            int m = gp->col_map[kk];
            double xj = (m >= 0) ? x_local[m] : gp->ghost_vals[-m - 1];
            sum += A->values[kk] * xj;
        }
        y_local[li] = sum;
    }
}

/* ------------------------ Main ------------------------ */

typedef enum { XM_AUTO=0, XM_HALO=1, XM_ALLGATHER=2 } XMode;

static XMode parse_xmode(const char *s) {
    if (!s || !*s) return XM_AUTO;
    if (strcmp(s, "AUTO") == 0) return XM_AUTO;
    if (strcmp(s, "HALO") == 0) return XM_HALO;
    if (strcmp(s, "ALLGATHER") == 0) return XM_ALLGATHER;
    return XM_AUTO;
}

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

    int nnz_local = 0;
    int *row_l = NULL, *col_l = NULL;
    double *val_l = NULL;

    if (!weak_mode) {
        if (rank == 0) {
            read_matrix_market_rank0(argv[1], &M, &N, &NNZ, &row_g, &col_g, &val_g);
            printf("Read global matrix %dx%d with %d nnz\n", M, N, NNZ);
            fflush(stdout);
        }
        MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&NNZ, 1, MPI_INT, 0, MPI_COMM_WORLD);

        distribute_coo_modrows(NNZ, row_g, col_g, val_g, rank, P,
                               &nnz_local, &row_l, &col_l, &val_l);

        if (rank == 0) { free(row_g); free(col_g); free(val_g); }
    } else {
        M = rows_per_rank * P;
        N = M;
        NNZ = M * nnz_per_row;

        gen_random_coo_local(rows_per_rank, nnz_per_row, M, N, rank, P, seed,
                             &nnz_local, &row_l, &col_l, &val_l);

        if (rank == 0) {
            printf("Generated WEAK matrix %dx%d with %d nnz (rows_per_rank=%d, nnz_per_row=%d, seed=%d)\n",
                   M, N, NNZ, rows_per_rank, nnz_per_row, seed);
            fflush(stdout);
        }
    }

    CSRLocal A = coo_to_local_csr_modrows(M, N, nnz_local, row_l, col_l, val_l, rank, P);
    free(row_l); free(col_l); free(val_l);

    /* NNZ stats */
    int nnz_min = 0, nnz_max = 0;
    long long nnz_sum_ll = 0;
    MPI_Reduce(&A.nnz_local, &nnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&A.nnz_local, &nnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    long long nnz_loc_ll = (long long)A.nnz_local;
    MPI_Reduce(&nnz_loc_ll, &nnz_sum_ll, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double nnz_avg = (double)nnz_sum_ll / (double)P;
        printf("[BONUS] NNZ per rank min/avg/max: %d / %.2f / %d\n", nnz_min, nnz_avg, nnz_max);
        fflush(stdout);
    }

    /* x ownership (modulo columns, same rule) */
    int x_local_len = local_len_mod(N, rank, P);
    double *x_local = (double*)malloc((size_t)x_local_len * sizeof(double));
    if (x_local_len > 0 && !x_local) die("Allocation failed (x_local).");

    /* Deterministic x per rank (but consistent with ownership) */
    unsigned int sx = 1234567u ^ (unsigned int)(seed * 2654435761u) ^ (unsigned int)(rank + 1);
    srand(sx);
    for (int lj = 0; lj < x_local_len; lj++) {
        x_local[lj] = -1000.0 + 2000.0 * ((double)rand() / RAND_MAX);
    }

    /* Precompute allgatherv metadata */
    int *x_counts = (int*)malloc((size_t)P * sizeof(int));
    int *x_displs = (int*)malloc((size_t)P * sizeof(int));
    if (!x_counts || !x_displs) die("Allocation failed (x meta).");
    int disp = 0;
    for (int r = 0; r < P; r++) {
        x_counts[r] = local_len_mod(N, r, P);
        x_displs[r] = disp;
        disp += x_counts[r];
    }
    if (disp != N) die("Allgatherv meta mismatch (disp != N).");

    double *x_full = (double*)malloc((size_t)N * sizeof(double));
    if (N > 0 && !x_full) die("Allocation failed (x_full).");

    /* Build HALO plan */
    GhostPlan gp;
    build_ghost_plan(&A, rank, P, &gp);

    /* Decide strategy */
    XMode mode = parse_xmode(getenv("XMODE"));

    double bytes_allgather = (double)N * (double)sizeof(double);
    double bytes_halo_vals = (double)(gp.total_send + gp.total_recv) * (double)sizeof(double);
    int use_halo = 0;

    if (mode == XM_HALO) use_halo = 1;
    else if (mode == XM_ALLGATHER) use_halo = 0;
    else {
        use_halo = (bytes_halo_vals < bytes_allgather) ? 1 : 0;
    }

    double *y_local = (double*)malloc((size_t)A.nrows_local * sizeof(double));
    if (A.nrows_local > 0 && !y_local) die("Allocation failed (y_local).");

    int L = 1;
    const char *Lenv = getenv("L");
    if (Lenv) L = atoi(Lenv);
    if (L < 1) L = 1;

    /* Warmup (not timed) */
    if (use_halo) {
        exchange_ghost_values(&gp, x_local, rank, P);
        spmv_local_halo(&A, &gp, x_local, y_local);
    } else {
        MPI_Allgatherv(x_local, x_local_len, MPI_DOUBLE, x_full, x_counts, x_displs, MPI_DOUBLE, MPI_COMM_WORLD);
        spmv_local_xfull(&A, x_full, y_local);
    }

    double comm_sum = 0.0, comp_sum = 0.0, total_sum = 0.0;

    for (int it = 0; it < L; it++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        double c0 = MPI_Wtime();
        if (use_halo) {
            exchange_ghost_values(&gp, x_local, rank, P);
        } else {
            MPI_Allgatherv(x_local, x_local_len, MPI_DOUBLE, x_full, x_counts, x_displs, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        double c1 = MPI_Wtime();

        double p0 = MPI_Wtime();
        if (use_halo) spmv_local_halo(&A, &gp, x_local, y_local);
        else          spmv_local_xfull(&A, x_full, y_local);
        double p1 = MPI_Wtime();

        double t1 = MPI_Wtime();

        comm_sum  += (c1 - c0);
        comp_sum  += (p1 - p0);
        total_sum += (t1 - t0);
    }

    /* critical path = MAX over ranks */
    double comm_max = 0.0, comp_max = 0.0, total_max = 0.0;
    MPI_Reduce(&comm_sum,  &comm_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_sum,  &comp_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_sum, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double iters = (double)L;
        double t_comm_ms  = (comm_max  / iters) * 1000.0;
        double t_comp_ms  = (comp_max  / iters) * 1000.0;
        double t_total_ms = (total_max / iters) * 1000.0;

        long long nnz_global = nnz_sum_ll;
        double flops = 2.0 * (double)nnz_global;
        double sec = t_total_ms / 1000.0;
        double gflops = (sec > 0.0) ? (flops / sec / 1e9) : 0.0;

        double bytes_per_iter = 0.0;
        if (use_halo) {
            double bytes_values = bytes_halo_vals;
            double bytes_idx_amort = (L > 0) ? ((double)gp.bytes_indices_once / (double)L) : 0.0;
            bytes_per_iter = bytes_values + bytes_idx_amort;
        } else {
            bytes_per_iter = bytes_allgather;
        }

#ifdef _OPENMP
        int threads = omp_get_max_threads();
#else
        int threads = 1;
#endif
        const char *mname = use_halo ? "HALO" : "ALLGATHER";
        printf("RESULT total_ms=%.6f comm_ms=%.6f comp_ms=%.6f gflops=%.6f bytes_per_iter=%.2f "
               "P=%d threads=%d M=%d N=%d nnz=%lld nnz_min=%d nnz_max=%d L=%d XMODE=%s\n",
               t_total_ms, t_comm_ms, t_comp_ms, gflops, bytes_per_iter,
               P, threads, M, N, nnz_global, nnz_min, nnz_max, L, mname);
        fflush(stdout);
    }

    free(y_local);
    free(x_full);
    free(x_displs);
    free(x_counts);
    free(x_local);
    free_ghost_plan(&gp);
    free_csr(&A);

    MPI_Finalize();
    return 0;
}

