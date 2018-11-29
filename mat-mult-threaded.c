#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define MAT_ELT(mat, cols, i, j) *(mat + (i * cols) + j)
#define ONE_BILLION (double)1000000000.0
#define BROADCASTER 0

void
usage(char *prog_name)
{
  fprintf(stderr, "usage:%s [flags]\n", prog_name);
  fprintf(stderr, "  -h           print help\n");
  fprintf(stderr, "  -a <input file>       file that contains A matrix\n");
  fprintf(stderr, "  -b <input file>       file that contains B matrix\n");
  fprintf(stderr, "  -c <output file>      file that will contain C matrix\n");
  exit(1);
}

typedef struct{
  int rows;
  int cols;
  int *ptr;
} matrix_t;

typedef struct{
  int cols; // b
  int cols_index;
  int rows; // a
  int rows_index;
  int num_cols_complete;
} parallel_track_t;

void write_matrix(void *matrix_i, char *name){
  matrix_t *matrix = (matrix_t *) matrix_i;

  FILE *file = fopen(name, "write");

  fprintf(file, "%d %d\n", matrix->rows, matrix->cols);
  int index = 0;
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      char end = ' ';
      if (j == (matrix->cols-1)) {
        end = '\n';
      }
      fprintf(file, "%d%c", matrix->ptr[index], end);
      index++;
    }
  }

  fclose(file);
}

void check_array(void *matrix_i, int rank) {

  matrix_t *matrix = (matrix_t *) matrix_i;

  fprintf(stderr, "\n\n%d: %d %d\n", rank, matrix->rows, matrix->cols);

  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      char end_char = ' ';
      if (j == matrix->cols - 1) {
        end_char = '\n';
      }
      fprintf(stderr, "%7d%c", MAT_ELT(matrix->ptr, matrix->cols, i, j), end_char);
    }
  }
}

void *read_file(char *file_name, void *in_matrix){
  FILE *file = NULL;
  file = fopen(file_name,"r");

  if (!file) {
    // fprintf(stderr, "File %s is not readable.\n", file_name);
    return NULL;
  }

  matrix_t *matrix = (matrix_t *) in_matrix;

  fscanf(file, "%d %d\n", &matrix->rows, &matrix->cols);

  int totalSize = matrix->rows * matrix->cols;
  matrix->ptr = malloc(totalSize * sizeof(int));

  for (int i = 0; i < totalSize; i++) {
    fscanf(file,"%d ", (matrix->ptr + i));
  }

  // check_array(matrix, BROADCASTER);

  return (void *) matrix;
}

double
now(void)
{
  struct timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

void copy_mat(void *src, void *dest, int rank){
  matrix_t *matrix_src = (matrix_t *) src;
  matrix_t *matrix_dest = (matrix_t *) dest;

  for (int i = 0; i < matrix_src->rows * matrix_src->cols; i++) {
    memcpy(&matrix_dest->ptr[(matrix_src->cols*rank)+i], &matrix_src->ptr[i], sizeof(int));
  }
}

void
mat_mult(int *c, int *a, int *b, int m, int n, int p, int rows_index)
{
  fprintf(stderr, "rows_index: %d\n", rows_index);
  for (int i = 0;  i < m;  i++) {
    for (int j = 0;  j < p;  j++) {
      for (int k = 0;  k < n;  k++) {
        MAT_ELT(c, p, i, rows_index) += MAT_ELT(a, n, i, k) * MAT_ELT(b, p, k, j);
      }
    }
    rows_index++;
  }
  // fprintf(stderr, "after: %d\n", *(c+cols_index));
}

int
main(int argc, char **argv)
{
  int num_procs;
  int rank;
  parallel_track_t parallel_track;

  char *prog_name = argv[0];
  char *a_file_name = NULL;
  char *b_file_name = NULL;
  char *c_file_name = NULL;


  int ch;
  while ((ch = getopt(argc, argv, "a:b:c:h")) != -1) {
    switch (ch) {
      case 'a':
      a_file_name = optarg;
      break;
      case 'b':
      b_file_name = optarg;
      break;
      case 'c':
      c_file_name = optarg;
      break;
      case 'h':
      default:
      usage(prog_name);
    }
  }

  if (a_file_name == NULL || b_file_name == NULL || c_file_name == NULL ) {
    usage(prog_name);
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start_time;
  MPI_Status status;

  matrix_t a_matrix;
  matrix_t b_matrix;
  matrix_t c_matrix;
  matrix_t a_matrix_master;
  matrix_t b_matrix_master;
  matrix_t c_matrix_master;

  if(rank == BROADCASTER){
    matrix_t *cur_matrix = &a_matrix_master;
    cur_matrix = (matrix_t *) read_file(a_file_name, cur_matrix);
    a_matrix_master = *cur_matrix;

    cur_matrix = &b_matrix_master;
    cur_matrix = (matrix_t *) read_file(b_file_name, cur_matrix);
    b_matrix_master = *cur_matrix;

    c_matrix_master.rows = a_matrix_master.rows;
    c_matrix_master.cols = b_matrix_master.cols;

    start_time = now();

    int rows_per_proc = a_matrix_master.rows / num_procs;
    int cols_per_proc = b_matrix_master.cols / num_procs;

    parallel_track.cols = cols_per_proc;
    parallel_track.cols_index = 0;
    parallel_track.rows = rows_per_proc;
    parallel_track.rows_index = 0;
    parallel_track.num_cols_complete = 0;

    for (int proc = 1; proc < num_procs; proc++) {
      parallel_track_t tmp = parallel_track;
      tmp.cols_index = proc * cols_per_proc;
      tmp.rows_index = proc * rows_per_proc;

      MPI_Send((void *) &a_matrix_master, 2, MPI_INT, proc, 0, MPI_COMM_WORLD);
      MPI_Send((void *) &b_matrix_master, 2, MPI_INT, proc, 0, MPI_COMM_WORLD);
      MPI_Send((void *) &tmp, 6, MPI_INT, proc, 0, MPI_COMM_WORLD);

      MPI_Send((void *) &( MAT_ELT(a_matrix_master.ptr, a_matrix_master.cols, rows_per_proc * proc, 0 )), rows_per_proc * a_matrix_master.cols, MPI_INT, proc, 0, MPI_COMM_WORLD);

      int *tmp_matrix = malloc(sizeof(int) * cols_per_proc * b_matrix_master.rows);
      int tmp_row=0;
      for (int i = 0; i < b_matrix_master.rows; i++) {
        for (int j = proc * cols_per_proc; j < (proc * cols_per_proc)+cols_per_proc; j++) {
          memcpy(tmp_matrix+tmp_row, &MAT_ELT(b_matrix_master.ptr, b_matrix_master.cols, i, j), sizeof(int));
          tmp_row+=1;
        }
      }
      MPI_Send((void *) tmp_matrix, cols_per_proc * b_matrix_master.rows, MPI_INT, proc, 0, MPI_COMM_WORLD);
      free(tmp_matrix);
    }

    a_matrix.ptr = malloc(sizeof(int) * a_matrix_master.cols * parallel_track.rows);
    b_matrix.ptr = malloc(sizeof(int) * b_matrix_master.rows * parallel_track.cols);
    int tmp_row=0;
    for (int i = 0; i < b_matrix_master.rows; i++) {
      for (int j = rank * cols_per_proc; j < (rank * cols_per_proc)+cols_per_proc; j++) {
        memcpy(&b_matrix.ptr[tmp_row], &MAT_ELT(b_matrix_master.ptr, b_matrix_master.cols, i, j), sizeof(int));
        tmp_row+=1;
      }
    }

    memcpy( a_matrix.ptr, &( MAT_ELT(a_matrix_master.ptr, a_matrix_master.cols, rows_per_proc * rank, 0 )), sizeof(int) * rows_per_proc*a_matrix_master.cols);
  }else{
    MPI_Recv((void *) &a_matrix_master, 2, MPI_INT, BROADCASTER, 0, MPI_COMM_WORLD, &status);
    MPI_Recv((void *) &b_matrix_master, 2, MPI_INT, BROADCASTER, 0, MPI_COMM_WORLD, &status);

    MPI_Recv((void *) &parallel_track, 6, MPI_INT, BROADCASTER, 0, MPI_COMM_WORLD, &status);

    a_matrix.ptr = malloc(sizeof(int) * a_matrix_master.cols * parallel_track.rows);
    b_matrix.ptr = malloc(sizeof(int) * b_matrix_master.rows * parallel_track.cols);

    MPI_Recv((void *) a_matrix.ptr, parallel_track.rows * a_matrix_master.cols, MPI_INT, BROADCASTER, 0, MPI_COMM_WORLD, &status);

    MPI_Recv((void *) b_matrix.ptr, parallel_track.cols * b_matrix_master.rows, MPI_INT, BROADCASTER, 0, MPI_COMM_WORLD, &status);

  }
  a_matrix.rows = parallel_track.rows;
  a_matrix.cols = a_matrix_master.cols;

  b_matrix.cols = parallel_track.cols;
  b_matrix.rows = b_matrix_master.rows;

  c_matrix.rows = parallel_track.rows;
  c_matrix.cols = b_matrix_master.cols;
  c_matrix.ptr = malloc(c_matrix.rows * c_matrix.cols * sizeof(int));
  for (int i = 0; i < c_matrix.rows * c_matrix.cols; i++) {
    *(c_matrix.ptr + i) = 0;
  }

  int rank_next = (rank + 1) % num_procs;
  int rank_prev = rank == 0 ? num_procs - 1 : rank - 1;

  // Making the round robin
  MPI_Barrier(MPI_COMM_WORLD);
  while (parallel_track.num_cols_complete < b_matrix_master.cols) {
    check_array(&b_matrix, rank);
    mat_mult( c_matrix.ptr, a_matrix.ptr, b_matrix.ptr, a_matrix.rows, a_matrix.cols, b_matrix.cols, parallel_track.rows_index);

    MPI_Barrier(MPI_COMM_WORLD);
    // fprintf(stderr, "%d sending to %d\n", rank, rank_next);
    MPI_Sendrecv(b_matrix.ptr, parallel_track.cols * b_matrix_master.rows, MPI_INT, rank_next, rank_next,
      b_matrix.ptr, parallel_track.cols * b_matrix_master.rows, MPI_INT, rank_prev, rank,
      MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Sendrecv(&parallel_track.rows_index, 1, MPI_INT, rank_next, rank_next,
      &parallel_track.rows_index, 1, MPI_INT, rank_prev, rank,
      MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);

    parallel_track.num_cols_complete+= parallel_track.cols;
  }

  check_array(&c_matrix, rank);

  // send and write results to the file.
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == BROADCASTER) {
    c_matrix_master.ptr = malloc(sizeof(int) * c_matrix_master.rows * c_matrix_master.cols);
    copy_mat(&c_matrix,&c_matrix_master, rank);

    for (int proc = 1; proc < num_procs; proc++) {
      MPI_Recv(c_matrix.ptr, c_matrix.rows * c_matrix.cols, MPI_INT, proc, proc, MPI_COMM_WORLD, &status);

      copy_mat(&c_matrix,&c_matrix_master, proc);
    }
  }else{
    MPI_Send(c_matrix.ptr, c_matrix.rows * c_matrix.cols, MPI_INT, BROADCASTER, rank, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);


  MPI_Finalize();
  if (rank == 0) {
    printf("    TOOK %5.3f seconds\n\n\n", now() - start_time);
    write_matrix(&c_matrix_master, c_file_name);
    check_array(&c_matrix_master, rank);
  }


  free(a_matrix.ptr);
  free(b_matrix.ptr);
  free(c_matrix.ptr);

  if (rank == BROADCASTER) {
    free(a_matrix_master.ptr);
    free(b_matrix_master.ptr);
    free(c_matrix_master.ptr);
  }
}
