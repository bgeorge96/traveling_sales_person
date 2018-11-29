#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* Original permuation code due to D. Jimenez, UT Austin
 * http://faculty.cse.tamu.edu/djimenez/ut/utsa/cs3343/
 */

/* Requires C99 compiler (gcc: -std=c99) */
#define DEBUG 0
#define debug_printf(fmt, ...) do { if (DEBUG) fprintf(stderr, fmt, __VA_ARGS__); } while (0)

/* Action function for each permuation. */
typedef void (*perm_action_t)(int *v, int n);

/* Reference an element in the TSP distance array. */
#define TSP_ELT(tsp, n, i, j) *(tsp + (i * n) + j)

/* Swap array elements. */
__device__ __host__ void
swap(int *v, int i, int j)
{
  int t = v[i];
  v[i] = v[j];
  v[j] = t;
}

/* Trivial action to pass to permutations--print out each one. */
__device__ void
print_perm(int *perm, int n, char *msge)
{
  for (int j = 0;  j < n;  j++) {
	   printf("%2d ", perm[j]);
  }
  printf(" - %s\n", msge);
}

__host__ void
smallest_in_list(int *list, int *num_short, int n, int *shortest_length, int *num_as_short)
{
  int min_path = INT_MAX;
  int num = 0;
  for (int j = 0;  j < n;  j++) {
    int tmp = list[j];
    if(tmp < min_path){
      min_path = tmp;
      num = 0;
    }
    if(tmp == min_path){
      num += num_short[j];
    }
  }
  *shortest_length = min_path;
  *num_as_short = num;
}

/* Create an instance of a symmetric TSP. */
__host__ int *
create_tsp(int n, int random_seed)
{
  int *tsp = (int *) malloc(n * n * sizeof(int));

  srandom(random_seed);
  for (int i = 0;  i < n;  i++) {
	for (int j = 0;  j <= i;  j++) {
	  int val = (int)(random() / (RAND_MAX / 100));
	  TSP_ELT(tsp, n, i, j) = val;
	  TSP_ELT(tsp, n, j, i) = val;
	}
  }
  return tsp;
}

/* Print a TSP distance matrix. */
__host__ __device__ void
print_tsp(int *tsp, int n)
{
  // printf("TSP (%d cities - seed %d)\n    ", n, random_seed);
  for (int j = 0;  j < n;  j++) {
	printf("%3d|", j);
  }
  printf("\n");
  for (int i = 0;  i < n;  i++) {
	printf("%2d|", i);
	for (int j = 0;  j < n;  j++) {
	  printf("%4d", TSP_ELT(tsp, n, i, j));
	}
	printf("\n");
  }
  printf("\n");
}

/* Evaluate a single instance of the TSP. */
__device__ int
eval_tsp(int *perm, int n, int* distances)
{
  /* Calculate the length of the tour for the current permutation. */
  int total = 0;
  for (int i = 0;  i < n;  i++) {
  	int j = (i + 1) % n;
  	int from = perm[i];
  	int to = perm[j];
  	int val = TSP_ELT(distances, n, from, to);
  	total += val;
  }
  return total;
}

/**** List ADT ****************/

typedef struct {
  int *values;					/* Values stored in list */
  int max_size;					/* Maximum size allocated */
  int cur_size;					/* Size currently in use */
} list_t;

/* Dump list, including sizes */
__device__ void
list_dump(list_t *list)
{
  printf("%2d/%2d", list->cur_size, list->max_size);
  for (int i = 0;  i < list->cur_size;  i++) {
	printf(" %d", list->values[i]);
  }
  printf("\n");
}

/* Allocate list that can store up to 'max_size' elements */
__device__ list_t *
list_alloc(int max_size)
{
  list_t *list = (list_t *)malloc(sizeof(list_t));
  list->values = (int *)malloc(max_size * sizeof(int));
  list->max_size = max_size;
  list->cur_size = 0;
  return list;
}

/* Free a list; call this to avoid leaking memory! */
__device__ void
list_free(list_t *list)
{
  free(list->values);
  free(list);
}

/* Add a value to the end of the list */
__device__ void
list_add(list_t *list, int value)
{
  if (list->cur_size >= list->max_size) {
	printf("List full");
	list_dump(list);
	// exit(1);
  }
  list->values[list->cur_size++] = value;
}

/* Return the current size of the list */
__device__ int
list_size(list_t *list)
{
  return list->cur_size;
}

/* Validate index */
__device__ void
_list_check_index(list_t *list, int index)
{
  if (index < 0 || index > list->cur_size - 1) {
	printf("Invalid index %d\n", index);
	list_dump(list);
	// exit(1);
  }
}

/* Get the value at given index */
__device__ int
list_get(list_t *list, int index)
{
  _list_check_index(list, index);
  return list->values[index];
}

/* Remove the value at the given index */
__device__ void
list_remove_at(list_t *list, int index)
{
  _list_check_index(list, index);
  for (int i = index; i < list->cur_size - 1;  i++) {
	list->values[i] = list->values[i + 1];
  }
  list->cur_size--;
}

/* Retrieve a copy of the values as a simple array of integers. The returned
   array is allocated dynamically; the caller must free the space when no
   longer needed.
 */
__device__ int *
list_as_array(list_t *list)
{
  int *rtn = (int *)malloc(list->max_size * sizeof(int));
  for (int i = 0;  i < list->max_size;  i++) {
	rtn[i] = list_get(list, i);
  }
  return rtn;
}

/**** Permutation ****************/

/* Permutation algorithms based on code found at:
   http://www.mathblog.dk/project-euler-24-millionth-lexicographic-permutation/
   which references:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
*/

/* Calculate n! iteratively */
__device__ __host__ long
factorial(int n)
{
  if (n < 1) {
	return 0;
  }

  long rtn = 1;
  for (int i = 1;  i <= n;  i++) {
	rtn *= i;
  }
  return rtn;
}

/* Return the kth lexographically ordered permuation of an array of k integers
   in the range [0 .. size - 1]. The integers are allocated dynamically and
   should be free'd by the caller when no longer needed.
*/
__device__ int *
kth_perm(int k, int size)
{
  long remain = k - 1;

  list_t *numbers = list_alloc(size);
  for (int i = 0;  i < size;  i++) {
    list_add(numbers, i);
  }

  list_t *perm = list_alloc(size);


  for (int i = 1;  i < size;  i++) {
    long f = factorial(size - i);
    long j = remain / f;
    remain = remain % f;

    list_add(perm, list_get(numbers, j));
    list_remove_at(numbers, j);

    if (remain == 0) {
      break;
    }
  }

  /* Append remaining digits */
  for (int i = 0;  i < list_size(numbers);  i++) {
    list_add(perm, list_get(numbers, i));
  }

  int *rtn = list_as_array(perm);
  list_free(perm);

  return rtn;
}

/* Given an array of size elements at perm, update the array in place to
   contain the lexographically next permutation. It is originally due to
   Dijkstra. The present version is discussed at:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
 */
__device__ void
next_perm(int *perm, int size)
{
  int i = size - 1;
  while (perm[i - 1] >= perm[i]) {
	i = i - 1;
  }

  int j = size;
  while (perm[j - 1] <= perm[i - 1]) {
	j = j - 1;
  }

  swap(perm, i - 1, j - 1);

  i++;
  j = size;
  while (i < j) {
	swap(perm, i - 1, j - 1);
	i++;
	j--;
  }
}

__global__
void tsp_go(int* perm, int num_cities, int num_threads, int* cperm, int* output, int* num_as_short){
  long one_index = threadIdx.x + 1;
  long cur_idx = (factorial(num_cities)/num_threads) * (threadIdx.x)+1;
  long end_idx = (factorial(num_cities)/num_threads) * (one_index);
  int min_path = INT_MAX;
  int num = 0;

  __syncthreads();

  perm = kth_perm(cur_idx, num_cities);
  while( cur_idx <= end_idx){
    // printf("Hello from %d, end_idx: %ld, cur_idx: %ld, perms: %ld\n", threadIdx.x, end_idx, cur_idx, factorial(num_cities));
    int tmp = eval_tsp(perm, num_cities, cperm);
    // printf("Hello from %d, cost: %d\n", threadIdx.x, tmp);
    if(tmp < min_path){
      min_path = tmp;
      num = 0;
    }

    if(tmp == min_path){
      num++;
    }

    cur_idx++;
    // MAKING SURE NOT OUT OF RANGE
    if( cur_idx <= end_idx){
      next_perm(perm, num_cities);
    }
    // __syncthreads();
  }
  __syncthreads();
  output[threadIdx.x] = min_path;
  num_as_short[threadIdx.x] = num;
}

void
usage(char *prog_name)
{
  fprintf(stderr, "usage: %s [flags]\n", prog_name);
  fprintf(stderr, "   -h\n");
  fprintf(stderr, "   -c <number of cities>\n");
  fprintf(stderr, "   -s <random seed>\n");
  exit(1);
}

int
main(int argc, char **argv)
{
  int num_cities = 5;
  int shortest_length;
  int num_as_short = -1;
  long num_trials = 0;
  int num_threads = 1;
  int random_seed = 42;

  /* Use "random" random seed by default. */
  random_seed = time(NULL);

  int ch;
  while ((ch = getopt(argc, argv, "c:hn:s:")) != -1) {
	switch (ch) {
	case 'c':
	  num_cities = atoi(optarg);
	  break;
  case 'n':
    num_threads = atoi(optarg);
    if(num_threads < 1){
      usage(argv[0]);
    }
    break;
	case 's':
	  random_seed = atoi(optarg);
	  break;
	case 'h':
	default:
	  usage(argv[0]);
	}
  }

  num_trials = factorial(num_cities);
  if(num_trials < num_threads){
    num_threads = num_trials;
  }

  // cost array
  int* h_cperm = create_tsp(num_cities, random_seed);
  // print_tsp(h_cperm, num_cities);
  int* d_cperm;

  // output Array
  int h_output[num_threads];
  int* d_output;

  // perm array
  int *d_perm;

  // num_as_short array
  int h_num_short[num_threads];
  int *d_num_short;

  cudaMalloc((void **)&d_perm, sizeof(int)*num_cities);
  cudaMalloc((void **)&d_cperm, sizeof(int)*num_cities*num_cities);
  cudaMalloc((void **)&d_output, sizeof(int)*num_threads);
  cudaMalloc((void **)&d_num_short, sizeof(int)*num_threads);
  cudaMemcpy(d_cperm, h_cperm, sizeof(int)*num_cities*num_cities, cudaMemcpyHostToDevice);

  /* "Travel, salesman!" */
  tsp_go<<<1, num_threads>>>(d_perm, num_cities, num_threads, d_cperm, d_output, d_num_short);
  cudaDeviceSynchronize();

  // collect results
  cudaMemcpy(h_output, d_output, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_num_short, d_num_short, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

  smallest_in_list(h_output, h_num_short, num_threads, &shortest_length, &num_as_short);

  /* Report. */
  printf("\n");
  printf("Trials %ld\n", num_trials);
  float percent_as_short = (float)num_as_short / (float)num_trials * 100.0;
  printf("Shortest %d - %d tours - %.6f%%\n",
		 shortest_length, num_as_short, percent_as_short);
  printf("\n");

  free(h_cperm);
  // free(h_output);
  cudaFree(d_perm);
  cudaFree(d_cperm);
  cudaFree(d_output);
}
