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
void
swap(int *v, int i, int j)
{
  int t = v[i];
  v[i] = v[j];
  v[j] = t;
}

/* Helper function to compute permutations recursively. */
void
perms(int *v, int n, int i, perm_action_t action)
{
  int j;

  if (i == n) {
	/* At the end of the array, we have a permutation we can use. */
	action(v, n);

  } else {
	/* Recursively explore the permutations from index i to (n - 1). */
	for (j = i;  j < n;  j++) {
	  /* Array with i and j switched */
	  swap(v, i, j);
	  perms(v, n, i + 1, action);
	  /* Swap back to the way they were */
	  swap(v, i, j);
	}
  }
}

/* Generate permutations of the elements i to (n - 1). */
// void
// permutations(int *v, int n, perm_action_t action)
// {
//   perms(v, n, 0, action);
// }

/* Trivial action to pass to permutations--print out each one. */
void
print_perm(int *perm, int n, char *msge)
{
  for (int j = 0;  j < n;  j++) {
	printf("%2d ", perm[j]);
  }
  printf(" - %s\n", msge);
}

/* No-op action */
void
nop(int *v, int n)
{
  return;
}

int num_cities = 5;
int shortest_length = INT_MAX;
int num_as_short = -1;
int num_trials = 0;
int num_threads = 1;
int random_seed = 42;

/* Create an instance of a symmetric TSP. */
int *
create_tsp(int n)
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
void
print_tsp(int *tsp, int n)
{
  printf("TSP (%d cities - seed %d)\n    ", n, random_seed);
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
void
eval_tsp(int *perm, int n)
{
  /* Initialize the distances array once per program run. */
  static int *distances = NULL;
  if (NULL == distances) {
	distances = create_tsp(num_cities);
	print_tsp(distances, num_cities);
  }

  /* Calculate the length of the tour for the current permutation. */
  int total = 0;
  for (int i = 0;  i < n;  i++) {
	int j = (i + 1) % n;
	int from = perm[i];
	int to = perm[j];
	int val = TSP_ELT(distances, n, from, to);
	debug_printf("tsp[%d, %d] = %d\n", from, to, val);
	total += val;
  }

#if DEBUG
  print_perm(perm, n, "PERM");
#endif

  /* Gather statistics. */
  if (total <= shortest_length) {
	char buf[80];
	sprintf(buf, "len %4d - trial %12d", total, num_trials);
	print_perm(perm, n, buf);

	if (total == shortest_length) {
	  num_as_short++;
	} else {
	  num_as_short = 1;
	}
	shortest_length = total;
  }
  num_trials++;
  debug_printf("Total %d\n", total);
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
__device__ long
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

#if DEBUG
  printf("k=%d, size=%d, remain=%ld\n", k, size, remain);
  printf("  perm");
  list_dump(perm);
  printf("  nums");
  list_dump(numbers);
#endif

  for (int i = 1;  i < size;  i++) {
    long f = factorial(size - i);
    long j = remain / f;
    remain = remain % f;
#if DEBUG
	printf("i=%d, f=%ld j=%ld, remain=%ld\n", i, f, j, remain);
#endif

	list_add(perm, list_get(numbers, j));
	list_remove_at(numbers, j);

#if DEBUG
	printf("  perm");
	list_dump(perm);
	printf("  nums");
	list_dump(numbers);
#endif
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

/* Print a permutation array */
// void
// print_perm(int *perm, int size)
// {
//   for (int k = 0; k < size; k++) {
// 	printf("%4d", perm[k]);
//   }
//   printf("\n");
// }

/* Given an array of size elements at perm, update the array in place to
   contain the lexographically next permutation. It is originally due to
   Dijkstra. The present version is discussed at:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
 */
void
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
void tsp_go(int* perm,int num_cities, int num_threads){

  // int perm[num_cities];
  perm = kth_perm((factorial(num_cities)/num_threads)*threadIdx.x, num_cities);

  char snum[5];
  snprintf(snum, 5, "%d",threadIdx.x);

  print_perm(perm, 1, snum);
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
  /* Use "random" random seed by default. */
  random_seed = time(NULL);

  int ch;
  while ((ch = getopt(argc, argv, "c:hs:n::")) != -1) {
	switch (ch) {
	case 'c':
	  num_cities = atoi(optarg);
	  break;
	case 's':
	  random_seed = atoi(optarg);
	  break;
  case 'n':

	  break;
	case 'h':
	default:
	  usage(argv[0]);
	}
  }

  // cost array
  int* h_cperm = create_tsp(num_cities);
  int* d_cperm;

  // output Array
  int h_output[num_threads];
  int* d_output;

  // malloc, make space
  // int *d_perm;
  // cudaMalloc((void **)&d_perm, sizeof(int)*num_cities);
  cudaMalloc((void **)&d_cperm, sizeof(int)*num_cities);
  cudaMalloc((void **)&d_output, sizeof(int)*num_threads);
  // copy if nessary
  cudaMemcpy(d_cperm, h_cperm, num_cities * sizeof(int), cudaMemcpyHostToDevice);



  /* "Travel, salesman!" */
  tsp_go<<<1,num_threads>>>(d_cperm,num_cities, num_threads);

  // collect results
  cudaMemcpy(h_output, d_output, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

  /* Report. */
  printf("\n");
  printf("Trials %d\n", num_trials);
  float percent_as_short = (float)num_as_short / (float)num_trials * 100.0;
  printf("Shortest %d - %d tours - %.6f%%\n",
		 shortest_length, num_as_short, percent_as_short);
  printf("\n");

  free(h_cperm);
  cudaFree(d_cperm);
  cudaFree(d_output);
}
