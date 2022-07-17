#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#define BUFFER_SIZE 1000

typedef struct {
    // necessary information for managing worker threads
} thread_pool_t;

typedef struct {
    // tba
} Task;

// create thread pool and start thread count worker threads
void thread_pool_init(thread_pool_t* thread_pool, int thread_count);

// pass pointer to function with args pointer, enqueue it in task list and signalize worker thread
// args might contain a set of indices to data vectors to calculate
// data type of arg can change depending on computation phase
void thread_pool_enqueue(thread_pool_t* thread_pool, void *(*function) (void *), void* args);

// blocking wait, until one of the threads completed a task.
// returns the passed function and args of the enqueue call. args can contain resulting values
Task* thread_pool_wait(thread_pool_t* thread_pool);

// End all worker threads and free all allocated memory
// called on end of program
void thread_pool_shutdown(thread_pool_t* thread_pool);

// error if printf or puts fails
void handleOutputError() {
    exit(1);
}

// print usage if user gives invalid arguments
void printUsage() {
    int ret = printf("USAGE: knn_main <input_file> <N> <k_max> <B> <n_threads>\n");
    if (ret < 0) handleOutputError();
}


void readLines(FILE *file) {
    char buffer[BUFFER_SIZE];
    while(fgets(buffer, BUFFER_SIZE, file) != NULL) {
        int line_length = strlen(buffer);
        if (buffer[line_length - 1] == '\n') {
            buffer[line_length - 1] = '\0';
        }
        puts(buffer);
    }
}

void readInputHeader(FILE *file, long *N_max_ptr, int *vec_dim_ptr, int *class_count_ptr) {
    int header_arg_count = 3;
    long headerArguments[header_arg_count ];
    for (int i = 0; i < header_arg_count ; ++i) {
        fscanf(file, "%ld ", &headerArguments[i]);
    }
    *N_max_ptr = headerArguments[0];
    *vec_dim_ptr = (int) headerArguments[1];
    *class_count_ptr = (int) headerArguments[2];
}


int main(int argc, char** argv) {
    if (argc != 6) {
        printUsage();
        return 1;
    }
    char *fileName = argv[1];
    long N = strtol(argv[2], NULL, 10);
    int k_max = (int) strtol(argv[3], NULL, 10);
    int B = (int) strtol(argv[4], NULL, 10);
    int n_threads = (int) strtol(argv[5], NULL, 10);
    printf("fileName: %s, N: %ld, k_max: %d, B: %d, n_threads: %d\n", fileName, N, k_max, B, n_threads);

    // read file contents
    FILE *file = malloc(sizeof (FILE*));
    file = fopen(fileName, "r");
    long N_max;
    int vec_dim;
    int class_count;
    readInputHeader(file, &N_max, &vec_dim, &class_count);
    printf("N_max: %ld, vec_dim: %d, class_count: %d\n", N_max, vec_dim, class_count);

    // split dataset into B sub sets
        // same size if N mod B == 0, otherwise close to same size
    // loop over k from 1 to k_max
    for (int k = 0; k < k_max; ++k) {
        // Rotate through sub sets, setting i as test set and the other B-1 sets as training set
        // calculate which k neighbors are closest using squared euclidean distance
            // use thread pool for this
        // assign most common class among k neighbors to vector
        // calculate classification quality: correct qualifications / all classifications
        double class_qual = 0.9751;
        // print "%d %g\n", k, class_qual
        printf("%d %g\n", k, class_qual);
    }
    // print k with optimal class_qual (no parallelization)
    int k_opt = 8;
    printf("%d\n", k_opt);

    // Parallelization
    // n_threads number of worker threads
    // created at the beginning of the program
    // thread pool consists of 4 functions
    // 3 computation phases, outside of pool, generalization is important here
    // overlap is allowed here, if one task in phase 1 is complete, tasks in the next phase is allowed to start
        // 1. Parallel computation of k_max nearest neighbors
            // for all test sets
                // distance of each vector of test set to all vectors in all training sets
                // continuously build list, sorted by ascending distance, of k_max nearest neighbors for each vector in dataset
                // this list can be used for all k's later
        // 2. Parallel classification and scoring
            // for set in B testsets
                // classification of all test vectors in set
                // for each k
                    // which class is most common among k neighbors
                    // on par: highest index wins
                    // store the results in a fitting data-structure (?)
            // think about efficiency for this phase
        // 3. evaluation of a classification quality
            // for each k
                // compare classification of knn with original class
                // calculate ratio: correct / all
                // also in parallel
        // results of these 3 phases can be stored in a single data structure, containing the data vectors and their classes,
        // as well as information about the k_max nearest neighbors and the classification.

    fclose(file);
    return(0);
}