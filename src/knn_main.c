#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define BUFFER_SIZE 1000

struct list_head {
    struct list_head *next, *prev;
};

typedef struct {
    double *values;
    int dims;
} vec_t;

struct neighbor_info {
    struct list_head head;
    vec_t *vec_ptr; // can be used to initialize data_vec
    double dist;
};

struct classification {
    int class;
};

typedef struct {
    vec_t vec;
    int class;
    struct neighbor_info *neighbors;
    struct classification **classifications_ptr;
} data_vec_t;

typedef struct {
    long size;
    data_vec_t **data;
} data_set_t;

typedef struct {
    // necessary information for managing worker threads
    pthread_t *threads;
    int count;
} thread_pool_t;

pthread_mutex_t mutex_task = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_task = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_done_task = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_done_task = PTHREAD_COND_INITIALIZER;

typedef struct {
    void *(*function) (void *);
    void *args;
} Task;

struct args_neighbor {
    data_vec_t *test_vector_ptr;
    data_vec_t *train_vector_ptr;
    int k_max;
};

/* initialize "shortcut links" for empty list */
extern void
list_init(struct list_head *head);

/* insert new entry after the specified head */
extern void
list_add(struct list_head *new, struct list_head *head);

/* insert new entry before the specified head */
extern void
list_add_tail(struct list_head *new, struct list_head *head);

/* deletes entry from list and reinitialize it, returns pointer to entry */
extern struct list_head*
list_del(struct list_head *entry);

/* delete entry from one list and insert after the specified head */
extern void
list_move(struct list_head *entry, struct list_head *head);

/* delete entry from one list and insert before the specified head */
extern void
list_move_tail(struct list_head *entry, struct list_head *head);

/* tests whether a list is empty */
extern int
list_empty(struct list_head *head);

// temp
Task task_queue[100000];
int task_count = 0;
Task done_tasks[100000];
int done_task_count = 0;
int total_tasks = 0;

void execute_task(Task *task_ptr);

//temp
void submitTask(Task *task_ptr) {
    pthread_mutex_lock(&mutex_task);
    task_queue[task_count] = *task_ptr;
    task_count++;
    pthread_mutex_unlock(&mutex_task);
    pthread_cond_signal(&cond_task);
    free(task_ptr);
}

void *start_thread(void* args);

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

void
print_list(struct list_head *head, int k_max);

// error if printf or puts fails
void handleOutputError() {
    exit(1);
}

// print usage if user gives invalid arguments
void printUsage() {
    int ret = printf("USAGE: knn_main <input_file> <N> <k_max> <B> <n_threads>\n");
    if (ret < 0) handleOutputError();
}

void readInputHeader(FILE *file, long *N_max_ptr, int *vec_dim_ptr, int *class_count_ptr);

void readInputData(FILE *file, data_set_t *data_set, int dims);

double euclideanDistance(data_vec_t *test_vec, data_vec_t *train_vec);

void sorted_insert(data_vec_t *test_vec, data_vec_t *train_vec, 
                   double distance, int k_max);

void* insert_neighbor(struct args_neighbor *args) {
    // printf("Got:\n   &args: %p\n   &test_vec: %p\n   test_vec.vec.values[0]: %g\n   &neighbor_vec: %p\n   neighbor_vec.vec.values[0]: %g\n",
    //        args, args->test_vector_ptr, args->test_vector_ptr->vec.values[0],
    //        args->train_vector_ptr, args->train_vector_ptr->vec.values[0]);
    data_vec_t *test_vec = args->test_vector_ptr;
    data_vec_t *neighbor_vec = args->train_vector_ptr;
    int k_max = args->k_max;
    double dist = euclideanDistance(test_vec, neighbor_vec);
    sorted_insert(test_vec, neighbor_vec, dist, k_max);
}

void split_data_set(data_set_t* src, data_set_t* dest, int B);

int classify(data_vec_t *data_vec_ptr, int k, int total_classes);
 
 
void free_neighbors(data_vec_t *data_vec_ptr);

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

    // create worker threads
    thread_pool_t *thread_pool = malloc(sizeof(thread_pool_t));
    thread_pool_init(thread_pool, n_threads);

    // read file contents
    FILE *file;
    file = fopen(fileName, "r");
    long N_max;
    int vec_dim;
    int total_classes;
    readInputHeader(file, &N_max, &vec_dim, &total_classes);
    printf("N_max: %ld, vec_dim: %d, class_count: %d\n", N_max, vec_dim, total_classes);

    if (N_max < N) N = N_max;

    data_set_t data_set;
    data_vec_t **data = malloc(N * sizeof(data_vec_t*));
    data_set.data = data;
    data_set.size = N;

    readInputData(file, &data_set, vec_dim);
    // data_set contains all N vectors
    fclose(file);

    // split dataset into B sub sets
    data_set_t *sub_sets = malloc(B * sizeof(data_set_t));
    split_data_set(&data_set, sub_sets, B);

    // Parallelization
    // thread pool consists of 4 functions
    // 3 computation phases, outside of pool, generalization is important here
    // overlap is allowed here, if one task in phase 1 is complete, tasks in the next phase is allowed to start
  
    // 1. Parallel computation of k_max nearest neighbors
    for (int i = 0; i < B; i++) {
        // distance of each vector of test set to all vectors in all training sets
        data_set_t test_set = sub_sets[i];
        for (long j = 0; j < test_set.size; j++) {
            data_vec_t *data_vec_ptr = test_set.data[j];
            for (int k = 0; k < B; k++) {
                if (i == k) continue;
                data_set_t training_set = sub_sets[k];
                for (long l = 0; l < training_set.size; l++) {
                    data_vec_t *training_data_vec_ptr = training_set.data[l];
                    //insert_neighbor(data_vec, training_data_vec, k_max);
                    struct args_neighbor *args_ptr = malloc(sizeof(struct args_neighbor));
                    args_ptr->test_vector_ptr = data_vec_ptr;
                    args_ptr->train_vector_ptr = training_data_vec_ptr;
                    args_ptr->k_max = k_max;
                    // printf("Expected:\n   &args: %p\n   &test_vec: %p\n   test_vec.vec.values[0]: %g\n   &neighbor_vec: %p\n   neighbor_vec.vec.values[0]: %g\n",
                    //        args_ptr, data_vec_ptr, data_vec_ptr->vec.values[0],
                    //        training_data_vec_ptr, training_data_vec_ptr->vec.values[0]);
                    thread_pool_enqueue(thread_pool, &insert_neighbor, args_ptr);
                }

            }
        }
    }
    for (int i = 0; i < total_tasks; i++) {
        Task *task_ptr = thread_pool_wait(thread_pool);
        free(task_ptr);
    }
    // 2. Parallel classification and scoring
    // think about efficiency for this phase
    // for set in B testsets
    for (int i = 0; i < B; i++) {
        data_set_t test_set = sub_sets[i];

        // classification of all test vectors in set
        for (long j = 0; j < test_set.size; j++) {
            data_vec_t *test_vec_ptr = test_set.data[j];
            test_vec_ptr->classifications_ptr = malloc(k_max * sizeof(struct classification*));
            // for each k
            for (int k = 1; k <= k_max; k++) {
                struct classification *classification_ptr = malloc(sizeof(struct classification));
                // which class is most common among k neighbors
                    // on par: highest index wins
                int class = classify(test_vec_ptr, k, total_classes);
                    // store the results in a fitting data-structure 
                classification_ptr->class = class;
                test_vec_ptr->classifications_ptr[k-1] = classification_ptr;
            }
            
        }
    }
    // 3. evaluation of a classification quality
    int k_opt = 0;
    double best_classification = 0.0;
    for (int k = 0; k < k_max; k++) {
        long correct_qualification_counter = 0;
        for (long i = 0; i < N; i++) {
            data_vec_t *data_vec_ptr = data_set.data[i];
            int correct_class = data_vec_ptr->class;
            struct classification *classification = data_vec_ptr->classifications_ptr[k];
            // compare classification of knn with original class
            if (classification->class == correct_class) {
                correct_qualification_counter++;
            }
        }
        // calculate ratio: correct / all
            // also in parallel
        double classification_quality = (double) correct_qualification_counter / (double) N;
        printf("%d %g\n", k, classification_quality);
        if (classification_quality >= best_classification) {
            best_classification = classification_quality;
            k_opt = k;
        }
    }
    printf("%d\n", k_opt);

    // results of these 3 phases can be stored in a single data structure, containing the data vectors and their classes,
    // as well as information about the k_max nearest neighbors and the classification.

    for (int i = 0; i < N; i++) {
        data_vec_t *data_vec = data_set.data[i];
        for (int j = 0; j < k_max; j++) {
            free(data_vec->classifications_ptr[j]);
        }
        free(data_vec->classifications_ptr);
        free(data_vec->vec.values);
        free_neighbors(data_vec);
        free(data_vec->neighbors);
        free(data_vec);
    }
    free(data_set.data);
    for (int i = 0; i < B; i++) {
        free(sub_sets[i].data);
    }
    free(sub_sets);
    thread_pool_shutdown(thread_pool);
    return(0);
}

/* initialize "shortcut links" for empty list */
void
list_init(struct list_head *head)
{
    head->next = head;
    head->prev = head;
}

/* insert new entry after the specified head */
void
list_add(struct list_head *new, struct list_head *head)
{
    new->next = head->next;
    new->prev = head;
    head->next->prev = new;
    head->next = new;
}

/* insert new entry before the specified head */
void
list_add_tail(struct list_head *new, struct list_head *head)
{
    head->prev->next = new;
    new->prev = head->prev;
    head->prev = new;
    new->next = head;
}

/* deletes entry from list and reinitialize it, returns pointer to entry */
struct list_head*
list_del(struct list_head *entry)
{
    entry->prev->next = entry->next;
    entry->next->prev = entry->prev;
    list_init(entry);
    return entry;
}

/* delete entry from one list and insert after the specified head */
void
list_move(struct list_head *entry, struct list_head *head)
{
    list_del(entry);
    list_add(entry, head);
}

/* delete entry from one list and insert before the specified head */
void
list_move_tail(struct list_head *entry, struct list_head *head)
{
    list_del(entry);
    list_add_tail(entry, head);
}

/* tests whether a list is empty */
int
list_empty(struct list_head *head)
{
    if (head->next == head && head->prev == head) return 1;
    return 0;
}

void
print_list(struct list_head *head, int k_max)
{
    int empty = list_empty(head);
    if (empty == 1) {
        printf("Liste ist leer.\n");
    } else {
        printf("Liste:\n");
        for (int i = 0; i < k_max; i++) {
            struct neighbor_info *next_neighbor = (struct neighbor_info *) head->next;
            data_vec_t *neighbor_data = (data_vec_t *) next_neighbor->vec_ptr;
            printf("val: %g, class: %d, distance: %g\n", neighbor_data->vec.values[0], neighbor_data->class, next_neighbor->dist);
            head = head->next;
        }
        printf("---\n");
    }
}

void *start_thread(void* args) {
    while (1) {
        pthread_mutex_lock(&mutex_task);
        while (task_count == 0) {
            pthread_cond_wait(&cond_task, &mutex_task);
        }
        // // dequeue and execute task
        // Task task = dequeue_task();
        // printf("dequeued task: func: %p, args: %p\n", task.function, task.args);
        Task task = task_queue[0];
        int i;
        for (i = 0; i < task_count - 1; i++) {
            task_queue[i] = task_queue[i + 1];
        }
        task_count--;
        pthread_mutex_unlock(&mutex_task);
        execute_task(&task);
        free(task.args);
    }
}

void execute_task(Task *task_ptr) {
    task_ptr->function(task_ptr->args);
    pthread_mutex_lock(&mutex_done_task);
    done_tasks[done_task_count] = *task_ptr;
    done_task_count++;
    pthread_mutex_unlock(&mutex_done_task);
    pthread_cond_signal(&cond_done_task);
}

// create thread pool and start thread count worker threads
void thread_pool_init(thread_pool_t* thread_pool, int thread_count) {
    thread_pool->threads = malloc(thread_count * sizeof(pthread_t));
    thread_pool->count = thread_count;
    for (int i = 0; i < thread_count; i++) {
        if (pthread_create(&thread_pool->threads[i], NULL, &start_thread, NULL) != 0) {
            fprintf(stderr, "Failed to create thread %d", i);
        }
    }
}

// pass pointer to function with args pointer, enqueue it in task list and signalize worker thread
// args might contain a set of indices to data vectors to calculate
// data type of arg can change depending on computation phase
void thread_pool_enqueue(thread_pool_t* thread_pool, void *(*function) (void *), void* args) {
    total_tasks++;
    Task *task_ptr = malloc(sizeof(Task));
    task_ptr->function = function;
    task_ptr->args = args;    
    submitTask(task_ptr);
}

Task* thread_pool_wait(thread_pool_t* thread_pool) {
    pthread_mutex_lock(&mutex_done_task);
    while (done_task_count == 0) {
        // cond wait
        pthread_cond_wait(&cond_done_task, &mutex_done_task);
    } 
    // this fucks up after moving
    Task task = done_tasks[0];
    Task *task_ptr = malloc(sizeof(Task));
    *task_ptr = task;
    for (int i = 0; i < done_task_count - 1; i++) {
        done_tasks[i] = done_tasks[i + 1];
    }
    done_task_count--;
    pthread_mutex_unlock(&mutex_done_task);
    return task_ptr;
}

// End all worker threads and free all allocated memory
// called on end of program
void thread_pool_shutdown(thread_pool_t* thread_pool) {
    for (int i = 0; i < thread_pool->count; i++) {
        pthread_cancel(thread_pool->threads[i]);
    }
    free(thread_pool->threads);
    free(thread_pool);
}

void readInputHeader(FILE *file, long *N_max_ptr, int *vec_dim_ptr, int *class_count_ptr) {
    int header_arg_count = 3;
    long headerArguments[header_arg_count ];
    for (int i = 0; i < header_arg_count ; i++) {
        int n = fscanf(file, "%ld ", &headerArguments[i]);
        if (n != 1) {
            fprintf(stderr, "scanf error\n");
            exit(1);
        }
    }
    *N_max_ptr = headerArguments[0];
    *vec_dim_ptr = (int) headerArguments[1];
    *class_count_ptr = (int) headerArguments[2];
}

void readInputData(FILE *file, data_set_t *data_set, int dims) {
    char buffer[BUFFER_SIZE];
    long N = data_set->size;
    for (long i = 0; i < N; i++) {
        data_vec_t *data_vec_ptr = malloc(sizeof(data_vec_t));
        data_vec_ptr->vec.dims = dims;

        // initialize the neighbor list
        struct neighbor_info *neighbors_ptr = malloc(sizeof(struct neighbor_info));
        neighbors_ptr->dist = 0;
        neighbors_ptr->vec_ptr = &data_vec_ptr->vec;
        list_init(&neighbors_ptr->head);
        data_vec_ptr->neighbors = neighbors_ptr;
         
        double *values = malloc(dims * sizeof(double));
        data_vec_ptr->vec.values = values;
         
        // read line into buffer
        char *s = fgets(buffer, BUFFER_SIZE, file);
        if (s == NULL) {
            fprintf(stderr, "fgets error\n");
            return;
        }

        // parse buffer
        char *token = strtok(buffer, " ");
        data_vec_ptr->vec.values[0] = strtod(token, NULL);
        for (int j = 1; j < dims; j++) {
            token = strtok(NULL, " ");
            data_vec_ptr->vec.values[j] = strtod(token, NULL);
        }
        token = strtok(NULL, " ");
        data_vec_ptr->class = (int) strtol(token, NULL, 10);
         
        data_set->data[i] = data_vec_ptr;
    }
}

void split_data_set(data_set_t* src, data_set_t* dest, int B) {
    long N = src->size;
    ldiv_t div = ldiv(N, B);
    long B_offsets = div.rem;
    long vectors_per_subset = div.quot;
    long data_start_index = 0;
    for (int i = 0; i < B; i++) {
        data_set_t sub_set;

        // calculate sub set size
        long size = vectors_per_subset + (i < B_offsets);
        sub_set.size = size;

        // copy data vectors from data set into sub set
        data_vec_t **sub_set_data = malloc(size * sizeof(data_vec_t*));
        for (long j = 0; j < size; j++) {
            sub_set_data[j] = src->data[j + data_start_index];
        }
        sub_set.data = sub_set_data;
         
        dest[i] = sub_set;

        // start each sub-set at the correct index in the data-set
        data_start_index += size;
    }
}

double euclideanDistance(data_vec_t *test_vec_ptr, data_vec_t *train_vec_ptr) {
    double dist = 0;
    int dims = test_vec_ptr->vec.dims;
    for (int m = 0; m < dims; m++) {
        dist += pow(test_vec_ptr->vec.values[m] - train_vec_ptr->vec.values[m], 2);
    }
    return dist;
}

void sorted_insert(data_vec_t *test_vec, data_vec_t *train_vec, 
                   double distance, int k_max) {
    struct list_head *anchor = &test_vec->neighbors->head;
    struct list_head *current = anchor;
    for (int i = 0; i < k_max; i++) {
        struct neighbor_info *next = (struct neighbor_info *) current->next;
        if (distance <= next->dist || current->next == anchor) {
            struct neighbor_info *new = malloc(sizeof (struct neighbor_info));
            new->dist = distance;
            new->vec_ptr = &train_vec->vec;
            if (distance < next->dist) list_add_tail(&new->head, current->next);
            else list_add(&new->head, current->next);
            return;
        } else {
            current = current->next;
        }
    }
}

int classify(data_vec_t *data_vec_ptr, int k, int total_classes) {
    int class_count[total_classes];
    for (int i = 0; i < total_classes; i++) {
        class_count[i] = 0;
    }
    struct list_head *anchor = &data_vec_ptr->neighbors->head;
    struct list_head *current = anchor;
    for (int i = 0; i < k; i++) {
        struct neighbor_info *next = (struct neighbor_info *)current->next;
        data_vec_t *neighbor_vec_ptr = (data_vec_t *) next->vec_ptr;
        class_count[neighbor_vec_ptr->class]++;
        current = current->next;
    }
    int max_count = 0;
    int winner_class = 0;
    for (int i = 0; i < total_classes; i++) {
        if (class_count[i] > max_count) {
            max_count = class_count[i];
            winner_class = i;
        } else {
            if (class_count[i] == max_count && i > winner_class) {
                winner_class = i;
            }
        }
    }
    return winner_class;
}

void free_neighbors(data_vec_t *data_vec_ptr) {
    struct list_head *anchor = &data_vec_ptr->neighbors->head;
    struct list_head *current = anchor->next;
    do {
        struct list_head *next = current->next;
        free(list_del(current));
        current = next; 
    } while (current != anchor);
}
 
int predict_sample(double *values, long N, int k, char *file_name)
{
    // read file contents
    FILE *file;
    file = fopen(file_name, "r");
    long N_max;
    int vec_dim;
    int total_classes;
    readInputHeader(file, &N_max, &vec_dim, &total_classes);
    printf("N_max: %ld, vec_dim: %d, class_count: %d\n", 
           N_max, vec_dim, total_classes);
    if (N_max < N) N = N_max;

    data_set_t data_set;
    data_vec_t **data = malloc(N * sizeof(data_vec_t*));
    data_set.data = data;
    data_set.size = N;

    readInputData(file, &data_set, vec_dim);
    // data_set contains all N vectors
    fclose(file);
    
    // create sample vector
    data_vec_t *sample_data_vec_ptr = malloc(sizeof(data_vec_t));
    sample_data_vec_ptr->vec.dims = vec_dim;

    // initialize the neighbor list
    struct neighbor_info *neighbors_ptr = malloc(sizeof(struct neighbor_info));
    neighbors_ptr->dist = 0;
    neighbors_ptr->vec_ptr = &sample_data_vec_ptr->vec;
    list_init(&neighbors_ptr->head);
    sample_data_vec_ptr->neighbors = neighbors_ptr;
     
    sample_data_vec_ptr->vec.values = values;
    
    // fill neighbor list
    for (long i = 0; i < N; i++) {
        data_vec_t *training_data_vec = data_set.data[i];
        double dist = euclideanDistance(sample_data_vec_ptr, training_data_vec);
        sorted_insert(sample_data_vec_ptr, training_data_vec, dist, k);
    }
    int prediction = classify(sample_data_vec_ptr, k, total_classes);
    return prediction;
}
