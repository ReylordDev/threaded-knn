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

// maybe make this pointers (swap with below)
struct classification {
    int class;
};

typedef struct {
    vec_t vec;
    int class;
    int completed_phases;
    struct neighbor_info *neighbors;
    struct classification **classifications_ptr;
} data_vec_t;

typedef struct {
    long size;
    data_vec_t **data;
} data_set_t;

typedef struct {
    pthread_t *threads;
    int count;
} thread_pool_t;

pthread_mutex_t mutex_open_task = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_open_task = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_done_task = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_done_task = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_phase_task = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_correct_classifications = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    void *(*function) (void *);
    void *args;
} Task;

struct args_neighbor {
    int phase;
    data_vec_t *test_vec_ptr;
    data_set_t **sub_sets_ptr;
    int k_max;
    int test_set_idx;
    int B;
};

struct args_classify {
    int phase;
    data_vec_t *test_vec_ptr;
    int k_max;
    int total_classes;
};

struct args_score {
    int phase;
    data_vec_t *test_vec_ptr;
    int k_max;
    int **correct_classifications_k_ptr;
};

struct args_quality {
    int k;
    long correct;
    long total;
    double **result_ptr;
};

struct task_queue {
    struct list_head head;
    Task *task_ptr;
} open_tasks, done_tasks;

void init_queue(struct task_queue *queue_ptr);

void enqueue_task(struct task_queue *queue_ptr, Task *task_ptr);

Task* dequeue_task(struct task_queue *queue_ptr);

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

int task_count = 0;
int done_task_count = 0;
int phase_task_count[] = {0, 0, 0};
 
void submitTask(Task *task_ptr);

void execute_task(Task *task_ptr);

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

double euclidean_distance_squared(data_vec_t *test_vec, data_vec_t *train_vec);

void sorted_insert(data_vec_t *test_vec, data_vec_t *train_vec, 
                   double distance, int k_max);

void split_data_set(data_set_t* src, data_set_t* dest, int B);

void classify(data_vec_t *data_vec_ptr, int k, int total_classes);
 
void free_list(struct list_head *anchor);

int compute_nearest_neighbors(struct args_neighbor *args);

int compute_classifcations(struct args_classify *args);

int evaluate_classifcations(struct args_score *args);

void compute_quality(struct args_quality *args);

void inc_phase_task_count(int phase) {
    pthread_mutex_lock(&mutex_phase_task);
    phase_task_count[phase]++;
    pthread_mutex_unlock(&mutex_phase_task);
}

void dec_phase_task_count(int phase) {
    pthread_mutex_lock(&mutex_phase_task);
    phase_task_count[phase]--;
    pthread_mutex_unlock(&mutex_phase_task);
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
    // printf("fileName: %s, N: %ld, k_max: %d, B: %d, n_threads: %d\n", fileName, N, k_max, B, n_threads);

    // create worker threads
    thread_pool_t *thread_pool = malloc(sizeof(thread_pool_t));
    thread_pool_init(thread_pool, n_threads);

    // init task_queues
    init_queue(&open_tasks);
    init_queue(&done_tasks);

    // read file contents
    FILE *file;
    file = fopen(fileName, "r");
    long N_max;
    int vec_dim;
    int total_classes;
    readInputHeader(file, &N_max, &vec_dim, &total_classes);
    // printf("N_max: %ld, vec_dim: %d, class_count: %d\n", N_max, vec_dim, total_classes);

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
            struct args_neighbor *args_ptr = malloc(sizeof(struct args_neighbor));
            args_ptr->phase = 0;
            args_ptr->test_vec_ptr = data_vec_ptr;
            args_ptr->sub_sets_ptr = &sub_sets;
            args_ptr->k_max = k_max;
            args_ptr->B = B;
            args_ptr->test_set_idx = i;
            thread_pool_enqueue(thread_pool, &compute_nearest_neighbors, args_ptr);
            inc_phase_task_count(args_ptr->phase);
        }
    }
    while (phase_task_count[0] > 0) {
        Task *task_ptr = thread_pool_wait(thread_pool);
        struct args_neighbor *args = task_ptr->args;
        if (args->phase != 0) {
            // printf("fast thread phase: %d\n", args->phase);
            pthread_mutex_lock(&mutex_done_task);
            enqueue_task(&done_tasks, task_ptr);
            done_task_count++;
            pthread_mutex_unlock(&mutex_done_task);
            continue;
        }
        dec_phase_task_count(0);
        data_vec_t *test_vec_ptr = args->test_vec_ptr;
        free(task_ptr->args);
        free(task_ptr);
        struct args_classify *args_ptr = malloc(sizeof(struct args_classify));
        args_ptr->phase = 1;
        args_ptr->test_vec_ptr = test_vec_ptr;
        args_ptr->k_max = k_max;
        args_ptr->total_classes = total_classes;
        thread_pool_enqueue(thread_pool, &compute_classifcations, args_ptr);
        inc_phase_task_count(args_ptr->phase);
    }
    int *correct_classifications_k = calloc(k_max, sizeof(int));
    while (phase_task_count[1] > 0) {
        Task *task_ptr = thread_pool_wait(thread_pool);
        struct args_classify *args = task_ptr->args;
        if (args->phase != 1) {
            // printf("fast thread phase: %d\n", args->phase);
            pthread_mutex_lock(&mutex_done_task);
            enqueue_task(&done_tasks, task_ptr);
            done_task_count++;
            pthread_mutex_unlock(&mutex_done_task);
            continue;
        }
        dec_phase_task_count(1);
        data_vec_t *test_vec_ptr = args->test_vec_ptr;
        free(task_ptr->args);
        free(task_ptr);
        struct args_score *args_ptr = malloc(sizeof(struct args_score));
        args_ptr->phase = 2;
        args_ptr->test_vec_ptr = test_vec_ptr;
        args_ptr->k_max = k_max;
        args_ptr->correct_classifications_k_ptr = &correct_classifications_k;
        thread_pool_enqueue(thread_pool, &evaluate_classifcations, args_ptr);
        inc_phase_task_count(args_ptr->phase);
    }
    for (int i = 0; i < phase_task_count[2]; i++) {
        Task *task_ptr = thread_pool_wait(thread_pool);
        struct args_score *args = task_ptr->args;
        free(task_ptr->args);
        free(task_ptr);
    }
    double *class_qual_k = calloc(k_max, sizeof(double));
    for (int k = 0; k < k_max; k++) {
        struct args_quality *args_ptr = malloc(sizeof(struct args_quality));
        args_ptr->k = k;
        args_ptr->correct = correct_classifications_k[k];
        args_ptr->total = N;
        args_ptr->result_ptr = &class_qual_k;
        thread_pool_enqueue(thread_pool, &compute_quality, args_ptr);
    }
    while (done_task_count > 0) {
        Task *task_ptr = thread_pool_wait(thread_pool);
        struct args_score *args = task_ptr->args;
        free(task_ptr->args);
        free(task_ptr);
    }
    int k_opt = 0;
    double best_classification = 0.0;
    for (int k = 0; k < k_max; k++) {
        double class_qual = class_qual_k[k];
        printf("%d %g\n", k, class_qual);
        if (class_qual >= best_classification) {
            best_classification = class_qual;
            k_opt = k;
        }
    }
    printf("%d\n", k_opt);

    // results of these 3 phases can be stored in a single data structure, containing the data vectors and their classes,
    // as well as information about the k_max nearest neighbors and the classification.

    for (int i = 0; i < N; i++) {
        data_vec_t *data_vec = data_set.data[i];
        // maybe this can go
        for (int j = 0; j < k_max; j++) {
            free(data_vec->classifications_ptr[j]);
        }
        free(data_vec->classifications_ptr);
        free(data_vec->vec.values);
        free_list(&data_vec->neighbors->head);
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

void init_queue(struct task_queue *queue_ptr) {
    list_init(&queue_ptr->head);
}

void enqueue_task(struct task_queue *queue_ptr, Task *task_ptr) {
    struct task_queue *new_element_ptr = malloc(sizeof(struct task_queue));
    new_element_ptr->task_ptr = task_ptr;
    list_add_tail(&new_element_ptr->head, &queue_ptr->head);
}

Task* dequeue_task(struct task_queue *queue_ptr) {
    struct task_queue *element = (struct task_queue*) queue_ptr->head.next;
    Task *task_ptr = element->task_ptr;
    list_del(&element->head);
    free(element);
    return task_ptr;
}

void *start_thread(void* args) {
    while (1) {
        pthread_mutex_lock(&mutex_open_task);
        while (task_count == 0) {
            pthread_cond_wait(&cond_open_task, &mutex_open_task);
        }
        // // dequeue and execute task
        Task *task_ptr = dequeue_task(&open_tasks);
        task_count--;
        pthread_mutex_unlock(&mutex_open_task);
        execute_task(task_ptr);
    }
}

void submitTask(Task *task_ptr) {
    pthread_mutex_lock(&mutex_open_task);
    enqueue_task(&open_tasks, task_ptr);
    task_count++;
    pthread_mutex_unlock(&mutex_open_task);
    pthread_cond_signal(&cond_open_task);
}

void execute_task(Task *task_ptr) {
    task_ptr->function(task_ptr->args);
    pthread_mutex_lock(&mutex_done_task);
    enqueue_task(&done_tasks, task_ptr);
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
        // this doesnt fix my problem completely but maybe the problem can be ignored
        pthread_detach(thread_pool->threads[i]);
    }
}

// pass pointer to function with args pointer, enqueue it in task list and signalize worker thread
// args might contain a set of indices to data vectors to calculate
// data type of arg can change depending on computation phase
void thread_pool_enqueue(thread_pool_t* thread_pool, void *(*function) (void *), void* args) {
    Task *task_ptr = malloc(sizeof(Task));
    task_ptr->function = function;
    task_ptr->args = args;    
    submitTask(task_ptr);
}

Task* thread_pool_wait(thread_pool_t* thread_pool) {
    pthread_mutex_lock(&mutex_done_task);
    while (done_task_count == 0) {
        pthread_cond_wait(&cond_done_task, &mutex_done_task);
    } 
    Task *task_ptr = dequeue_task(&done_tasks);
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
        data_vec_ptr->completed_phases = 0;

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

double euclidean_distance_squared(data_vec_t *test_vec_ptr, data_vec_t *train_vec_ptr) {
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
            if (distance == next->dist) list_add(&new->head, current->next);
            else list_add_tail(&new->head, current->next);
            return;
        } else {
            current = current->next;
        }
    }
}

void classify(data_vec_t *data_vec_ptr, int k, int total_classes) {
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
    struct classification *classification_ptr = malloc(sizeof(struct classification));
    classification_ptr->class = winner_class;
    data_vec_ptr->classifications_ptr[k-1] = classification_ptr;
}

void free_list(struct list_head *anchor) {
    struct list_head *current = anchor->next;
    do {
        struct list_head *next = current->next;
        free(list_del(current));
        current = next; 
    } while (current != anchor);
}

// return the phase?
int compute_nearest_neighbors(struct args_neighbor *args_ptr) {
    int B = args_ptr->B;
    int test_set_idx = args_ptr->test_set_idx;
    data_set_t **sub_sets_ptr = args_ptr->sub_sets_ptr;
    data_set_t *sub_sets = *sub_sets_ptr;
    int k_max = args_ptr->k_max;
    data_vec_t *test_vec_ptr = args_ptr->test_vec_ptr;
    for (int k = 0; k < B; k++) {
        if (k == test_set_idx) continue;
        data_set_t training_set = sub_sets[k];
        for (long l = 0; l < training_set.size; l++) {
            data_vec_t *train_vec_ptr = training_set.data[l];
            double distance = euclidean_distance_squared(test_vec_ptr, train_vec_ptr);
            sorted_insert(test_vec_ptr, train_vec_ptr, distance, k_max);
        }
    }
    test_vec_ptr->completed_phases++;
    return 0;
}

// return the phase?
int compute_classifcations(struct args_classify *args) {
    data_vec_t *test_vec_ptr = args->test_vec_ptr;
    int k_max = args->k_max;
    int total_classes = args->total_classes;
    test_vec_ptr->classifications_ptr = malloc(k_max * sizeof(struct classification*));
    // for each k
    for (int k = 1; k <= k_max; k++) {
        classify(test_vec_ptr, k, total_classes);
    }
    test_vec_ptr->completed_phases++;
    return 1; 
}
 
// return the phase?
int evaluate_classifcations(struct args_score *args) {
    data_vec_t *data_vec_ptr = args->test_vec_ptr;
    int k_max = args->k_max;
    int *args_classification_ptr = *args->correct_classifications_k_ptr;
    for (int k = 0; k < k_max; k++) {
        int correct_class = data_vec_ptr->class;
        struct classification *classification = data_vec_ptr->classifications_ptr[k];
        if (classification->class == correct_class) {
            pthread_mutex_lock(&mutex_correct_classifications);
            args_classification_ptr[k]++;
            pthread_mutex_unlock(&mutex_correct_classifications);
        }
    }
    return 2; 
}

void compute_quality(struct args_quality *args) {
    double *result = *args->result_ptr;
    int k = args->k;
    result[k] = (double) args->correct / (double) args->total;
}
