#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

// buffer for lines of the input file
#define BUFFER_SIZE 1000

struct list_head {
    struct list_head *next, *prev;
};

typedef struct {
    double *values;
    int dims;
} vec_t;

// list of neighbors of a data_vec with their distance
struct neighbor_info {
    struct list_head head;
    vec_t *vec_ptr; // shares address with corresponding data_vec_t
    double dist;
};

typedef struct {
    vec_t vec;
    int class;
    struct neighbor_info *neighbors;
    int *classifications;
} data_vec_t;

typedef struct {
    long size;
    data_vec_t **data;
} data_set_t;

typedef struct {
    pthread_t *threads;
    int count;
} thread_pool_t;

typedef struct {
    void *(*function) (void *);
    void *args;
} Task;

// args for the k_max nearest neighbor calculation (phase 1)
struct args_neighbor {
    int phase;
    data_vec_t *test_vec_ptr;
    data_set_t **sub_sets_ptr;
    int k_max;
    int test_set_idx;
    long B;
};

// args for the classification (phase 2)
struct args_classify {
    int phase;
    data_vec_t *test_vec_ptr;
    int k_max;
    int total_classes;
};

// args for the evaluation of the classifications (phase 3.1)
struct args_score {
    int phase;
    data_vec_t *test_vec_ptr;
    int k_max;
    int **correct_classifications_ptr;
};

// args for the classification quality (phase 3.2)
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

/* initialize "shortcut links" for empty task queue */
void init_queue(struct task_queue *queue_ptr);

/* add a new task to the back of the queue */
void enqueue_task(struct task_queue *queue_ptr, Task *task_ptr);

/* pop the first task from the queue */
Task* dequeue_task(struct task_queue *queue_ptr);

/* initialize "shortcut links" for empty list */
void list_init(struct list_head *head);

/* insert new entry after the specified head */
void list_add(struct list_head *new, struct list_head *head);

/* insert new entry before the specified head */
void list_add_tail(struct list_head *new, struct list_head *head);

/* deletes entry from list and reinitialize it, returns pointer to entry */
struct list_head* list_del(struct list_head *entry);
 
/* free ressources in list */
void free_list(struct list_head *anchor);

/* begin thread life-cycle */
void* start_thread(void* args);

/* thread-safe opening a new task*/
void submitTask(Task *task_ptr);

/* thread-safe executing and closing a task*/
void execute_task(Task *task_ptr);

/* initialize thread pool with thread count threads */
void thread_pool_init(thread_pool_t* thread_pool, int thread_count);

/* register a new function with args in the task list and signal worker thread */
void thread_pool_enqueue(thread_pool_t* thread_pool, void *(*function) (void *), void* args);

/* wait for any thread to complete a task and return the task */
Task* thread_pool_wait(thread_pool_t* thread_pool);

/* end all worker threads and free thread data */
void thread_pool_shutdown(thread_pool_t* thread_pool);

/* parse first line of input file */
void readInputHeader(FILE *file, long *N_max_ptr, int *vec_dim_ptr, int *class_count_ptr);

/* create data set of input file */
void readInputData(FILE *file, data_set_t *data_set, int dims);

/* split up a data set into B almost equally sized data sets */
void split_data_set(data_set_t* src, data_set_t* dest, long B);

/* calculate the squared euclidean distance */
double euclidean_distance_squared(data_vec_t *test_vec, data_vec_t *train_vec);

/* insert vector into sorted neighbor list */
void sorted_insert(data_vec_t *test_vec, data_vec_t *train_vec, 
                   double distance, int k_max);

/* find most common class among k closest neighbors */
void classify(data_vec_t *data_vec_ptr, int k, int total_classes);

/* task function for calculating the neearest neighbors of a vector */
void compute_nearest_neighbors(struct args_neighbor *args);

/* task function for calculating the most common class among k neighbors */
void compute_classifcations(struct args_classify *args);

/* task function for adding to the correct classification counter for k classifications */
void evaluate_classifcations(struct args_score *args);

/* task function for computing the over all classificatin quality for a given k */
void compute_quality(struct args_quality *args);

/* threadless implementation */
void sequential_implementation(long N, long B, data_set_t *data_set_ptr, data_set_t **sub_sets_ptr, int k_max, int total_classes);

/* queue lengths */
int open_task_count = 0;
int done_task_count = 0;
 
pthread_mutex_t mutex_open_task = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_open_task = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_done_task = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_done_task = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_correct_classifications = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char** argv) {
    if (argc != 6) {
        printf("USAGE: knn_main <input_file> <N> <k_max> <B> <n_threads>\n");
        return 1;
    }
    // read command line arguments
    char *fileName = argv[1];
    long N = strtol(argv[2], NULL, 10);
    int k_max = (int) strtol(argv[3], NULL, 10);
    long B = strtol(argv[4], NULL, 10);
    int n_threads = (int) strtol(argv[5], NULL, 10);

    // create worker threads
    thread_pool_t *thread_pool;
    if (n_threads > 0) {
        thread_pool = malloc(sizeof(thread_pool_t));
        thread_pool_init(thread_pool, n_threads);
    }

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
    if (N_max < N) N = N_max;

    // prepare main data set
    data_set_t data_set;
    data_vec_t **data = malloc(N * sizeof(data_vec_t*));
    data_set.data = data;
    data_set.size = N;

    readInputData(file, &data_set, vec_dim);
    // main data set contains all N vectors
    fclose(file);

    // split dataset into B sub sets
    data_set_t *sub_sets = malloc(B * sizeof(data_set_t));
    split_data_set(&data_set, sub_sets, B);

    if (n_threads <= 0) {
        if (n_threads == 0) sequential_implementation(N, B, &data_set, &sub_sets, k_max, total_classes);
        return(0);
    }

    int patient_tasks = 0;
    for (long i = 0; i < B; i++) {
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
            patient_tasks++;
        }
    }
    int *correct_classifications_k = calloc(k_max, sizeof(int));
    while (patient_tasks > 0) {
        patient_tasks--;
        Task *task_ptr = thread_pool_wait(thread_pool);
        struct args_neighbor *args = task_ptr->args;
        data_vec_t *test_vec_ptr = args->test_vec_ptr;
        switch (args->phase) {
            case 0:
            {
                struct args_classify *args_ptr = malloc(sizeof(struct args_classify));
                args_ptr->phase = 1;
                args_ptr->test_vec_ptr = test_vec_ptr;
                args_ptr->k_max = k_max;
                args_ptr->total_classes = total_classes;
                thread_pool_enqueue(thread_pool, &compute_classifcations, args_ptr);
                patient_tasks++;
                break;
            }
            case 1:
            {
                struct args_score *args_ptr = malloc(sizeof(struct args_score));
                args_ptr->phase = 2;
                args_ptr->test_vec_ptr = test_vec_ptr;
                args_ptr->k_max = k_max;
                args_ptr->correct_classifications_ptr = &correct_classifications_k;
                thread_pool_enqueue(thread_pool, &evaluate_classifcations, args_ptr);
                patient_tasks++;
                break;
            }
            default:
            {
                break;
            }
        }
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
        patient_tasks++;
    }
    free(correct_classifications_k);
    while (patient_tasks > 0) {
        Task *task_ptr = thread_pool_wait(thread_pool);
        free(task_ptr->args);
        free(task_ptr);
        patient_tasks--;
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
    free(class_qual_k);

    // free all data vector memory
    for (long i = 0; i < N; i++) {
        data_vec_t *data_vec = data_set.data[i];
        free(data_vec->classifications);
        free(data_vec->vec.values);
        free_list(&data_vec->neighbors->head);
        free(data_vec->neighbors);
        free(data_vec);
    }
    free(data_set.data);

    // free sub sets memory
    for (long i = 0; i < B; i++) {
        free(sub_sets[i].data);
    }
    free(sub_sets);
    thread_pool_shutdown(thread_pool);
    return(0);
}

void list_init(struct list_head *head) {
    head->next = head;
    head->prev = head;
}

void list_add(struct list_head *new, struct list_head *head) {
    new->next = head->next;
    new->prev = head;
    head->next->prev = new;
    head->next = new;
}

void list_add_tail(struct list_head *new, struct list_head *head) {
    head->prev->next = new;
    new->prev = head->prev;
    head->prev = new;
    new->next = head;
}

struct list_head* list_del(struct list_head *entry) {
    entry->prev->next = entry->next;
    entry->next->prev = entry->prev;
    list_init(entry);
    return entry;
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

void free_list(struct list_head *anchor) {
    struct list_head *current = anchor->next;
    do {
        struct list_head *next = current->next;
        free(list_del(current));
        current = next; 
    } while (current != anchor);
}

void *start_thread(void* args) {
    while (1) {
        pthread_mutex_lock(&mutex_open_task);
        // wait for a new task
        while (open_task_count == 0) {
            pthread_cond_wait(&cond_open_task, &mutex_open_task);
        }
        // dequeue and execute task
        Task *task_ptr = dequeue_task(&open_tasks);
        open_task_count--;
        pthread_mutex_unlock(&mutex_open_task);
        execute_task(task_ptr);
    }
}

void submitTask(Task *task_ptr) {
    pthread_mutex_lock(&mutex_open_task);
    enqueue_task(&open_tasks, task_ptr);
    open_task_count++;
    pthread_mutex_unlock(&mutex_open_task);
    pthread_cond_signal(&cond_open_task);
}

void execute_task(Task *task_ptr) {
    task_ptr->function(task_ptr->args);
    pthread_mutex_lock(&mutex_done_task);
    // enqueue done task into the done queue
    enqueue_task(&done_tasks, task_ptr);
    done_task_count++;
    pthread_mutex_unlock(&mutex_done_task);
    pthread_cond_signal(&cond_done_task);
}

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

void thread_pool_shutdown(thread_pool_t* thread_pool) {
    for (int i = 0; i < thread_pool->count; i++) {
        pthread_cancel(thread_pool->threads[i]);
    }
    free(thread_pool->threads);
    free(thread_pool);
}

void readInputHeader(FILE *file, long *N_max_ptr, int *vec_dim_ptr, int *class_count_ptr) {
    int header_arg_count = 3;
    long headerArguments[header_arg_count];
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
    // line buffer
    char buffer[BUFFER_SIZE];
    for (long i = 0; i < data_set->size; i++) {
        data_vec_t *data_vec_ptr = malloc(sizeof(data_vec_t));
        data_vec_ptr->vec.dims = dims;

        // initialize the neighbor list
        struct neighbor_info *neighbors_ptr = malloc(sizeof(struct neighbor_info));
        neighbors_ptr->dist = -1;
        list_init(&neighbors_ptr->head);
        data_vec_ptr->neighbors = neighbors_ptr;
         
        // initialize the vector values 
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

void split_data_set(data_set_t* src, data_set_t* dest, long B) {
    ldiv_t div = ldiv(src->size, B);
    long B_offsets = div.rem;
    long vectors_per_subset = div.quot;
    long data_start_index = 0;
    for (long i = 0; i < B; i++) {
        data_set_t sub_set;

        // calculate sub set size
        long size = vectors_per_subset + (i < B_offsets);
        sub_set.size = size;

        // copy data vector pointers from data set into sub set
        data_vec_t **sub_set_data = malloc(size * sizeof(data_vec_t*));
        for (long j = 0; j < size; j++) {
            sub_set_data[j] = src->data[j + data_start_index];
        }
        // start each sub-set at the correct index in the data-set
        data_start_index += size;

        sub_set.data = sub_set_data;
        dest[i] = sub_set;
    }
}

double euclidean_distance_squared(data_vec_t *test_vec_ptr, data_vec_t *train_vec_ptr) {
    double dist = 0;
    int dims = test_vec_ptr->vec.dims;
    for (int i = 0; i < dims; i++) {
        dist += pow(test_vec_ptr->vec.values[i] - train_vec_ptr->vec.values[i], 2);
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
            // insert if smaller distance or not k_max entries in list
            struct neighbor_info *new = malloc(sizeof (struct neighbor_info));
            new->dist = distance;
            new->vec_ptr = &train_vec->vec;
            if (distance >= next->dist) list_add(&new->head, current->next);
            else list_add_tail(&new->head, current->next);
            return;
        } else {
            current = current->next;
        }
    }
}

void classify(data_vec_t *data_vec_ptr, int k, int total_classes) {
    int *class_count = calloc(total_classes, sizeof(int));
    // traverse k closest neighbors and increment class counts
    struct list_head *anchor = &data_vec_ptr->neighbors->head;
    struct list_head *current = anchor;
    for (int i = 0; i <= k; i++) {
        struct neighbor_info *next = (struct neighbor_info *) current->next;
        if (current->next == anchor) {
            continue;
        }
        data_vec_t *neighbor_vec_ptr = (data_vec_t *) next->vec_ptr;
        class_count[neighbor_vec_ptr->class]++;
        current = current->next;
    }
    // calculate most common class
    int max_count = 0;
    int winner_class = 0;
    for (int i = 0; i < total_classes; i++) {
        if (class_count[i] >= max_count) {
            max_count = class_count[i];
            winner_class = i;
        } 
    }
    data_vec_ptr->classifications[k] = winner_class;
    free(class_count);
}

void compute_nearest_neighbors(struct args_neighbor *args_ptr) {
    long B = args_ptr->B;
    int test_set_idx = args_ptr->test_set_idx;
    data_set_t *sub_sets = *args_ptr->sub_sets_ptr;
    int k_max = args_ptr->k_max;
    data_vec_t *test_vec_ptr = args_ptr->test_vec_ptr;
    for (long i = 0; i < B; i++) {
        if (i == test_set_idx) continue;
        data_set_t training_set = sub_sets[i];
        for (long j = 0; j < training_set.size; j++) {
            data_vec_t *train_vec_ptr = training_set.data[j];
            double distance = euclidean_distance_squared(test_vec_ptr, train_vec_ptr);
            sorted_insert(test_vec_ptr, train_vec_ptr, distance, k_max);
        }
    }
}

void compute_classifcations(struct args_classify *args) {
    data_vec_t *test_vec_ptr = args->test_vec_ptr;
    int k_max = args->k_max;
    int total_classes = args->total_classes;
    test_vec_ptr->classifications = malloc(k_max * sizeof(int));
    for (int k = 0; k < k_max; k++) {
        classify(test_vec_ptr, k, total_classes);
    }
}
 
void evaluate_classifcations(struct args_score *args) {
    data_vec_t *data_vec_ptr = args->test_vec_ptr;
    int k_max = args->k_max;
    int *args_classification_ptr = *args->correct_classifications_ptr;
    for (int k = 0; k < k_max; k++) {
        int correct_class = data_vec_ptr->class;
        int classification = data_vec_ptr->classifications[k];
        if (classification == correct_class) {
            pthread_mutex_lock(&mutex_correct_classifications);
            args_classification_ptr[k]++;
            pthread_mutex_unlock(&mutex_correct_classifications);
        }
    }
}

void compute_quality(struct args_quality *args) {
    double *result = *args->result_ptr;
    int k = args->k;
    result[k] = (double) args->correct / (double) args->total;
}

void sequential_implementation(long N, long B, data_set_t *data_set_ptr, data_set_t **sub_sets_ptr, int k_max, int total_classes) {
    data_set_t *sub_sets = *sub_sets_ptr;
    for (long i = 0; i < B; i++) {
        data_set_t test_set = sub_sets[i];
        for (long j = 0; j < test_set.size; j++) {
            data_vec_t *data_vec_ptr = test_set.data[j];
            struct args_neighbor *args_ptr = malloc(sizeof(struct args_neighbor));
            args_ptr->test_vec_ptr = data_vec_ptr;
            args_ptr->sub_sets_ptr = sub_sets_ptr;
            args_ptr->k_max = k_max;
            args_ptr->B = B;
            args_ptr->test_set_idx = i;
            compute_nearest_neighbors(args_ptr);
            free(args_ptr);
        }
    }
    for (long i = 0; i < B; i++) {
        data_set_t test_set = sub_sets[i];
        for (long j = 0; j < test_set.size; j++) {
            data_vec_t *data_vec_ptr = test_set.data[j];
            struct args_classify *args_ptr = malloc(sizeof(struct args_classify));
            args_ptr->test_vec_ptr = data_vec_ptr;
            args_ptr->k_max = k_max;
            args_ptr->total_classes = total_classes;
            compute_classifcations(args_ptr);
            free(args_ptr);
        }
    }
    int *correct_classifications_k = calloc(k_max, sizeof(int));
    for (long i = 0; i < B; i++) {
        data_set_t test_set = sub_sets[i];
        for (long j = 0; j < test_set.size; j++) {
            data_vec_t *data_vec_ptr = test_set.data[j];
            struct args_score *args_ptr = malloc(sizeof(struct args_score));
            args_ptr->test_vec_ptr = data_vec_ptr;
            args_ptr->k_max = k_max;
            args_ptr->correct_classifications_ptr = &correct_classifications_k;
            evaluate_classifcations(args_ptr);
            free(args_ptr);
        }
    }
    double *class_qual_k = calloc(k_max, sizeof(double));
    for (int k = 0; k < k_max; k++) {
        struct args_quality *args_ptr = malloc(sizeof(struct args_quality));
        args_ptr->k = k;
        args_ptr->correct = correct_classifications_k[k];
        args_ptr->total = N;
        args_ptr->result_ptr = &class_qual_k;
        compute_quality(args_ptr);
        free(args_ptr);
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

    for (long i = 0; i < N; i++) {
        data_vec_t *data_vec = data_set_ptr->data[i];
        free(data_vec->classifications);
        free(data_vec->vec.values);
        free_list(&data_vec->neighbors->head);
        free(data_vec->neighbors);
        free(data_vec);
    }
    free(correct_classifications_k);
    free(class_qual_k);
    free(data_set_ptr->data);
    for (long i = 0; i < B; i++) {
        free(sub_sets[i].data);
    }
    free(sub_sets);
}
