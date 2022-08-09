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
    long thread;
} Task;

typedef struct {
    data_vec_t *test_vector_ptr;
    data_vec_t *train_vector_ptr;
    int k_max;
} Args_phase_one;

// temp
Task task_queue[256];
int task_count = 0;
Task done_tasks[256];
int done_task_count = 0;


void execute_task(Task *task_ptr);

//temp
void submitTask(Task task) {
    pthread_mutex_lock(&mutex_task);
    task_queue[task_count] = task;
    task_count++;
    pthread_mutex_unlock(&mutex_task);
    pthread_cond_signal(&cond_task);
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

void* test_func(void *arg){

}

// End all worker threads and free all allocated memory
// called on end of program
void thread_pool_shutdown(thread_pool_t* thread_pool);

// #####################MAIN###########################
int main(int argc, char** argv) {
    int n_threads = (int) strtol(argv[1], NULL, 10);

    // create worker threads
    thread_pool_t *thread_pool = malloc(sizeof(thread_pool_t));
    thread_pool_init(thread_pool, n_threads);

    void *args = malloc(sizeof(void));
    
    // nthreads is not actual number
    for (int i = 0; i < n_threads; i++) {
        thread_pool_enqueue(thread_pool, &test_func, &args);
    }
    // nthreads is not actual number
    for (int i = 0; i < n_threads; i++) {
        Task* task_ptr = thread_pool_wait(thread_pool);
        printf("Thread %ld finished executing func: %p\n", task_ptr->thread, task_ptr->function);
    }
    thread_pool_shutdown(thread_pool);
    return(0);
}


void execute_task(Task *task_ptr) {
    task_ptr->thread = pthread_self();
    printf("%ld is executing: func: %p, args: %p\n", task_ptr->thread, task_ptr->function, task_ptr->args);
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
        for (int i = 0; i < task_count - 1; i++) {
            task_queue[i] = task_queue[i + 1];
        }
        task_count--;
        pthread_mutex_unlock(&mutex_task);
        execute_task(&task);
    }
}

// pass pointer to function with args pointer, enqueue it in task list and signalize worker thread
// args might contain a set of indices to data vectors to calculate
// data type of arg can change depending on computation phase
void thread_pool_enqueue(thread_pool_t* thread_pool, void *(*function) (void *), void* args) {
    Task *task_ptr = malloc(sizeof(Task));
    task_ptr->function = function;
    task_ptr->args = args;    
    submitTask(*task_ptr);
}

Task* thread_pool_wait(thread_pool_t* thread_pool) {
    pthread_mutex_lock(&mutex_done_task);
    while (done_task_count == 0) {
        // cond wait
        pthread_cond_wait(&cond_done_task, &mutex_done_task);
    } 
    // this fucks up after moving
    Task task = done_tasks[0];
    Task *task_ptr = malloc(sizeof(Task*));
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
    free(thread_pool);
}
