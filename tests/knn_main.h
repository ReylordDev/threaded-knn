#ifndef KNN_MAIN_H
#define KNN_MAIN_H
#include <pthread.h>

typedef struct {
    // necessary information for managing worker threads
    pthread_t *threads;
} thread_pool_t;

int predict_sample(double *values, long N, int k, char *file_name);
void thread_pool_init(thread_pool_t* thread_pool, int thread_count);

#endif // !KNN_MAIN_H
