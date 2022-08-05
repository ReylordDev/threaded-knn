#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "knn_main.h"

int main (int argc, char *argv[])
{
    int thread_count =  (int) strtol(argv[1], NULL, 10);
    thread_pool_t *thread_pool = malloc(sizeof(thread_pool_t));
    thread_pool_init(thread_pool, thread_count);
    return 0;
}
//command: gcc -Wall -O3 -g -o tests/thread_pool_init tests/test_thread_pool_init.c tests/test_source.c -pthread -lpthread -lm
