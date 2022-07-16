#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// error if printf or puts fails
void handleOutputError() {
    exit(1);
}

// print usage if user gives invalid arguments
void printUsage() {
    int ret = printf("USAGE: knn_main <input_file> <N> <k_max> <B> <n_threads>\n");
    if (ret < 0) handleOutputError();
}

void* test(void *arg){
    while(1) {
        printf("yo\n");
        sleep(1000);
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printUsage();
        return 1;
    }
    pthread_t newthread;

    pthread_create(&newthread, NULL, test, NULL);
    for (int i = 1; i < argc; ++i) {
        printf("%s\n", argv[i]);
    }
    pthread_join(newthread, NULL);
    return(0);
}