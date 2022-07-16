#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

struct arg_struct {
    int argc;
    char **argv;
};

// error if printf or puts fails
void handleOutputError() {
    exit(1);
}

// print usage if user gives invalid arguments
void printUsage() {
    int ret = printf("USAGE: knn_main <input_file> <N> <k_max> <B> <n_threads>\n");
    if (ret < 0) handleOutputError();
}

void* printArgs(void *args){
    struct arg_struct *arguments = (struct arg_struct *) args;
    int argc = arguments->argc;
    char **argv = arguments->argv;
    for (int i = 1; i < argc; ++i) {
        printf("%s\n", argv[i]);
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        printUsage();
        return 1;
    }
    pthread_t newthread;

    struct arg_struct args;
    args.argc = argc;
    args.argv = argv;
    pthread_create(&newthread, NULL, printArgs, (void *)&args);
    pthread_join(newthread, NULL);
    return(0);
}