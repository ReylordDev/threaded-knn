#include <stdio.h>
#include <stdlib.h>

// error if printf or puts fails
void handleOutputError() {
    exit(1);
}

void printUsage() {
    int ret = printf("USAGE: knn_main <input_file> <N> <k_max> <B> <n_threads>\n");
    if (ret < 0) handleOutputError();
}

int main(int argc, char** argv) {
    if (argc != 5) {
        printUsage();
        return 1;
    }
    printf("Hello world\n");
    return(0);
}