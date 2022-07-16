#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#define BUFFER_SIZE 1000

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

void readFile(char *fileName, FILE *file) {
    file = fopen(fileName, "r");
    readLines(file);
    fclose(file);
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
    readFile(fileName, file);

    // split dataset into B sub sets
        // same size if N mod B == 0, otherwise close to same size
    // loop over k from 1 to k_max
        // Rotate through sub sets, setting i as test set and the other B-1 sets as training set
            // calculate which k neighbors are closest using squared euclidean distance
                // use thread pool for this
            // assign most common class among k neighbors to vector
        // calculate classification quality: correct qualifications / all classifications
        // print "%d %g\n", k, class_qual
    // print k with optimal class_qual (no parallelization)

    return(0);
}