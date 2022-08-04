#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "knn_main.h"

int main (int argc, char *argv[])
{
    char *fileName = argv[1];
    long N = strtol(argv[2], NULL, 10);
    int k = (int) strtol(argv[3], NULL, 10);
    double values[] = {7.0, 7.0};
    int prediction = predict_sample(values, N, k, fileName);
    printf("k: %d, performance: [%d]\n", k, prediction);
    return 0;
}
