#include "knn_main.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

int main (int argc, char *argv[])
{
    char *fileName = argv[1];
    long N = strtol(argv[2], NULL, 10);
    int k = (int) strtol(argv[3], NULL, 10);
    double values[] = {10.0}
    int prediction = predict_sample(values, N, k, fileName);
    return 0;
}
