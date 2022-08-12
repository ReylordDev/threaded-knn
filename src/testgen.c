//
// Created by Jonas Hinz on 21.07.22.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double
randn (double mu, double sigma)
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}

double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int
main(int argc, char *argv[]) {
    if(argc < 4) {
        printf("count, dimensions, classes are required arguments\n");
        exit(EXIT_FAILURE);
    }
    char *p;
    long count = strtol(argv[1], &p, 10);
    long dim = strtol(argv[2], &p, 10);
    long classes = strtol(argv[3], &p, 10);

    FILE *fd = fopen("testdata.txt", "wb");
    if(!fd) {
        printf("couldnt open file\n");
        exit(EXIT_FAILURE);
    }

    fprintf(fd, "%ld %ld %ld\n", count, dim, classes);

    srand(time(NULL));
    double *class_vectors[classes];
    double *vectors = malloc(classes*dim*sizeof(double));
    for(int i = 0; i < classes; i++) {
        class_vectors[i] = vectors+(i*dim);
        for(int j = 0; j < dim; j++) {
            class_vectors[i][j] = randfrom(-3.0, 3.0);
        }
    }
    for(int i = 0; i < classes; i++) {
        printf("class: %d \n", i);
        for(int j = 0; j < dim; j++) {
            printf("%lf ", class_vectors[i][j]);
        }
        printf("\n");
    }
    for (long i = 0; i < count; i++) {
        int class = (rand() % ((classes-1) - 0 + 1)) + 0;
        double randvector[dim];
        for(int j = 0; j < dim; j++) {
            randvector[j] = randn(class_vectors[class][j], 0.75);
            fprintf(fd, "%lf ", randvector[j]);
        }
        fprintf(fd, "%d\n", class);
    }
    printf("finished\n");
    fclose(fd);
}
