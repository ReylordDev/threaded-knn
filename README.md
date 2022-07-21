# threaded-knn
C implemententation of cross-validated knn for operating systems course

# Compile
`gcc -Wall -O3 -g -o knn_main knn_main.c -pthread -lpthread -lm`

# TODO / ideas
* If I'm done and don't need the normal data_set then refactor to initialize subsets directly from input file
* check with big data if long conversions are bad
* rename file name in comments and prints
* valgrind needs to be run
* save addresses or something in neighbor list
* for statements have inc on different sides
