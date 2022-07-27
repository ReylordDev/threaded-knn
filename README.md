# threaded-knn
C implementation of cross-validated knn for operating systems course

# Compile
`gcc -Wall -O3 -g -o knn_main knn_main.c -pthread -lpthread -lm`

# TODO / ideas
* If I'm done and don't need the normal data_set then refactor to initialize subsets directly from input file
* check with big data if long conversions are bad
* rename file name in comments and printaf
* there are small number errors compared to examples
  * incorrect counting of correct class in low ks?
* refactor classification
* write header file to get direct access to methods for tests
* calculate a test by hand and write the classes

