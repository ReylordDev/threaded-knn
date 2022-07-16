cmake_minimum_required(VERSION 3.22)
project(threaded_knn C)

set(CMAKE_C_STANDARD 11)

add_executable(threaded_knn
        knn_main.c)
