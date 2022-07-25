from array import ArrayType
from io import TextIOWrapper
from numpy._typing import ArrayLike, NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
import numpy as np
import sys

 
def readInputHeader(file: TextIOWrapper) -> tuple[int, int, int]:
    header: str = file.readline()
    header_args = header.split(' ')
    N_max: int = int(header_args[0])
    dimensions: int = int(header_args[1])
    class_count: int = int(header_args[2])
    return N_max, dimensions,  class_count
     
def readInputData(file: TextIOWrapper, N, 
                  dimensions: int) -> tuple[NDArray, NDArray]:
    data: list[list[np.double]] = []
    classes: list[int] = []
    for i in range(N):
        line = file.readline()
        line_content = line.split(' ')
        vec = []
        for j in range(dimensions):
            vec.append(np.double(line_content[j]))
        data.append(vec)
        classes.append(int(line_content[dimensions]))
    X = np.array(data)
    Y = np.array(classes)
    file.close()
    return X, Y
 

def knn(argv) -> tuple[int, dict[int, float]]:
    file_name: str = argv[1]
    N: int = int(argv[2])
    k_max: int = int(argv[3])
    B: int = int(argv[4])
    n_thread: int = int(argv[5])
     
    print(f'{file_name}, {N}, {k_max}, '
          f'{B}, {n_thread}')
           
    file: TextIOWrapper = open(file_name, 'r')      
    N_max, dimensions, class_count = readInputHeader(file)
    if (N_max < N):
        N = N_max

    X, Y = readInputData(file, N, dimensions)
            
    max_score = 0
    winner = 0
    scores = {}
    for k in range(1, k_max+1):
        kf =  KFold(n_splits=B)
        performances = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            performances.append(knn.score(X_test, Y_test))
        avg = np.average(performances)
        scores[k] = round(avg, 4)
        if avg >= max_score:
            max_score = avg
            winner = k
    return winner, scores 

def main(argv):
    if len(argv) != 6:
        print("invalid arguments")
        return 1
    knn(argv)
    return 0
 
if __name__ == "__main__":
    exit(main(sys.argv))
