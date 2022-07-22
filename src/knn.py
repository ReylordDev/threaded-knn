from io import TextIOWrapper
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
from sklearn.model_selection import KFold 
import numpy as np
import sys
import typing

def main():
    if len(sys.argv) != 6:
        print("invalid arguments")
        return(1)
    file_name: str = sys.argv[1]
    N: int = int(sys.argv[2])
    k_max: int = int(sys.argv[3])
    B: int = int(sys.argv[4])
    n_thread: int = int(sys.argv[5])
     
    print(f'file_name: {file_name}, N: {N}, k_max: {k_max}, '
          f'B: {B}, n_thread: {n_thread}')
           
           
    file: TextIOWrapper = open(file_name, 'r')      
    header: str = file.readline()
    header_args = header.split(' ')
    N_max: int = int(header_args[0])
    dimensions: int = int(header_args[1])
    class_count: int = int(header_args[2])
    print(f'N_max: {N_max}, dimensions: {dimensions}, '
          f'class_count: {class_count}')
    if (N_max < N):
        N = N_max
            
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
    max_score = 0
    winner = 0
    for k in range(1, k_max+1):
        kf =  KFold(n_splits=B)
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)
            scores.append(knn.score(X_test, Y_test))
        avg = np.average(scores)
        print(f'k: {k - 1}, average: {round(avg, 4)}')
        if avg >= max_score:
            max_score = avg
            winner = k - 1
    print(winner)   

 
if __name__ == "__main__":
    exit(main())
