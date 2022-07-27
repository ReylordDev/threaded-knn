from numpy import array
import pytest
import subprocess
from src.knn import knn as py_knn
from src.knn import predictSample, readInputData, readInputHeader
 
source_file_path = './src/knn_main.c'
sample_test_file_path = './tests/test_predict_sample.c'

@pytest.fixture
def compile() -> subprocess.CompletedProcess:
    command: str = f'gcc -Wall -O3 -g -o tests/test {source_file_path} -pthread -lpthread -lm'
    return subprocess.run(command.split(' '), capture_output=True)
     
@pytest.fixture
def compile_sample() -> subprocess.CompletedProcess:
    command: str = f'gcc -Wall -O3 -g -o tests/predict_sample {sample_test_file_path} src/knn_main.c -pthread -lpthread -lm'
    return subprocess.run(command.split(' '), capture_output=True)

def create_input_file(N_max: int, dimensions: int, class_count: int, 
                      data) -> str:
    input_file_name = 'tests/test_input.txt'
    file = open(input_file_name, 'w')
    header = f'{N_max} {dimensions} {class_count}\n'
    file.write(header)
    for line in data:
        tuple = line[0]
        tuple_str = ''
        for i in range(dimensions):
            tuple_str += f'{tuple[i]} '
        file.write(f'{tuple_str}{line[1]}\n')
    file.close()
    return file.name
     
def knn(file_name: str, N: int, k_max: int, B: int, n_threads: int):
    command = f'./tests/test {file_name} {N} {k_max} {B} {n_threads}'
    return subprocess.run(command.split(' '), capture_output=True)
     
def python_knn(file_name: str, N: int, k_max: int, B: int, n_threads: int):
    winner, scores = py_knn(f'src/knn.py {file_name} '
                            f'{N} {k_max} {B} {n_threads}'.split(' '))
    print_result(winner, scores)
     
def print_result(winner: int, scores: dict[int, float]):
    if winner != -1:
        print(f'Winner: {winner}')
    for key in scores.keys():
        print(f'k: {key}, performance: {scores[key]}')
         
def parse_result(completed_process: subprocess.CompletedProcess
                 ) -> tuple[int, dict[int, float]]:
    stdout: bytes = completed_process.stdout
    stdout_str = stdout.decode("utf-8")
    lines = stdout_str.split('\n')
    winner = int(lines[len(lines) - 2]) + 1
    score_strs = lines[2: -2]
    scores: dict[int, float]= {}
    for score_str in score_strs:
        score_str_splt = score_str.split(' ')
        k = int(score_str_splt[0]) + 1
        performance = float(score_str_splt[1])
        scores[k] = performance
    return winner, scores


def test_compile(compile):
    ret = compile
    assert ret.returncode == 0
    assert ret.stdout == b''

# TODO: Explore this further.
def test_sample(compile_sample):
    data = [
        ([8.5], 1),
        ([11.0], 0),
        ([12.0], 0),
        ([6.0], 1),
        ([15.0], 1),
        ([16.0], 1),
    ]
    file_name = create_input_file(len(data), len(data[0][0]), 2, data)
    N, k_max, B, n_threads= 6, 5, 6, 1
    file = open(file_name, 'r')
    N_max, dimensions, n_threads = readInputHeader(file)
    if (N_max < N):
        N = N_max
    X_train,  Y_train = readInputData(file, N, dimensions)
    sample = array([[10.0],]) 
    subprocess.run(f'tests/predict_sample {file_name} {N} {1}'.split(' '))
    prediction = predictSample(sample, 2, X_train, Y_train)
    print_result(-1,  prediction)
    subprocess.run(f'tests/predict_sample {file_name} {N} {5}'.split(' '))
    prediction = predictSample(sample, 5, X_train, Y_train)
    print_result(-1,  prediction)

    
     
def test_1(compile):
    data = [
        ([2.0], 1),
        ([4.0], 1),
        ([16.0], 0),
        ([3.0], 1),
        ([5.0], 1),
        ([17.0], 0),
        ([14.0], 0),
        ([15.0], 0),
    ]
    file_name = create_input_file(len(data), len(data[0][0]), 2, data)
    N, k_max, B, n_threads= 8, 7, 8, 1
    ret = knn(file_name, N, k_max, B, n_threads)
    assert ret.returncode == 0
    winner, scores = parse_result(ret)
    print(f'\n{file_name}, {N}, {k_max}, {B}, {n_threads}')
    print_result(winner, scores)
    print('comparison:')
    python_knn(file_name, N, k_max, B, n_threads)

def test_2(compile):
    data = [
        ([9.4, 9.1], 0),
        ([3.0, 3.5], 1),
        ([7.9, 7.5], 0),
        ([3.6, 4.0], 1),
        ([4.5, 4.0], 1),
        ([8.0, 8.8], 0),
        ([9.0, 8.0], 0),
        ([3.2, 4.5], 1),
    ]
    file_name = create_input_file(len(data), len(data[0][0]), 2, data)
    ret = knn(file_name, 8, 7, 8, 1)
    assert ret.returncode == 0
    winner, scores = parse_result(ret)
    print(f'\n{file_name}, 8, 7, 8, 1')
    print_result(winner, scores)

     
def test_example(compile):
    ret = knn('src/hsl_codebook.txt', 10000, 10, 5, 1)
    assert ret.returncode == 0
    winner, scores = parse_result(ret)
    print(f'\nsrc/hsl_codebook.txt, 10000, 10, 5, 1')
    print_result(winner, scores)
    print('comparison:')
    python_knn('src/hsl_codebook.txt', 10000, 10, 5, 1)

