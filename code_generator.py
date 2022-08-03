import numpy as np
import argparse
import os

base_code = """
import numpy as np
# Numba supports CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model.
# so we use numba to access cuda 
from numba import cuda, types, float32

# just in time compiler is a low-level entry point to the CUDA features in Numba
@cuda.jit
def cuda_matrix_multiplication(A, B, C):
    # each thread in each block is the number of columns in matrices
    thread_per_block = 4
    # define shared arrays with type of float32 for operations
    a = cuda.shared.array(shape=(thread_per_block, thread_per_block), dtype=float32)
    b = cuda.shared.array(shape=(thread_per_block, thread_per_block), dtype=float32)
    # getting block corrdiantes
    x, y = cuda.grid(2)
    a = A
    b = B
    cuda.syncthreads()
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    block_per_grid = cuda.gridDim.x    # blocks per grid

    # check if return matrix shapes are correct otherwise return
    if x >= C.shape[0] and y >= C.shape[1]:
        return
"""

def init_matrix(M):
    a = np.array(range(M**2)).astype('object')
    for i in range(M**2):
        a[i] = f'a[{i}]'
    a = a.reshape(M,M)

    b = np.array(range(M**2)).astype('object')
    for i in range(M**2):
        b[i] = f'b[{i}]'
    b = b.reshape(M,M)
    return a, b

def code_generator(M, LOOK_UP):
    os.remove('generated_code.py')
    first = True
    with open("generated_code.py", "a") as myfile:
        myfile.write(f"{base_code}\n")

    for i in range(M):
        for j in range(M):
            if first:
                out = f'    if x == {i} and y == {j}:C[x, y] = {LOOK_UP[i,j]}'
                first = False
            else:
                out = f'    elif x == {i} and y == {j}:C[x, y] = {LOOK_UP[i,j]}'
            with open("generated_code.py", "a") as myfile:
                    myfile.write(f"{out}\n")

def matrix_multiplication(A, B):
    C = np.zeros((A.shape[0], B.shape[1])).astype('object')
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i][j] = ''
            for k in range(A.shape[0]):
                if C[i][j] == '':
                    C[i][j] =  f'({A[i][k]} * {B[k][j]})'
                else:
                    C[i][j] =  f'({C[i][j]} + ({A[i][k]} * {B[k][j]}))'
    return C

def add(A, B):
    C = np.zeros((A.shape[0], A.shape[1])).astype('object')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i][j] =  f'({A[i][j]} + {B[i][j]})'
    return C

def sub(A, B):
    C = np.zeros((A.shape[0], A.shape[1])).astype('object')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i][j] =  f'({A[i][j]} - {B[i][j]})'
    return C

def strassen(x, y):
    if len(x) == 1:
        return matrix_multiplication(x, y)
 
    a, b, c, d = x[:x.shape[0]//2, :x.shape[1]//2], x[:x.shape[0]//2, x.shape[1]//2:], x[x.shape[0]//2:, :x.shape[1]//2], x[x.shape[0]//2:, x.shape[1]//2:]
    e, f, g, h = y[:y.shape[0]//2, :y.shape[1]//2], y[:y.shape[0]//2, y.shape[1]//2:], y[y.shape[0]//2:, :y.shape[1]//2], y[y.shape[0]//2:, y.shape[1]//2:]
 
    p1 = strassen(a, sub(f, h)) 
    p2 = strassen(add(a, b), h)       
    p3 = strassen(add(c, d), e)       
    p4 = strassen(d, sub(g, e))       
    p5 = strassen(add(a, d), add(e, h))       
    p6 = strassen(sub(b, d), add(g, h)) 
    p7 = strassen(sub(a, c), add(e, f)) 

    c11 = add(sub(add(p5 , p4), p2) , p6 )
    c12 = add(p1 , p2)          
    c21 = add(p3 , p4 )          
    c22 = sub(sub(add(p1 , p5 ), p3) , p7) 

    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    return c

def main(args):
    a, b = init_matrix(args.size)
    LOOK_UP = strassen(a, b)
    code_generator(args.size, LOOK_UP)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=8)
    main(parser.parse_args())