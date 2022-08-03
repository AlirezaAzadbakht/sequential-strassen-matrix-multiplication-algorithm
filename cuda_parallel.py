from numba import cuda, types, float32
import numpy as np
from generated_code import cuda_matrix_multiplication

M = 4
N = 4

# prepare 2 random matrix of size 32*32 and one zeros matrix of size 32*32 for results
a = np.random.rand(M,N).flatten()
b = np.random.rand(N,M).flatten()
c = np.zeros((M, M))

# convert host arrays to device arrays
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# set size of blocks and grid here we have one grid and 32*32 block size 
block_size = (N,N)
grid_size = (int(M/N),int(M/N))

# calling cuda funtion with grid and block size and device arrays
cuda_matrix_multiplication[grid_size, block_size](d_a, d_b, d_c)

# copy back result to host arrays
c = d_c.copy_to_host()
print(c)