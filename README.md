# Sequential Strassen Matrix Multiplication Algorithm (for parallelism)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlirezaAzadbakht/sequential-strassen-matrix-multiplication-algorithm/blob/main/sequential_strassen_matrix_multiplication_algorithm.ipynb)

As you may already know Strassen Matrix Multiplication Algorithm is a recursive algorithm that uses a divide and conquer manner to solve matrix multiplication problem with a time complexity of $O(n^{log_2^7})=O(n^{2.807...})$ which is better than $O(n^3)$ native method.

 I faced the problem of parallelizing the Strassen algorithm for Cuda GPU in the Parallel Algorithm course's project, and generally parallelizing recursive algorithms is not efficient in GPU and you can't achieve much speedup, so the workaround this efficiency problem I came up with is the idea of trying to find the final formula of each element in resulting matrix and after that, I can assign each formula of each element to a GPU thread and voilà solving Strassen algorithm without recursion calls.
 
To do so first we solve the Strassen algorithm in a recursive manner but instead of doing the operations in any step we try to generate the formula for that step and add it to the final formula string after that we have a `LOOK_UP` variable that has all the elements formula then we generate Cuda Kernel from this variable.

The `code_generator.py` generate a Cuda kernel that you can easily run on a Cuda GPU.
 
For example, the formula to generate the [0, 0] element of the final matrix is as follows:

    if x == 0  and y == 0:C[x, y] = (((((((((a[0] + a[10]) + (a[5] + a[15])) * ((b[0] + b[10]) + (b[5] + b[15]))) + ((a[5] + a[15]) * ((b[4] + b[14]) - (b[0] + b[10])))) - (((a[0] + a[10]) + (a[1] + a[11])) * (b[5] + b[15]))) + (((a[1] + a[11]) - (a[5] + a[15])) * ((b[4] + b[14]) + (b[5] + b[15])))) + (((((a[10] + a[15]) * ((b[8] - b[0]) + (b[13] - b[5]))) + (a[15] * ((b[12] - b[4]) - (b[8] - b[0])))) - ((a[10] + a[11]) * (b[13] - b[5]))) + ((a[11] - a[15]) * ((b[12] - b[4]) + (b[13] - b[5]))))) - ((((((a[0] + a[2]) + (a[5] + a[7])) * (b[10] + b[15])) + ((a[5] + a[7]) * (b[14] - b[10]))) - (((a[0] + a[2]) + (a[1] + a[3])) * b[15])) + (((a[1] + a[3]) - (a[5] + a[7])) * (b[14] + b[15])))) + ((((((a[2] - a[10]) + (a[7] - a[15])) * ((b[8] + b[10]) + (b[13] + b[15]))) + ((a[7] - a[15]) * ((b[12] + b[14]) - (b[8] + b[10])))) - (((a[2] - a[10]) + (a[3] - a[11])) * (b[13] + b[15]))) + (((a[3] - a[11]) - (a[7] - a[15])) * ((b[12] + b[14]) + (b[13] + b[15])))))
