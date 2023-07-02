#!/usr/bin/env python
# coding: utf-8

# Implementing faster Cholesky Decomposition algorithm on Python.

# In[1]:


import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    z = 0
    s = 1

    for c in range(n):
        if c == z + s:
            R = L[c:n, z:c]
            S = np.dot(R, R.T)

            for i in range(c, n):
                for j in range(c, n):
                    A[i, j] -= S[i-c, j-c]

            z = c

        L[c, c] = (A[c, c])

        for k in range(z, c):
            L[c, c] -= L[c, k] ** 2

        L[c, c] = np.sqrt(L[c, c])

        for i in range(c + 1, n):
            L[i, c] = A[i, c]

            for k in range(z, c):
                L[i, c] -= L[i, k] * L[c, k]

            L[i, c] /= L[c, c]

    return L


# Accurate Cholesky decomposition using NumPy
def accurate_cholesky(A):
    return np.linalg.cholesky(A)

# Calculate Frobenius norm of the matrix
def frobenius_norm(matrix):
    return np.linalg.norm(matrix, 'fro')

# Calculate the error between two matrices
def calculate_error(A, B):
    return frobenius_norm(A - B)

# Example usage
A = np.array([[4, 1, 2], [1, 5, 3], [2, 3, 6]], dtype=float)

# Cholesky decomposition using the provided algorithm
L = cholesky_decomposition(A)

# Accurate Cholesky decomposition using NumPy
L_acc = accurate_cholesky(A)

# Calculate the error between the two solutions
error = calculate_error(L, L_acc)

print("Lower triangular matrix L (Cholesky decomposition):")
print(L)
print("Lower triangular matrix L (Accurate Cholesky decomposition):")
print(L_acc)
print("Error between the two solutions: {:.8e}".format(error))


# In[ ]:





# In[ ]:




