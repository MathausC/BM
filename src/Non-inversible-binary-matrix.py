import numpy as np


Mu = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1],
               [1, 1, 0, 0],
               [0, 0, 1, 1]])

Mv = np.array([[1, 1, 0, 0],
               [0, 1, 1, 0],
               [1, 0, 0, 1],
               [0, 0, 1, 1]])

def create_non_invertible_binary_matrices(Mu, Mv):
    m = Mu.shape[0]
    Im = np.eye(m, dtype=int)
    Mu_Mv = np.logical_and(Mu, Mv).astype(int)

    BNM1 = np.block([[Im, Mu],
                     [Mv, Mu_Mv]])

    BNM2 = np.block([[Mu, Im],
                     [Mu_Mv, Mv]])

    BNM3 = np.block([[Mu_Mv, Mv],
                     [Mu, Im]])

    BNM4 = np.block([[Mv, Mu_Mv],
                     [Im, Mu]])

    return BNM1, BNM2, BNM3, BNM4

BNM1, BNM2, BNM3, BNM4 = create_non_invertible_binary_matrices(Mu, Mv)

primary_matrices = [BNM1, BNM2, BNM3, BNM4]

def binary_matrix_multiplication(X, Y):
    m, n = X.shape
    q = Y.shape[1]
    Z = np.zeros((m, q), dtype=int)

    for i in range(m):
        for j in range(q):
            tmp = 0
            for k in range(n):
                tmp ^= X[i, k] & Y[k, j]
            Z[i, j] = tmp

    return Z

def construct_final_diffusion_matrix(primary_matrices):
    final_matrix = primary_matrices[0]
    for i in range(1, len(primary_matrices)):
        final_matrix = binary_matrix_multiplication(final_matrix, primary_matrices[i])
    
    return final_matrix

final_diffusion_matrix = construct_final_diffusion_matrix(primary_matrices)
print("Final Diffusion Matrix:")
print(final_diffusion_matrix)