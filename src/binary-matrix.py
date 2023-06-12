import numpy as np

Mu = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1],
               [1, 1, 0, 0],
               [0, 0, 1, 1]])

Mv = np.array([[1, 1, 0, 0],
               [0, 1, 1, 0],
               [1, 0, 0, 1],
               [0, 0, 1, 1]])

def create_binary_diffusion_matrices(Mu, Mv):
    m = Mu.shape[0]
    n = Mu.shape[1]
    I_l = np.eye(m, dtype=int)
    I_m = np.eye(n, dtype=int)
    Muv = np.logical_and(Mu, Mv).astype(int)
    Mvl = np.logical_xor(I_l, Muv).astype(int)
    
    BM1 = np.block([[I_m, Mu],
                    [Mv, Mvl]])
    
    BM2 = np.block([[Mu, I_m],
                    [Mvl, Mv]])
    
    BM3 = np.block([[Mvl, Mv],
                    [Mu, I_m]])
    
    BM4 = np.block([[Mv, Mvl],
                    [I_m, Mu]])
    
    return BM1, BM2, BM3, BM4

BM1, BM2, BM3, BM4 = create_binary_diffusion_matrices(Mu, Mv)

primary_matrices = [BM1, BM2, BM3, BM4]

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