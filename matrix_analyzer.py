import numpy as np
"""
This program computes the eigenvalues and condition number (k) of the matrices used to benchmark the HHL algorithm 
in the Jupyter Notebook. In order to work well, matrices must be positive-definite (positive eigenvalues) hermitian.
Kappa should be kept as low as possible to maximize performance 
"""

#print only 4 decimals
np.set_printoptions(precision=4, suppress=True)

def analyze_matrix(name, matrix):
    #calculate Eigenvalues
    evals = np.linalg.eigvalsh(matrix)
    
    #calculate Condition Number 'Kappa'
    #kappa=lambda_max/lambda_min
    kappa = np.max(np.abs(evals)) / np.min(np.abs(evals))
    
    #print results
    print(f"{name:<25} | Kappa = {kappa:<6.2f} | Eigenvalues = {evals}")


A1 = np.array([[1.0, -1/3], 
               [-1/3, 1.0]])
analyze_matrix("A1 (Fractional)", A1)

A2 = np.array([[1.5, 0.5], 
               [0.5, 1.5]])
analyze_matrix("A2 (Integer)", A2)

A3 = np.array([[2.0, 1.0], 
               [1.0, 2.0]])
analyze_matrix("A3 (High Contrast)", A3)

main_diag = 2 * np.ones(4)
off_diag  = -1 * np.ones(3)
A4 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
analyze_matrix("A4 (Poisson 4x4)", A4)

A5 = np.kron(np.eye(2), A2)
analyze_matrix("A5 (Degenerate)", A5)

np.random.seed(42)
R = np.random.rand(4, 4)
A6 = R + R.T + 2*np.eye(4)
analyze_matrix("A6 (Random Dense)", A6)