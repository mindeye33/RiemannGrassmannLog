import numpy as np
import scipy


#------------------------------------------------------------------------------
# Exponential map on Grassmann manifold
# Input arguments      
#          U0    = base point on Gr(n,p)
#          Delta = tangent vector in T_U0 Gr(n,p)
# Output arguments
#          U1    = Exp^{Gr}_U0(Delta),
#------------------------------------------------------------------------------
def Grassmann_Exp(U0, Delta):
    Q, Sigma, V = np.linalg.svd(Delta, full_matrices=False)
    cosSigma = np.diag(np.cos(Sigma))
    sinSigma = np.diag(np.sin(Sigma))

    U1 = (U0 @ V.T @ cosSigma + Q @ sinSigma) @ V

    return U1





#------------------------------------------------------------------------------
# This is a naive (and slow) way to exponentiate on Grassmann manifold
# Input arguments      
#          U0    = base point on Gr(n,p)
#          Delta = tangent vector in T_U0 Gr(n,p)
# Output arguments
#          U1    = Exp^{Gr}_U0(Delta),
#------------------------------------------------------------------------------
def Grassmann_Exp_Naive(U0, Delta):
    n, p = U0.shape

    U0_span = U0 @ U0.T
    U0_eval, U0_evec = np.linalg.eigh(U0_span)
    U0_ortho = U0_evec[:, :n-p]

    B = np.linalg.pinv(U0_ortho) @ Delta
    mat_to_exp = np.zeros((n,n))
    mat_to_exp[p:, :p] = B
    mat_to_exp[:p, p:] = -B.T
    U1 = np.block([U0, U0_ortho]) @ scipy.linalg.expm(mat_to_exp) @ np.eye(n,p)
    return U1