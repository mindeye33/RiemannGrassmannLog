import numpy as np
import scipy


#------------------------------------------------------------------------------
# Grassmann Log on Grassmann manifold
# Input arguments      
#             U0 = base point on Gr(n,p)
#             U1 = end point on Gr(n,p)
# Output arguments
#         U1star = Adapted Stiefel representative of U1
#          Delta = Tangent vector in horizontal space at U0, from U0 to U1star
#
# This version of the Grassmann logarithm corresponds
# to Alg. 5.3 of the associated paper.
#------------------------------------------------------------------------------
def GrassmannLog(U0, U1):
    # Step 1: Procrustes
    M = U1.T @ U0
    Q1, S1, R1 = np.linalg.svd(M)
    U1star = U1 @ (Q1 @ R1)

    # Step 2: SVD
    H = U1star - U0 @ ( U0.T @ U1star)
    Q2, S2, R2 = np.linalg.svd(H, full_matrices=False)
    S2[S2>1.0] = 1.0 # avoids NaNs
    Sigma = np.diag(np.arcsin(S2))

    # Step 3: Tangent vector
    Delta = Q2 @ Sigma @ R2

    return Delta, U1star



#------------------------------------------------------------------------------
# Grassmann Log on Grassmann manifold
# Input arguments      
#             U0 = base point on Gr(n,p)
#             U1 = end point on Gr(n,p)
# Output arguments
#          Delta = Tangent vector in horizontal space at U0
#                  from U0 to subspace spanned by U1
#
#
# This implementation of the algorithm follows
#
# P.-A. Absil, R. Mahony, and R. Sepulchre. 
# "Riemannian geometry of Grassmann manifolds with 
#  a view on algorithmic computation."
# Acta Applicandae Mathematica, 80(2):199–220, 2004. 
# doi:%10.1023/B:ACAP.0000013855.14971.91.
#
#------------------------------------------------------------------------------
def GrassmannLog_standard(U0, U1):
    # Step 1: (I-U0*U0')*U1*(U0'*U1)^-1
    M = U0.T @ U1
    # N = U1 @ np.linalg.pinv(M) - U0
    N = (np.eye(U0.shape[0]) - U0 @ U0.T) @ U1 @ np.linalg.pinv(M) # more numerically stable


    # Step 2: SVD
    Q2, S2, R2 = np.linalg.svd(N, full_matrices=False)
    Sigma = np.diag(np.arctan(S2))

    # Step 3: Tangent vector
    Delta = Q2 @ Sigma @ R2

    return Delta



#------------------------------------------------------------------------------
# Grassmann Log on Grassmann manifold
# Input arguments      
#             U0 = base point on Gr(n,p)
#             U1 = end point on Gr(n,p)
# Output arguments
#         U1star = Adapted Stiefel representative of U1
#          Delta = Tangent vector in horizontal space at U0, from U0 to U1star
#
#
# This version of the Grassmann logarithm avoids the computation of
# the SVD of "H = U1star-U0*(U0'*U1star)"
# A short consideration shows that all the required information are 
# already encoded in the SVD of M = U1'*U0
#
# This modification of the Grassmann logarithm of Alg. 5.3 
# corresponds to the remark in Section 5.2 of the paper.
#
#
#------------------------------------------------------------------------------
def GrassmannLogOneSVD(U0, U1):
    # Step 1: Procrustes
    M = U1.T @ U0
    Q1, S1, R1 = np.linalg.svd(M)
    
    # # Reorder the columns #not sure this is needed for scipy
    # Q1 = Q1[:,::-1]
    # R1 = R1[:,::-1]
    # S1 = S1[::-1]

    # Calculate new rep
    U1star = U1 @ Q1

    # Step 2: SVD without actual SVD
    H = U1star - U0 @ ( U0.T @ U1star)
    singvals = np.sqrt(1.0-S1**2.0)

    Q2 = H * (singvals **-1.0)
    Sigma = np.diag(np.arcsin(singvals))

    # Step 3: Tangent vector
    Delta = Q2 @ Sigma @ R1

    return Delta, U1star

