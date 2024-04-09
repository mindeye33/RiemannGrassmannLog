import numpy as np
import scipy

from . import grExp

#------------------------------------------------------------------------------
# create a random data set 
# U0, U1 on Gr(n,p),
#  Delta on T_U Gr(n,p) with norm 'dist',
# which is also the Riemannian distance dist(U0,U1)
#
# input arguments
#        (n,p) = dimension of the Grassmann matrices
#        dist  = Riemannian distance between the points U0,U1
#                that are to be created
# Output arguments
#             U0 = base point on Gr(n,p)
#             U1 = end point on Gr(n,p)
#          Delta = tangent vector
#------------------------------------------------------------------------------
def create_random_Grassmann_data(n, p, dist=1e-1):
    #create random Grassmann matrix:
    # np.random.seed(seed=0)
    X =  np.random.rand(n,p)
    U0, R = scipy.linalg.qr(X, overwrite_a=True,\
                                lwork=None,\
                                mode='economic',\
                                pivoting=False,\
                                check_finite=True)


    # borrowing Stiefel's creat_random_Steifel_data function
    # create pseudo-random tangent vector in T_U0 St(n,p)
    # A = np.random.rand(p,p)
    # A = A-A.T   # "random" p-by-p skew symmetric matrix
    # T = np.random.rand(n,p)
    # Delta = np.dot(U0,A) + T-np.dot(U0,np.dot(U0.transpose(),T))
    # Delta = U0 @ A + T - U0 @ U0.T @ T

    # For Grassmann, only I-UU^T is needed:
    T = np.random.rand(n,p)
    Delta = T - U0 @ U0.T @ T

    Delta = dist*Delta
    # 'project' Delta onto Gr(n,p) via the Grassmann exponential
    U1 = grExp.Grassmann_Exp(U0, Delta)
    return U0, U1, Delta



#------------------------------------------------------------------------------
# Grassmann distance
# Input arguments      
#             U0 = base point on Gr(n,p)
#             U1 = end point on Gr(n,p)
# Output arguments
#           dist = Grassmann distance in terms of the norm of canonical angles
#
#------------------------------------------------------------------------------
def subspaceDist(U0, U1):
    M = U0.T @ U1
    # Q1, S1, R1 = np.linalg.svd(M) # change to svals only
    S1 = scipy.linalg.svdvals(M)
    for k in range(S1.size):
        if S1[k] > 1:
            S1[k] = 1.0
    
    theta = np.arccos(S1).real
    # replace with S1[S1<1.0]
    dist = np.linalg.norm(theta)
    return dist

