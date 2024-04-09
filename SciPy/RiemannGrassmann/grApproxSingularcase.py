import numpy as np
from . import grAux
from . import grExp
from . import grLog
from matplotlib import pyplot as plt

#------------------------------------------------------------------------------
# Approximation of the singular case
# Input arguments      
#              n = dimension of the Grassmann manifold
#              p = dimension of the Grassmann manifold
#              N = number of repetitions
#              r = number of singular values
#              T = number of points to test over, log spaced
# Output:
#          Plot
#------------------------------------------------------------------------------

def test(n = 1000, p = 200, N = 10, r = 1, T = np.logspace(-20, -1, 100)):
    np.random.seed(seed=0)

    # T = np.logspace(-20, 0, 100); # probably should be adjustable from arguments
    # T = np.logspace(-20, -1, 100); # above crashes in numpy because of unconverged svd

    lenT = T.size;
    SubspaceErrorLog1 = np.zeros((N, lenT));
    SubspaceErrorLog2 = np.zeros((N, lenT));
    SubspaceErrorLog3 = np.zeros((N, lenT));

    pio2 = np.pi/2.0

    for k in range(N):
        # Create random Stiefel representative U0 with orthogonal completion U0perp
        X = np.random.rand(n,n);
        Q0, _ = np.linalg.qr(X);
        U0 = Q0[:,:p];
        U0perp = Q0[:,p:n];
        # Create a random tangent vector with r largest singular values of pi/2
        B = np.random.rand(n-p,p);
        Delta = U0perp @ B;
        Q, _, V = np.linalg.svd(Delta, full_matrices=False);
        S = lambda t : np.diag(np.sort(np.hstack((pio2*np.ones(r), pio2*np.random.rand(p-r))))[::-1])*t;
        Delta = lambda t: Q @ S(t) @ V;

        for i in range(lenT):
            t = 1-T[i];
            
            # Calculate the subspace associated with the tangent vector
            Deltat = Delta(t);
            U1 = grExp.Grassmann_Exp(U0,Deltat);
            
            # Calculate the new and the standard log
            DeltaLog, _ = grLog.GrassmannLogOneSVD(U0,U1);
            DeltaLog_standard = grLog.GrassmannLog_standard(U0,U1);
            
            # Project the result of the standard log algorithm onto the
            # horizontal space
            DeltaLog_standardproj = DeltaLog_standard - U0@(U0.T@DeltaLog_standard);
            
            # Calculate the subspaces associated with the log results
            U1Log = grExp.Grassmann_Exp(U0,DeltaLog);
            U1Log_standardproj = grExp.Grassmann_Exp(U0,DeltaLog_standardproj);
            U1Log_standard = grExp.Grassmann_Exp(U0,DeltaLog_standard);
            
            # Calculate the subspace errors
            SubspaceErrorLog1[k,i] = grAux.subspaceDist(U1,U1Log);
            SubspaceErrorLog2[k,i] = grAux.subspaceDist(U1,U1Log_standardproj);
            SubspaceErrorLog3[k,i] = grAux.subspaceDist(U1,U1Log_standard);


    # Plot the results on a log-log plot
    plt.xscale('log')
    plt.yscale('log')

    for k in range(N):
        plt.plot(T,SubspaceErrorLog1[k,:], '*', color=[0, 0.4470, 0.7410]);
        plt.plot(T,SubspaceErrorLog2[k,:], 'x', color=[0.8500, 0.3250, 0.0980]);
        plt.plot(T,SubspaceErrorLog3[k,:], '+', color=[0.9290, 0.6940, 0.1250]);

    plt.xlabel(r'$\tau$')
    plt.ylabel('Error by subspace distance')
    plt.legend(['New log algorithm', 'Standard log alg. (with horiz. projection)', 'Standard log algorithm'])

    print("approxSingularcase")

