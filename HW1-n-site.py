import pandas as pd 
import numpy as np
from numpy import linalg as LA

def Spin05(n, J, h, BC): ## Output ground state eigvec/eigval and spin matrix array 
    sx = ([[0,0.5],[0.5,0]])
    sz = ([[0.5,0],[0,-0.5]])
    arr = []
    for i in range(n):
        site = i+1
        if (site == 1):
            x = np.kron(sx,np.eye(2))
            z = np.kron(sz,np.eye(2))
            for j in range(n-2):
                x = np.kron(x,np.eye(2))
                z = np.kron(z,np.eye(2))
        if (site == 2):
            x = np.kron(np.eye(2),sx)
            z = np.kron(np.eye(2),sz)
            for j in range(n-2):
                x = np.kron(x,np.eye(2))
                z = np.kron(z,np.eye(2))
        if (site >= 3):
            x = np.kron(np.eye(2),np.eye(2))
            z = np.kron(np.eye(2),np.eye(2))
            for j in range(n-2):
                if (j+3 == site):
                    x = np.kron(x,sx)
                    z = np.kron(z,sz)
                elif (j+3 != site):
                    x = np.kron(x,np.eye(2))
                    z = np.kron(z,np.eye(2))
        arr.append(x)
        arr.append(z)

    Single = 0 ## Single site value
    Inter = 0  ## Interaction site value

    for i in range(n-1):
        I = i*2
        Inter += np.matmul(arr[I+1],arr[I+3])
        if (BC == 'PBC' and i+1 == n-1):
            Inter += np.matmul(arr[-1],arr[1]) ## last interacion part (Snz*S1z) for PBC
            
    for i in range(n):
        Single += arr[i*2]

    ## Hamiltonion 
    H = -J*(Inter) - h*(Single)
    w, v = LA.eigh(H)

    return w ,v ,arr

def S05_Expctation_Sz(n, w, v, arr): ## Output ground state Sz Expectation value 
    Exp_Sz = 0
    EVe = v[:,0]
    TEVe = np.transpose(EVe)
    for i in range(n): ## sum (VT * Si * V)
        site = i+1
        Exp_Sz += np.matmul(np.matmul(TEVe ,arr[2*site-1]) ,EVe) 
    Exp_Sz = Exp_Sz/n 

    return Exp_Sz

def S05_Expctation_Sx(n, w, v, arr): ## Output ground state Sx Expectation value 
    Exp_Sx = 0
    EVe = v[:,0]
    TEVe = np.transpose(EVe)
    for i in range(n): ## VT * S * V
        site = i+1
        Exp_Sx += np.matmul(np.matmul(TEVe ,arr[2*site-2]) ,EVe) 
    Exp_Sx = Exp_Sx/n  ## sum<Si> / n

    return Exp_Sx

## Initial condition 
n = 4
J = 1
h = 0.5
BC = 'PBC'

## Calculation
w ,v ,arr = Spin05(n, J, h, BC)
EVal = w[0] 
print('Eigenvalue = ',EVal)
EVec = v[:,0]
print('Eigenvector = ',EVec)
Exp_Sz = S05_Expctation_Sz(n, w, v, arr)
print('<Sz> = ',Exp_Sz)
Exp_Sx = S05_Expctation_Sx(n, w, v, arr)
print('<Sx> = ',Exp_Sx)
