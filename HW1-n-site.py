import pandas as pd 
import numpy as np
from numpy import linalg as LA

def Spin05(n): ## Output spin matrix
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
    return arr

def S05_EigenValue(n, J, h, BC, arr): ## Output ground state eigenvalue
    Single = 0
    Inter = 0

    for i in range(n-1):
        I = i*2
        Inter += np.matmul(arr[I+1],arr[I+3])
        if (BC == 'PBC' and i+1 == n-1):
            Inter += np.matmul(arr[-1],arr[1]) ## last interacion Snz*S1z
            
    for i in range(n):
        Single += arr[i*2]

    ## Hamiltonion 
    H = -J*(Inter) - h*(Single)
    w, v = LA.eigh(H)

    return w[0]

def S05_EigenVector(n, J, h, BC, arr): ## Output ground state eigenvector
    Single = 0
    Inter = 0

    for i in range(n-1):
        I = i*2
        Inter += np.matmul(arr[I+1],arr[I+3])
        if (BC == 'PBC' and i+1 == n-1):
            #print(i)
            Inter += np.matmul(arr[-1],arr[1]) ## last interacion Snz*S1z
            
    for i in range(n):
        Single += arr[i*2]

    ## Hamiltonion 
    H = -J*(Inter) - h*(Single)
    w, v = LA.eigh(H)
    ## Test H*EVe=EVa*EVe
    #err = np.dot(H, v[:, 0]) - w[0] * v[:, 0]
    return v[:, 0]

def S05_Expctation_Sz(n, J, h, BC, arr): ## Output ground state Sz Expectation value 
    Exp_Sz = 0
    EVe = S05_EigenVector(n, J, h, BC, arr)
    TEVe = np.transpose(EVe)
    for i in range(n):
        site = i+1
        Exp_Sz += np.matmul(np.matmul(TEVe ,arr[2*site-1]) ,EVe)
    Exp_Sz = Exp_Sz/n

    return Exp_Sz

def S05_Expctation_Sx(n, J, h, BC, arr): ## Output ground state Sz Expectation value 
    Exp_Sx = 0
    EVe = S05_EigenVector(n, J, h, BC, arr)
    TEVe = np.transpose(EVe)
    for i in range(n):
        site = i+1
        Exp_Sx += np.matmul(np.matmul(TEVe ,arr[2*site-2]) ,EVe)
    Exp_Sx = Exp_Sx/n

    return Exp_Sx


n = 4
J = 1
h = 0.5
BC = 'PBC'
arr = Spin05(n)
"""
EVa = S05_EigenValue(n, 1, 0.5, 'PBC', arr)
EVe = S05_EigenVector(n, 1, 0.5, 'PBC', arr)
print(EVe)"""
Exp_Sz = S05_Expctation_Sz(n, J, h, BC, arr)
print('<Sz> = ',Exp_Sz)
Exp_Sx = S05_Expctation_Sx(n, J, h, BC, arr)
print('<Sx> = ',Exp_Sx)
