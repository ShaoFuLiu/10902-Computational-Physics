import pandas as pd
import numpy as np
import math as math
from numpy import linalg as LA
from sklearn import preprocessing

def Spin05(n, BC):
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

        arr.append(x)## arr[2*i-2] = Six; i=1,2,3...
        arr.append(z)## arr[2*i-1] = Siz; i=1,2,3...

    Single = 0 ## Single site value
    Inter = 0  ## Interaction site value

    for i in range(n-1):
        I = i*2
        Inter += np.matmul(arr[I+1],arr[I+3])
        if (BC == 'PBC' and i+1 == n-1):
            Inter += np.matmul(arr[-1],arr[1]) ## Last interacion part (Snz*S1z) for PBC

    for i in range(n):
        Single += arr[i*2]

    H = Inter ## Wirte down hamiltonion

    return H

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

sz = ([[0.5,0],[0,-0.5]])
n = 2
J = 1
h = 0.5
BC = "OBC"
Phi = [[1, 2], [3, 4]]
Phi_2 = np.reshape(Phi, (4, 1))
# normalize_phi = normalization(phi)
# Phi = normalize_phi.reshape((2, 2, 2))
# print(sz)
# print(Phi)

for i in range(n):
    Phi = np.tensordot(sz,Phi, axes=(1,n-1)) ## SumSz1'(SumSz2'(Sz1 * Sz2 * Phi))
print(Phi.reshape(4, 1))

H = Spin05(n, BC)
print(H,'\n',Phi_2)
Vec = np.tensordot(H, Phi_2, axes=(1,0))
print(Vec) ## H |Phi>