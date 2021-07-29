import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

## Output eigvec/eigval and spin matrix array of Spin-1/2
def Spin05(n, J, h, BC):
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
        ## arr[2*i-1] = Siz; arr[2*i-2] = Six; i=1,2,3...
        arr.append(x)
        arr.append(z)

    Single = 0 ## Single site value
    Inter = 0  ## Interaction site value

    for i in range(n-1):
        I = i*2
        Inter += np.matmul(arr[I+1],arr[I+3])
        if (BC == 'PBC' and i+1 == n-1):
            Inter += np.matmul(arr[-1],arr[1]) ## Last interacion part (Snz*S1z) for PBC

    for i in range(n):
        Single += arr[i*2]

    H = -J*(Inter) - h*(Single) ## Write down hamiltonion
    w, v = LA.eigh(H) ## Diagonal H and find eigenvalue/eigenvector

    return w ,v ,arr

## Output "ground state" Sz Expectation value
def S05_Expectation_Sz(n, w, v, arr):
    Exp_Sz = 0
    EVe = v[:,0]
    TEVe = np.transpose(EVe)
    for i in range(n): ## VT * Si * V
        site = i+1
        Exp_Sz += np.matmul(np.matmul(TEVe ,arr[2*site-1]) ,EVe)
    Exp_Sz = Exp_Sz/n ## Sum<Siz> / n

    return Exp_Sz

## Output "ground state" Sx Expectation value
def S05_Expectation_Sx(n, w, v, arr):
    Exp_Sx = 0
    EVe = v[:,0]
    TEVe = np.transpose(EVe)
    for i in range(n): ## VT * Sx * V
        site = i+1
        Exp_Sx += np.matmul(np.matmul(TEVe ,arr[2*site-2]) ,EVe)
    Exp_Sx = Exp_Sx/n  ## Sum<Six> / n

    return Exp_Sx

## Initial condition
n = 6
J = 1
h = 0.5
BC = 'PBC'

## Calculation
w ,v ,arr = Spin05(n, J, h, BC)
EVal = w[0]
print('Grond state eigenvalue = ',EVal ,'\n')
EVec = v[:,0]
print('Ground state eigenvector = ',EVec, '\n')
Exp_Sz = S05_Expectation_Sz(n, w, v, arr)
print('<Sz> = ',Exp_Sz, '\n')
Exp_Sx = S05_Expectation_Sx(n, w, v, arr)
print('<Sx> = ',Exp_Sx)

## Initial condition
ns = [8,10]
J = 1
h = np.linspace(0,1,num=11)
BC = 'OBC'
for i in range(len(ns)):
    n = ns[i]
    Szs = []
    hs = []
    for j in range(len(h)):
        w ,v ,arr = Spin05(n, J, h[j], BC)
        EVal_0 = w[0]
        EVal_1 = w[1]
        Exp_Sz = S05_Expectation_Sz(n, w, v, arr)
        #EVec = v[:,0]
        # Szs.append(EVal_1-EVal_0)
        Szs.append(Exp_Sz)
        hs.append(h[j])

    plt.plot(hs, Szs, '-o', markersize = 4, label = 'L=%d' %(n))

# print(Szs)
# print(hs)

plt.xlabel(r'h', fontsize=14)
plt.ylabel(r'$Sz(h)$', fontsize=14)
# plt.xlim(3,32)
# plt.ylim(0.001, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.title(r'Sz vs h, J = %d' %(J), fontsize=12)
plt.legend(loc = 'best')
plt.savefig('/home/liusf/10902-Computational-Physics/3.pdf', format='pdf', dpi=4000)
# plt.show()