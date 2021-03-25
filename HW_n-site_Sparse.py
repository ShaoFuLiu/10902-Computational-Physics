import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh

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

    H = Inter ## Wirte down Hamiltonion

    return H

def Sn_Operation(S, Phi, n): # return operated Phi
    phi_Snz = np.swapaxes(Phi, 0, n)
    phi_Snz = np.tensordot(S, phi_Snz, axes=(1,0))
    phi_Snz = np.swapaxes(phi_Snz, 0, n)

    return phi_Snz

BC = "OBC"
sz = np.array([[0.5,0],[0,-0.5]])
N = 4 # number of sites
elements = np.power(2, N) # number of element
Phi = np.random.randint(elements, size=(2, 2, 2, 2)) # Creat random vector Phi
Phi_total = 0

for m in range(N-1): # For PBC(N),For OBC(N-1)
    n = m+1
    Phi_SnSm = Sn_Operation(sz, Sn_Operation(sz, Phi, n), m)
    Phi_total += Phi_SnSm

## Direct product H |Phi>
H = Spin05(N, BC)
Phi_V = np.reshape(Phi, (elements, 1)) # Reshape of Phi
Vec = np.tensordot(H, Phi_V, axes=(1,0))

print('Two ways diff = \n',Vec-Phi_total.reshape((elements, 1)))