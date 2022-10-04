from base64 import encode
from pyexpat.errors import XML_ERROR_BAD_CHAR_REF
from xml.dom import XHTML_NAMESPACE
import numpy as np
import matplotlib.pyplot as plt 
import syntheticdata
from random import randint

def center_data(A):
    N, M = A.shape
    B = np.zeros(shape=(N, M))
    for i in range(M):
        col = A[:, i]
        mean = np.mean(col)
        B[:, i] = col - mean
    
    return B

def compute_covariance_matrix(A): # assumes A is centred
    N = A.shape[0]
    C = 1/N * np.transpose(A) @ A
    return C

def compute_eigenvalue_eigenvectors(A):
    eigval, eigvec = np.linalg.eig(A)

    eigval = eigval.real
    eigvec = eigvec.real 

    return eigval, eigvec

def sort_eigenvalue_eigenvectors(eigval, eigvec):
    indices = np.argsort(eigval)
    indices = indices[::-1] #reverse order from argsort
    
    sorted_eigval = eigval[indices]
    sorted_eigvec = eigvec[:, indices] 
    # sort eigenvectors by the sorting done for eigenvalues, 
    # so that each eigen-pairing is correct

    return sorted_eigval, sorted_eigvec


def pca(A, m):
    N, M = A.shape
    A = center_data(A)
    cov_A = compute_covariance_matrix(A)
    eigval, eigvec = compute_eigenvalue_eigenvectors(cov_A) 
    pca_eigval, pca_eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)
    pca_eigvec = pca_eigvec[:,:m]

    P = np.transpose(pca_eigvec) @ np.transpose(A)

    return pca_eigvec, P.T


# 1.3 how does PCA work?

# UNCOMMENT THE PLOT COMMANDS TO SEE THE PLOTS

# 1.3.1
X = syntheticdata.get_synthetic_data1()

# 1.3.2
#plt.scatter(X[:,0],X[:,1])
#plt.show()


# 1.3.3
# X = center_data(X)
# plt.scatter(X[:,0],X[:,1], color="maroon")
# plt.show()

# 1.3.4
# pca_eigvec, _ = pca(X, m=2)
# first_eigvec = pca_eigvec[0]
# plt.scatter(X[:,0],X[:,1], color="navy")
# x = np.linspace(-5, 5, 1000)
# y = first_eigvec[1]/first_eigvec[0] * x
# plt.plot(x,y, color="red")
# plt.show()

# 1.3.5
# _, P = pca(X, 1)
# plt.scatter(P, np.zeros(shape=P.shape), color="purple")
# plt.show()


# 1.4. when are the results of PCA sensible?

# 1.4.1
X, y = syntheticdata.get_synthetic_data_with_labels1()
# 1.4.2
# plt.scatter(X[:,0],X[:,1],c=y[:,0])
# plt.figure()
# _, P = pca(X, 1)
# plt.scatter(P, np.ones(P.shape[0]),c=y[:,0])
# plt.show()

# 1.4.3 
X, y = syntheticdata.get_synthetic_data_with_labels2()

# 1.4.4
X = center_data(X)
# plt.scatter(X[:,0],X[:,1],c=y[:,0])
#plt.figure()
pca_eigvecs, P = pca(X, 1)
# plt.scatter(P, np.ones(P.shape[0]),c=y[:,0])
# plt.show()
x = np.linspace(-2, 2, 1000)
eig0 = pca_eigvecs[0]
y0 = eig0 * x
eig1 = pca_eigvecs[1]
y1 = eig1 * x

# plt.plot(x, y0, color="blue")
# plt.plot(x, y1, color="red")
# plt.show()

# 1.5

# 1.5.1
X,y = syntheticdata.get_iris_data()

# 1.5.2 & 1.5.3
i, j = randint(0, 3), randint(0, 3)
while (i==j):
    i = randint(0, 3)
#plt.scatter(X[:,i],X[:,j],c=y)
#plt.figure()
_,P = pca(X, 1)
#plt.scatter(P, np.zeros(P.shape[0]),c=y)
#plt.show()

# 1.6

# 1.6.1
X, y, h, w = syntheticdata.get_lfw_data()

# 1.6.2 
# plt.imshow(X[0, :].reshape((h, w)), cmap=plt.cm.gray)
# plt.show()

# 1.6.3
def encode_decode_pca(A, m):
    mean = np.mean(A, axis=0)
    eigvecs, A = pca(A, m)
    A = A.T 

    A_hat = (eigvecs @ A).T + mean

    return A_hat

# 1.6.4
# X_hat = encode_decode_pca(X, 200)

# 1.6.5
# plt.imshow(X_hat[0, :].reshape((h, w)), cmap=plt.cm.gray)
# plt.show()

# 1.6.6
# for i in [1000, 800, 300, 100, 40]:
#     X_hat = encode_decode_pca(X, i)
#     plt.figure()
#     plt.imshow(X_hat[0, :].reshape((h, w)), cmap=plt.cm.gray)
#     plt.savefig(f"1.6.6_{i}.png")

# plt.show()

