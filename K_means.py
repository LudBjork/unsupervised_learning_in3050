# 2.0.1 - 2.0.3
from sklearn.cluster import KMeans
import numpy as np 
import matplotlib.pyplot as plt 
from PCA import encode_decode_pca, pca
import syntheticdata


X, y = syntheticdata.get_iris_data()

_, P = pca(X,2)

#plt.scatter(P, np.zeros(shape=P.shape), color="maroon")
#plt.show()

# 2.0.4
KM2 = KMeans(2)
yhat2 = KM2.fit_predict(P)

KM3 = KMeans(3)
yhat3 = KM3.fit_predict(P)

KM4 = KMeans(4)
yhat4 = KM4.fit_predict(P)

KM5 = KMeans(5)
yhat5 = KM5.fit_predict(P)



# 2.0.5
# for i in [2,3,4,5]:
#     KM = KMeans(i)
#     KM.fit_predict(P)
#     centres = KM.cluster_centers_
#     plt.figure()
#     plt.scatter(P[:,0], P[:,1], c=y)
#     plt.scatter(centres[:,0], centres[:,1], marker="o", color="lime")
#     plt.title(f"$k={i}$")
#     plt.savefig(f"Kmeans_{i}.png")
# plt.show()
