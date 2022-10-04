from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from PCA import encode_decode_pca, pca
import syntheticdata
from K_means import KM2, KM3, KM4, KM5, yhat2, yhat3, yhat4, yhat5

def one_hot_encoding(t):
    t_one_hot = []
    t_max = max(t) + 1 
    for i in t:
        vec = np.zeros(t_max)
        vec[i] = 1  
        t_one_hot.append(vec)
    t_one_hot = np.array(t_one_hot)
    return t_one_hot

# I
X, y = syntheticdata.get_iris_data()
_ , P = pca(X, 2)

logreg = LogisticRegression()
logreg.fit(P, y)
# print(accuracy_score(y, yhat2))

# II 
yhat2_hot = one_hot_encoding(yhat2)
yhat3_hot = one_hot_encoding(yhat3)
yhat4_hot = one_hot_encoding(yhat4)
yhat5_hot = one_hot_encoding(yhat5)

logreg2 = LogisticRegression()
logreg3 = LogisticRegression()
logreg4 = LogisticRegression()
logreg5 = LogisticRegression()

logreg2.fit(yhat2_hot, y)
logreg3.fit(yhat3_hot, y)
logreg4.fit(yhat4_hot, y)
logreg5.fit(yhat5_hot, y)


print("k \t||\t Accuracy")
print(2, "\t||\t", logreg2.score(yhat2_hot, y))
print(3, "\t||\t", logreg3.score(yhat3_hot, y))
print(4, "\t||\t", logreg4.score(yhat4_hot, y))
print(5, "\t||\t", logreg5.score(yhat5_hot, y))

k_arr = [2,3,4,5]
accs = [logreg2.score(yhat2_hot, y), logreg3.score(yhat3_hot, y), 
        logreg4.score(yhat4_hot, y), logreg5.score(yhat5_hot, y)]

plt.plot(k_arr, accs, "o--")
plt.show()